import JuMP
import MathOptInterface as MOI
import LinearAlgebra
# import Gurobi
import SCIP

function Base.sum(e::Vector{Any})
    return isempty(e) ? 0. : error("cannot sum a Any-element Vector")
end

# OPF model in paper [Aigner/Burlacu_ijoc_2023]
function vdict(x, x_)
    return Dict(x[i] => x_[i] for i in eachindex(x))
end
function ct(uk,tk,ul,tl) # (2)
    return uk * ul * cos(tk - tl), -uk * ul * sin(tk - tl) # ckl, tkl
end # Additional: ckk = uk ^ 2, for any bus k


if true # data
    LD = Dict( # lines
        "from"  =>[1,1,2,2,2,3,4],
        "to"    =>[2,3,3,4,5,4,5],
        "heat_lim" => fill(0.8,7),
        "anglediff_lim" => fill(pi/6,7),
        "r"     =>[2.,8,6,6,4,1,8]/100,
        "x"     =>[6.,24,18,18,12,3,24]/100,
        "half_b"=>[3,2.5,2,2,1.5,1,2.5]/100,
    )
    L = length(LD["x"])
    N = max(maximum(LD["from"]), maximum(LD["to"]))
    GD = Dict(  # generators
        "node" => [1,2], # Generator 1 is at node 1, the former "1" is natural counting, the latter "1" = GD["node"][1]
        "Pmin" => [.1, .1],
        "Pmax" => [2., 2.],
        "Qmin" => [-3., -3.],
        "Qmax" => [3., 3.],
        "c0" =>   [60.,60.],
        "c1" =>   [3.4,3.4],
        "c2" =>   [.004,.004],
    )
    ND = merge( Dict(  # nodes/buses, this part is manually filled
    "type"  => [3,2,1,1,1],
    "P"     => [0.0, 0.2, 0.45, 0.4, 0.6], # demand
    "Q"     => [0.0, 0.1, 0.15, 0.05, 0.1], # demand
    "Vmin"  => [0.9, 0.9, 0.9, 0.9, 0.9],
    "Vmax"  => [1.5, 1.1, 1.1, 1.1, 1.1]
    ),          Dict(  # this part is auto-generated
    "injecting_branch" => [findall(x -> x == n,LD["to"]) for n in 1:N],
    "leaving_branch" => [findall(x -> x == n,LD["from"]) for n in 1:N],
    "G_ind" => [findall(x -> x == n,GD["node"]) for n in 1:N]
    ))
    
    @assert length(ND["P"]) == N
    NG = length(GD["node"])
    Y = [.0im for i in 1:N, j in 1:N] # node admittance matrix
    for b in 1:L
        x,y = LD["from"][b], LD["to"][b]
        l,c = min(x,y), max(x,y)
        Y[l,c] = -1.0 / (LD["r"][b] + LD["x"][b] * im)
        Y[c,l] = Y[l,c]
    end
    for n in 1:N
        for b in [findall(x -> x == n,LD["from"]); findall(x -> x == n,LD["to"])] # δ(n)
            Y[n,n] += LD["half_b"][b] * im # this is a preprocess step
        end
    end
    gkk = real.(LinearAlgebra.diag(Y)) # this should be zero for general lines which has only `b/2` part shunt branch
    bkk = imag.(LinearAlgebra.diag(Y)) # these parameters presents in paper [Aigner/Burlacu_ijoc_2023]
    for n in 1:N
        for b in [findall(x -> x == n,LD["from"]); findall(x -> x == n,LD["to"])] # δ(n)
            Y[n,n] += LD["half_b"][b] * im + 1.0 / (LD["r"][b] + LD["x"][b] * im)
        end
    end # this version is Y in TextBook
    # these 2 lines provide another way to get `gkk` and `bkk`, from "Y" in TextBook derived just above
    # tmp = [Y[n,n] + sum(Y[n,i] for i in 1:N if i != n) for n in 1:N]
    # gkk, bkk = real.(tmp), imag.(tmp) # this is an 
    G,B = real(Y),imag(Y)
end

if true # JuMP modeling
    model = JuMP.Model(SCIP.Optimizer)
    # generation
    JuMP.@variable(model, GD["Pmin"][g] <= pg[g = 1:NG] <= GD["Pmax"][g])
    JuMP.@objective(model, Min, sum(GD["c2"][g] * pg[g] * pg[g] + GD["c1"][g] * pg[g] + GD["c0"][g] for g in 1:NG))
    JuMP.@variable(model, GD["Qmin"][g] <= qg[g = 1:NG] <= GD["Qmax"][g])
    # branch
    JuMP.@variable(model, pkl[1:L])
    JuMP.@variable(model, qkl[1:L])
    JuMP.@variable(model, ckl[1:L])
    JuMP.@variable(model, tkl[1:L])
    # nodes
    JuMP.@variable(model, ND["Vmin"][n]^2 <= ckk[n = 1:N] <= ND["Vmax"][n]^2)
    JuMP.@variable(model, tha[1:N])
    JuMP.@constraint(model, tha[end] == -pi/6) # ref angle
    # constraints
    JuMP.@constraint(model, [n = 1:N], sum([pkl[i] for i in ND["leaving_branch"][n]])
                                        - sum([pkl[i] for i in ND["injecting_branch"][n]])
                                        - sum([pg[g] for g in ND["G_ind"][n]])
                                        == -ND["P"][n]) # 3b
    JuMP.@constraint(model, [n = 1:N],sum([qkl[i] for i in ND["leaving_branch"][n]]) 
                                        - sum([qkl[i] for i in ND["injecting_branch"][n]])
                                        - sum([qg[g] for g in ND["G_ind"][n]])
                                        - bkk[n] * ckk[n] == -ND["Q"][n]) # 3c
    for b in 1:L # for each branch b
        k, l, ad = LD["from"][b], LD["to"][b], LD["anglediff_lim"][b]
        JuMP.@constraint(model, pkl[b] + G[k,l] * ckk[k] - G[k,l] * ckl[b] + B[k,l] * tkl[b] == 0.) # 3d
        JuMP.@constraint(model, qkl[b] - B[k,l] * ckk[k] + B[k,l] * ckl[b] + G[k,l] * tkl[b] == 0.) # 3e
        JuMP.@constraint(model, ckl[b]^2 + tkl[b]^2 - ckk[k] * ckk[l] == 0.) # 3f, nonconvex
        JuMP.@constraint(model, -ad <= tha[k] - tha[l] <= ad) # 3m
        JuMP.@NLconstraint(model, ckl[b] * sin(tha[k] - tha[l]) + tkl[b] * cos(tha[k] - tha[l]) == 0.) # (5) revised
    end
    JuMP.@constraint(model, [b = 1:L], pkl[b]^2 + qkl[b]^2 <= LD["heat_lim"][b]^2) # 3i
end

JuMP.optimize!(model)
@assert JuMP.termination_status(model) == JuMP.OPTIMAL

if true # get primal value
    _pg = JuMP.value.(  pg  )
    _qg = JuMP.value.(  qg  )
    _pkl = JuMP.value.( pkl )
    _qkl = JuMP.value.( qkl )
    _ckl = JuMP.value.( ckl )
    _tkl = JuMP.value.( tkl )
    _ckk = JuMP.value.( ckk )
    _tha = JuMP.value.( tha )
    primal_value_dict = merge(
        vdict(pg , _pg ),
        vdict(qg , _qg ),
        vdict(pkl, _pkl),
        vdict(qkl, _qkl),
        vdict(ckl, _ckl),
        vdict(tkl, _tkl),
        vdict(ckk, _ckk),
        vdict(tha, _tha)
    )                       
end

JuMP.primal_feasibility_report(model, primal_value_dict)

S = zeros(ComplexF64, N)
S[1:2] .= _pg .+ im .* _qg
S = S .- (ND["P"] .+ im .* ND["Q"]) # vector of nodal injecting complex power

V = sqrt.(_ckk)
U = V .* exp.(im .* _tha) # vector of nodal voltage phasor


# ------------- validation check -----------------
# 1, defining relations (2)
for b in 1:L
    k,l = LD["from"][b], LD["to"][b]
    tmp1, tmp2 = ct(V[k], _tha[k], V[l], _tha[l])
    println( "err: $(tmp1 - _ckl[b]), $(tmp2 - _tkl[b])" )
end
# 2, power flows
z = [LD["r"][b] + im * LD["x"][b] for b in 1:L]
y = [im * LD["half_b"][b] for b in 1:L]
Iz = [ ( U[ LD["from"][b] ] - U[ LD["to"][b] ] ) / z[b] for b in 1:L ]
Ill = [ U[ LD["from"][b] ] * y[b] for b in 1:L ]
Ilr = [ U[ LD["to"][b] ] * y[b] for b in 1:L ]
Iul = Iz .+ Ill
Iur = Iz .- Ilr
sul =               [ U[ LD["from"][b] ] * conj( Iul[b]) for b in 1:L ]
sz_before_sent =    [ U[ LD["from"][b] ] * conj( Iz[b] ) for b in 1:L ]
sz_after_sent =     [ U[  LD["to"][b]  ] * conj( Iz[b] ) for b in 1:L ]
sur =               [ U[  LD["to"][b]  ] * conj( Iur[b]) for b in 1:L ]

power_inj_net = [sum([ sul[i] for i in ND["leaving_branch"][n] ]) - sum([ sur[i] for i in ND["injecting_branch"][n] ]) for n in 1:N]
node_power_error_vector = S .- power_inj_net


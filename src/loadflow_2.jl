import JuMP
import MathOptInterface as MOI
import LinearAlgebra
# import Gurobi
import SCIP

# (MINLP) Model in section 2.1 in [Aigner/Burlacu_ijoc_2023]
# solving AC optimal power flow
# 01/12/23

if true # functions
    function Base.sum(e::Vector{Any})
        return isempty(e) ? 0. : error("cannot sum a Any-element Vector")
    end
    function vdict(x, x_) # used only once, at primal_feasibility_report
        return Dict(x[i] => x_[i] for i in eachindex(x))
    end
    function ct(uk,ul,tkl) # this function is not related to m building
        return uk * ul * cos(tkl), -uk * ul * sin(tkl) # ckl, tkl
    end # Additional: ckk = uk ^ 2, for any bus k
    function cnt_range(k, sense)
        if sense == 0 # injecting
            return ND["injecting_break_p"][k] : ND["injecting_break"][k]
        else # sense == 1: leaving
            return ND["leaving_break_p"][k] : ND["leaving_break"][k]
        end
    end
end
if true # data
    line_heat_cap = .9
    LD = Dict( # lines
        "from"  =>[1,1,2,2,2,3,4],
        "to"    =>[2,3,3,4,5,4,5],
        "r"     =>[2.,8,6,6,4,1,8]/100,
        "x"     =>[6.,24,18,18,12,3,24]/100,
        "half_b"=>[3,2.5,2,2,1.5,1,2.5]/100,
        "heat_lim" => fill(0.8,7),
        "anglediff_lim" => fill(pi/6,7),
    )
    L, N = length(LD["x"]), max(maximum(LD["from"]), maximum(LD["to"]))
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
    NG = length(GD["node"]) # number of generators
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
    tmp1, tmp2 = [length(i) for i in ND["leaving_branch"]], [length(i) for i in ND["injecting_branch"]]
    ND = merge(ND, Dict( # this part is auto-generated
    "leaving_break"     => [sum(tmp1[1:e]) for e in eachindex(tmp1)],
    "injecting_break"   => [sum(tmp2[1:e]) for e in eachindex(tmp2)],
    "injecting_node" => [Int[LD["from"][b] for b in ND["injecting_branch"][n]] for n in 1:N],
    "leaving_node" => [Int[LD["to"][b] for b in ND["leaving_branch"][n]] for n in 1:N])
    )
    ND = merge(ND, Dict(
        "leaving_break_p" => ND["leaving_break"] .- tmp1 .+ 1,
        "injecting_break_p" => ND["injecting_break"] .- tmp2 .+ 1
    ))
    @assert length(ND["P"]) == N # finish constructing ND
    bkk = [sum(LD["half_b"][b] for b in [ND["injecting_branch"][n]; ND["leaving_branch"][n]]) for n in 1:N]
    gkk = zero(bkk) # this should be zero for general lines which has only `b/2` part shunt branch
    # these parameters presents in paper 
    Y = [.0im for i in 1:N, j in 1:N] # node admittance matrix
    for b in 1:L
        k, l = LD["from"][b], LD["to"][b]
        Y[l,k] = Y[k,l] = -1.0 / (LD["r"][b] + LD["x"][b] * im)
    end
    for n in 1:N
        for b in [ND["injecting_branch"][n]; ND["leaving_branch"][n]] # δ(n)
            Y[n,n] += LD["half_b"][b] * im + 1.0 / (LD["r"][b] + LD["x"][b] * im)
        end
    end # this version is Y in TextBook
    G,B = real(Y),imag(Y)
end

if true # JuMP modeling
    m = JuMP.Model(SCIP.Optimizer)
    JuMP.@variable(m, GD["Pmin"][g] <= pg[g = 1:NG] <= GD["Pmax"][g])
    JuMP.@variable(m, GD["Qmin"][g] <= qg[g = 1:NG] <= GD["Qmax"][g])
    JuMP.@variable(m, ckl[1:L])
    JuMP.@variable(m, tkl[1:L])
    JuMP.@variable(m, ND["Vmin"][n]^2 <= ckk[n = 1:N] <= ND["Vmax"][n]^2)
    JuMP.@variable(m, tha[1:N])
    JuMP.fix(tha[end], 0.)
    JuMP.@variable(m, -LD["anglediff_lim"][b] <= thakl[b=1:L] <= LD["anglediff_lim"][b]) # 3m   
    JuMP.@variable(m, pbl[1:L])# define 4 quantities related to heat capacity of a line, where pbl = real(sul), pbi = real(sur), [ul = upper left in a `π`-m]
    JuMP.@variable(m, qbl[1:L])
    JuMP.@variable(m, pbi[1:L])
    JuMP.@variable(m, qbi[1:L])
    if true # modeling of the complicated last 4 variables
        for k in 1:N, (b,l,cnt) in zip(ND["leaving_branch"][k], ND["leaving_node"][k], cnt_range(k, 1)) # (k) ---b--> (l)
            JuMP.@constraint(m, G[k, l] * (ckl[b]-ckk[k]) - B[k, l] * tkl[b] - pbl[cnt] == 0.)
            JuMP.@constraint(m, -G[k, l] * tkl[b] - B[k, l] * (ckl[b]-ckk[k]) - qbl[cnt] == 0.)
        end
        for k in 1:N, (b,l,cnt) in zip(ND["injecting_branch"][k], ND["injecting_node"][k], cnt_range(k, 0)) # (l) ---b--> (k)
            JuMP.@constraint(m, G[l, k] * (ckl[b]-ckk[k]) + B[l, k] * tkl[b] - pbi[cnt] == 0.)
            JuMP.@constraint(m, G[l, k] * tkl[b] - B[l, k] * (ckl[b]-ckk[k]) - qbi[cnt] == 0.)
        end
        JuMP.@constraint(m, [b = 1:L], pbl[b] ^ 2 + qbl[b] ^ 2 <= line_heat_cap ^ 2) # 3i_part_1
        JuMP.@constraint(m, [b = 1:L], pbi[b] ^ 2 + qbi[b] ^ 2 <= line_heat_cap ^ 2) # 3i_part_2
    end
    
    JuMP.@constraint(m, [k = 1:N], gkk[k] * ckk[k] + sum(pbl[ cnt_range(k, 1) ]) + sum(pbi[ cnt_range(k, 0) ]) 
    - sum([pg[g] for g in ND["G_ind"][k]]) == -ND["P"][k] ) # reformulated nodal P balance 
    JuMP.@constraint(m, [k = 1:N], -bkk[k] * ckk[k] + sum(qbl[ cnt_range(k, 1) ]) + sum(qbi[ cnt_range(k, 0) ])
    - sum([qg[g] for g in ND["G_ind"][k]]) == -ND["Q"][k]) # reformulated nodal Q balance
    for b in 1:L, k in [LD["from"][b]], l in [LD["to"][b]] # loop for k,l is dumb
        JuMP.@constraint(m, thakl[b] == tha[k] - tha[l]) # defining relation of thakl
        JuMP.@constraint(m, ckl[b]^2 + tkl[b]^2 - ckk[k] * ckk[l] == 0.) # 3f, nonconvex quadratic
        JuMP.@NLconstraint(m, ckl[b] * sin(thakl[b]) + tkl[b] * cos(thakl[b]) == 0.) # (5) revised, << NONLINEAR Constraint !!
    end
    # # KVL (7 variables - 3 constraints = 5 variables - 1 constraints)
    # JuMP.@constraint(m, thakl[1] + thakl[3] - thakl[2] == 0.)
    # JuMP.@constraint(m, thakl[4] + thakl[7] - thakl[5] == 0.)
    # JuMP.@constraint(m, thakl[3] + thakl[6] - thakl[4] == 0.)
    
    JuMP.@objective(m, Min, sum(GD["c2"][g] * pg[g] * pg[g] + GD["c1"][g] * pg[g] + GD["c0"][g] for g in 1:NG)) # only pg related
end

JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL
@info "Optimal_ObjVal = $(JuMP.objective_value(m))"

if true # get primal value
    _pg = JuMP.value.(  pg  )
    _qg = JuMP.value.(  qg  )
    _ckl = JuMP.value.( ckl )
    _tkl = JuMP.value.( tkl )
    _ckk = JuMP.value.( ckk )
    _tha = JuMP.value.( tha )
    _thakl = JuMP.value.( thakl )
    _pbl = JuMP.value.(pbl)
    _qbl = JuMP.value.(qbl)
    _pbi = JuMP.value.(pbi)
    _qbi = JuMP.value.(qbi)
    primal_value_dict = merge(
        vdict(pg , _pg ),
        vdict(qg , _qg ),
        vdict(ckl, _ckl),
        vdict(tkl, _tkl),
        vdict(ckk, _ckk),
        vdict(tha, _tha),
        vdict(thakl, _thakl),
        vdict(pbl, _pbl),
        vdict(qbl, _qbl),
        vdict(pbi, _pbi),
        vdict(qbi, _qbi)
    )                       
end

vio_dict = JuMP.primal_feasibility_report(m, primal_value_dict)
println("MaxVio = $(maximum(values(vio_dict)))")

S = zeros(ComplexF64, N)
S[1:2] .= _pg .+ im .* _qg
S = S .- (ND["P"] .+ im .* ND["Q"]) # vector of nodal injecting complex power

V = sqrt.(_ckk)
U = V .* exp.(im .* _tha) # vector of nodal voltage phasor

# ------------- validation check -----------------
# 1, defining relations (2)
for b in 1:L
    tmp1, tmp2 = ct(V[LD["from"][b]], V[LD["to"][b]], _thakl[b])
    println( "trigono_err: $(tmp1 - _ckl[b]), $(tmp2 - _tkl[b])" )
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

I_inj_net = [sum([ Iul[i] for i in ND["leaving_branch"][n] ]) - sum([ Iur[i] for i in ND["injecting_branch"][n] ]) for n in 1:N]
I_inj = Y * U
node_current_error_vector = I_inj_net .- I_inj

# [ Info: Optimal_ObjVal = 125.70735558298271

# julia> S
# 5-element Vector{ComplexF64}:
#  0.3251605723594919 - 0.003775866886460394im
#  1.1512011005038718 + 0.03969576617389689im
#               -0.45 - 0.15im
#                -0.4 - 0.05im
#                -0.6 - 0.1im

# julia> V
# 5-element Vector{Float64}:
#  1.1025352351484579
#  1.0999999999999812
#  1.0766472900704889
#  1.0764605556602285
#  1.072048075080663

# julia> _tha
# 5-element Vector{Float64}:
#   0.05890025075269935
#   0.055762995860397145
#   0.01021165128577195
#   0.0071299249925721635
#  -0.0

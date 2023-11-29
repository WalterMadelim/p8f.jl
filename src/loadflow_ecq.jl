import MathOptInterface as MOI
import LinearAlgebra
# import Gurobi
import SCIP

# OPF model in paper [Aigner/Burlacu_ijoc_2023]

function ct(uk,tk,ul,tl) # (2)
    return uk * ul * cos(tk - tl), -uk * ul * sin(tk - tl) # ckl, tkl
end # Additional: ckk = uk ^ 2, for any bus k
function rightmost_3b(n) # actually "rightmost_3b - pg"
    leaving_inds = findall(x -> x == n,LD["from"])
    injecting_inds = findall(x -> x == n,LD["to"])
    g_inds = findall(x -> x == n,GD["node"])
    if isempty(injecting_inds)
        f =  sum(1. * pkl[i] for i in leaving_inds)
    elseif isempty(leaving_inds)
        f =  -sum(1. * pkl[i] for i in injecting_inds)
    else
        f =  sum(1. * pkl[i] for i in leaving_inds) - sum(1. * pkl[i] for i in injecting_inds)
    end
    if isempty(g_inds)
        return f
    else # an element exists in g_inds
        return f - pg[g_inds[1]]
    end
end
function rightmost_3c(n)
    leaving_inds = findall(x -> x == n,LD["from"])
    injecting_inds = findall(x -> x == n,LD["to"])
    g_inds = findall(x -> x == n,GD["node"])
    if isempty(injecting_inds)
        f =  sum(1. * qkl[i] for i in leaving_inds)
    elseif isempty(leaving_inds)
        f =  -sum(1. * qkl[i] for i in injecting_inds)
    else
        f =  sum(1. * qkl[i] for i in leaving_inds) - sum(1. * qkl[i] for i in injecting_inds)
    end
    if isempty(g_inds)
        return f
    else # an element exists in g_inds
        return f - qg[g_inds[1]]
    end
end

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
    ND = Dict(  # nodes/buses
        "type"  => [3,2,1,1,1],
        "P"     => [0.0, 0.2, 0.45, 0.4, 0.6], # demand
        "Q"     => [0.0, 0.1, 0.15, 0.05, 0.1], # demand
        "Vmin"  => [0.9, 0.9, 0.9, 0.9, 0.9],
        "Vmax"  => [1.5, 1.1, 1.1, 1.1, 1.1],
    )
    N, L, NG = length(ND["type"]), length(LD["from"]), length(GD["node"]) # frequently referred numbers
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

# GRB_ENV = Gurobi.Env();
# o = Gurobi.Optimizer(GRB_ENV); # create an optimizer after an ENV's settings are determined
o = SCIP.Optimizer()
const Nonlinear = MOI.Nonlinear
nlpor = Nonlinear.Model()
# define decision variables
tha = MOI.add_variables(o,N) # nodal Voltage angle
MOI.add_constraint(o, tha[end], MOI.EqualTo(-pi/6)) # θ[end] as a ref
pg = MOI.add_variables(o,NG) # real power output of generators
qg = MOI.add_variables(o,NG) # reactive power output of generators
pkl = MOI.add_variables(o,L) # defined in (3d)
qkl = MOI.add_variables(o,L) # defined in (3e)
ckk = MOI.add_variables(o,N) # node voltage magnitude ^ 2, for all nodes
ckl = MOI.add_variables(o,L) # see function ct(uk,tk,ul,tl)
tkl = MOI.add_variables(o,L) # see function ct(uk,tk,ul,tl)
aux_obj = MOI.add_variable(o)
# MOI.add_constraint(o, sum(GD["c2"][g] * pg[g] * pg[g] + GD["c1"][g] * pg[g] + GD["c0"][g] for g in 1:NG) - aux_obj, MOI.EqualTo(0.))
MOI.add_constraint(o, sum(GD["c2"][g] * pg[g] * pg[g] + GD["c1"][g] * pg[g] for g in 1:NG) - aux_obj, MOI.EqualTo(0.)) # remove the constant term in cost function to facilitate MOI modeling
obj_function = 1. * aux_obj
MOI.set(o, MOI.ObjectiveFunction{typeof(obj_function)}(), obj_function) # DON'T revise this line!
MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
if true # main constraints
    for n in 1:N # 3b
        MOI.add_constraint(o, rightmost_3b(n), MOI.EqualTo(-ND["P"][n]))
    end
    for n in 1:N # 3c
        MOI.add_constraint(o, rightmost_3c(n) - bkk[n] * ckk[n], MOI.EqualTo(-ND["Q"][n]))
    end
    for b in 1:L # 3d
        k,l = LD["from"][b], LD["to"][b]
        MOI.add_constraint(o, 1. * pkl[b] + G[k,l] * ckk[k] - G[k,l] * ckl[b] + B[k,l] * tkl[b], MOI.EqualTo(0.))
    end
    for b in 1:L # 3e
        k,l = LD["from"][b], LD["to"][b]
        MOI.add_constraint(o, 1. * qkl[b] - B[k,l] * ckk[k] + B[k,l] * ckl[b] + G[k,l] * tkl[b], MOI.EqualTo(0.))
    end
    for b in 1:L # (3f) : relaxing to "≤" would Not guarantee Voltage angle `cos(asin(sin_delta)) == cos_delta` relations
        k,l = LD["from"][b], LD["to"][b]
        MOI.add_constraint(o, 1. * ckl[b] * ckl[b] + 1. * tkl[b] * tkl[b] - 1. * ckk[k] * ckk[l] , MOI.EqualTo(0.)) # this Eq constr is NonConvex, thus run `MOI.set(o, MOI.RawOptimizerAttribute("NonConvex"), 2)` 
        # MOI.add_constraint(o, 1. * ckl[b] * ckl[b] + 1. * tkl[b] * tkl[b] - 1. * ckk[k] * ckk[l] , MOI.LessThan(0.))
    end
    # (3g) is put off to the NL portion, (3h) is useless
    for b in 1:L # 3i
        MOI.add_constraint(o, 1. * pkl[b] * pkl[b] + 1. * qkl[b] * qkl[b], MOI.LessThan(LD["heat_lim"][b] ^ 2))
    end
    MOI.add_constraint.(o, pg, MOI.GreaterThan.(GD["Pmin"])) # 3j
    MOI.add_constraint.(o, pg, MOI.LessThan.(GD["Pmax"]))
    MOI.add_constraint.(o, qg, MOI.GreaterThan.(GD["Qmin"])) # 3k
    MOI.add_constraint.(o, qg, MOI.LessThan.(GD["Qmax"]))
    for n in 1:N # 3l
        MOI.add_constraint(o, ckk[n], MOI.LessThan( ND["Vmax"][n] ^ 2 ))
        MOI.add_constraint(o, ckk[n], MOI.GreaterThan( ND["Vmin"][n] ^ 2 ))
    end
    for b in 1:L # 3m
        k, l, ad = LD["from"][b], LD["to"][b], LD["anglediff_lim"][b]
        MOI.add_constraint(o, 1. * tha[k] - tha[l], MOI.LessThan(    ad))
        MOI.add_constraint(o, 1. * tha[k] - tha[l], MOI.GreaterThan(-ad))
    end
    # (3n) specify decisions and is not a constraint
    # dropping (3g) would Not ensure Voltage angle "KVL" relations
    for b in 1:L # NL portion
        Nonlinear.add_constraint(nlpor,:($(ckl[b]) * sin($(tha[LD["from"][b]]) - $(tha[LD["to"][b]])) + $(tkl[b]) * cos($(tha[LD["from"][b]]) - $(tha[LD["to"][b]]))), MOI.EqualTo(0.)) # see notes on (5)
    end
    evaluator = Nonlinear.Evaluator(nlpor,Nonlinear.ExprGraphOnly(),MOI.VariableIndex[]) # Only the 1st arg matters. the 2nd parameter appears as a hint, the last parameter is a place-holder
    MOI.set(o,MOI.NLPBlock(),MOI.NLPBlockData(evaluator))
end

# MOI.set(o, MOI.RawOptimizerAttribute("NonConvex"), 2) # For Gurobi
MOI.set(o, MOI.RawOptimizerAttribute("limits/gap"), 0.005) # For SCIP

MOI.optimize!(o)
@assert MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL

pkl_ = MOI.get(o, MOI.VariablePrimal(), pkl)
qkl_ = MOI.get(o, MOI.VariablePrimal(), qkl)

S = zeros(ComplexF64, N)
S[1:2] .= MOI.get(o, MOI.VariablePrimal(), pg) .+ im .* MOI.get(o, MOI.VariablePrimal(), qg)
S = S .- (ND["P"] .+ im .* ND["Q"]) # vector of nodal injecting complex power

V = sqrt.(MOI.get(o, MOI.VariablePrimal(), ckk))
tha_ = MOI.get(o, MOI.VariablePrimal(), tha)
U = V .* exp.(im .* tha_) # vector of nodal voltage phasor


# ------------- validation check -----------------
ckl_ = MOI.get(o, MOI.VariablePrimal(), ckl)
tkl_ = MOI.get(o, MOI.VariablePrimal(), tkl)
# 1, defining relations (2)
for b in 1:L
    k,l = LD["from"][b], LD["to"][b]
    tmp1, tmp2 = ct(V[k], tha_[k], V[l], tha_[l])
    println( "err: $(tmp1 - ckl_[b]), $(tmp2 - tkl_[b])" )
end
# 2, power flows
z = [LD["r"][b] + im * LD["x"][b] for b in 1:L]
y = [im * LD["half_b"][b] for b in 1:L]
Iz = [ ( U[ LD["from"][b] ] - U[ LD["to"][b] ] ) / z[b] for b in 1:L ]
sz = [ U[ LD["from"][b] ] * conj( Iz[b] ) for b in 1:L ]
Ill = [ U[ LD["from"][b] ] * y[b] for b in 1:L ]
Ilr = [ U[ LD["to"][b] ] * y[b] for b in 1:L ]
Iul = Iz .+ Ill
Iur = Iz .- Ilr



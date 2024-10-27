# import PowerModels # to parse Matpower *.m data files
import LinearAlgebra
import Distributions
import Random
import Gurobi
import JuMP
using Logging

# inf-norm based uncertainty set
# lb can be refined continually, but the dist cannot be reduced to 0, hence the ub is not available

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function ip(x, y) return LinearAlgebra.dot(x, y) end
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        # JuMP.set_attribute(ø, "QCPDual", 1)
    elseif i == 1 
        ø = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 
        ø = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(ø) # JuMP.unset_silent(ø)
    return ø
    # (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    # if status != JuMP.OPTIMAL
    #     if status == JuMP.INFEASIBLE_OR_UNBOUNDED
    #         JuMP.set_attribute(ø, "DualReductions", 0)
    #         (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    #     end
    #     if status == JuMP.DUAL_INFEASIBLE
    #         @info "The program is unbounded"
    #         error()
    #     else
    #         error(" $status ")
    #     end
    # else
    #     return worstObj, worstZ = JuMP.objective_value(ø), JuMP.value.(Z)
    # end
end
function load_data()
    # network_data = PowerModels.parse_file("data/case6ww.m")
    # basic_net_data = PowerModels.make_basic_network(network_data)
    # F = PowerModels.calc_basic_ptdf_matrix(basic_net_data)
    S_BASE = 100 # MVA
    T = 8
    W = 2
    L = 3
    F = [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    Bℷ = Dict(
        "f" =>  Int[1,1,1,2,2,2,2,3,3,4,5],
        "t" =>  Int[2,4,5,3,4,5,6,5,6,5,6],
        "BC" => [4,6,4,4,4,3,9,7,8,2,4]/10
    ) # lines
    Wℷ = Dict(
        "id" => Int[1, 2],
        "n" => Int[2, 3],
        "CW" => 1000. * ones(T, W),
        "M" => [1.8, 1.7]
    ) # wind
    Lℷ = Dict(
        "id" => Int[1,2,3],
        "n" => Int[4,5,6],
        "CL" => [4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 9600.0 8265.6 8665.6],
        "M" => [1, 1.2, 1]
    ) # load
    Gℷ = Dict(
        "id" => Int[1, 2, 3], # the 1st one is slack generator
        "n" => Int[1, 2, 3], # the 1st generator resides at bus 1
        "ZS" => [1., 0, 0],
        "ZP" => [0.5, 0, 0], # Pzero, consistent with IS
        "C2" => [53.3, 88.9, 74.1], # this quadratic cost is not prominent
        "C1" => [1166.9, 1033.3, 1083.3],
        "C0" => [213.1, 200, 240],
        "CR" => [83., 74, 77],
        "CST" => [210., 200, 240],
        "CSH" => [210., 200, 240],
        "PI" => [0.5, 0.375, 0.45],
        "PS" => [2, 1.5, 1.8],
        "RU" => [.6, .6, .6],
        "SU" => [.6, .6, .6],
        "RD" => [.6, .6, .6],
        "SD" => [.6, .6, .6],
        "UT" => Int[3, 3, 3],
        "DT" => Int[3, 3, 3]
    )
    G = length(Gℷ["n"])
    Gℷ = merge(Gℷ, Dict("M" => [Gℷ["C2"][g] * Gℷ["PS"][g]^2 + Gℷ["C1"][g] * Gℷ["PS"][g] + Gℷ["C0"][g] for g in 1:G]))
    @assert W == length(Wℷ["n"])
    @assert L == length(Lℷ["n"])
    B, N = size(F)
    SRD = 1.5 # system reserve demand
    PE = 1.2e3
    MY = [0.654662 0.641636; 0.656534 0.819678; 0.511227 0.392445; 0.435808 0.123746; 0.100096 0.35382; 0.782138 0.542857; 0.798238 0.76052; 0.39923 0.309604]
    MZ = [[0.7310,0.4814,0.6908,0.4326,0.1753,0.8567,0.8665,0.6107] [0.7010,0.5814,0.3908,0.1326,0.4153,0.7567,0.8565,0.5107] [0.2010,0.6814,0.1908,0.4326,0.8153,0.7567,0.6565,0.7107]]
    return T, B, G, W, L, F, SRD, Bℷ, Gℷ, Wℷ, Lℷ, PE, MY, MZ
end
# precision = 5e-4 approximately
T, B, G, W, L, F, SRD, Bℷ, Gℷ, Wℷ, Lℷ, PE, MY, MZ = load_data()
@enum State begin
    t1
    t2f
    t3
    t2b
end

seed = abs(rand(Int))
show(seed)
Random.seed!(seed)
u, v, x = 1. * rand(Bool, T, G), 1. * rand(Bool, T, G), 1. * rand(Bool, T, G)
x1 = (u, v, x)
β1, Y = rand(T, W), rand(T, W)
x2 = x1, Y
β2, Z = rand(T, L), rand(T, L)
if true # Dicts
    Ϝ1_2 = Dict(
        "cn" => Float64[],
        "px" => typeof(x1)[] # (pu, pv, px)
    )
    ℶ1_2 = Dict(
        "cn" => Float64[],
        "px" => typeof(x1)[], # (pu, pv, px)
        "pβ" => Matrix{Float64}[] # β1
    )
    Δ1_2 = Dict( # 1️⃣ used only in termination assessment
        "f" => Float64[],
        "x" => typeof(x1)[], # (u, v, x)
        "β" => Matrix{Float64}[] # β1
    )
    Ϝ2_2 = Dict(
        "cn" => Float64[],
        "px" => typeof(x2)[] # ((pu, pv, px), pY)
    )
    ℶ2_2 = Dict(
        "cn" => Float64[],
        "px" => typeof(x2)[], # ((pu, pv, px), pY)
        "pβ" => Matrix{Float64}[] # β2
    )
    Δ2_2 = Dict(
        "f" => Float64[],
        "x" => typeof(x2)[], # ((u, v, x), Y)
        "β" => Matrix{Float64}[] # β2
    )
    function pushℶ(ℶ, cn, px, pβ)
        push!(ℶ["cn"], cn)
        push!(ℶ["px"], px)
        push!(ℶ["pβ"], pβ)
    end
    function pushΔ(Δ, f, x, β)
        push!(Δ["f"], f)
        push!(Δ["x"], x)
        push!(Δ["β"], β)
    end
    function pushϜ(Ϝ, cn, px) # 😊
        push!(Ϝ["cn"], cn)
        push!(Ϝ["px"], px)
    end
end
function eval_Δ_at(Δ, x1, β1) # used in termination assessment
    isempty(Δ["f"]) && return Inf
    fV2, x1V2, β1V2 = Δ["f"], Δ["x"], Δ["β"]
    R2 = length(fV2)
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G], sum(λ[r] * x1V2[r][i][t, g] for r in 1:R2) == x1[i][t, g])
    JuMP.@constraint(ø, [t = 1:T, w = 1:W],          sum(λ[r] * β1V2[r][t, w]    for r in 1:R2) ==    β1[t, w])
    JuMP.@objective(ø, Min, ip(fV2, λ))
    JuMP.optimize!(ø)
    return (JuMP.termination_status(ø) == JuMP.OPTIMAL ? JuMP.objective_value(ø) : Inf)
end
# value function in FWD pass
function master(is01::Bool) # 😊 A special value function
    cnV1, px1V1, R1 = Ϝ1_2["cn"], Ϝ1_2["px"], length(Ϝ1_2["cn"])
    cnV2, px1V2, R2, pβV = ℶ1_2["cn"], ℶ1_2["px"], length(ℶ1_2["cn"]), ℶ1_2["pβ"]
    ø = JumpModel(0)
    JuMP.@variable(ø, oβ)
    JuMP.@variable(ø, o)
    JuMP.@variable(ø, β[t = 1:T, w = 1:W]) # trade off between 'oβ' and 'o'
    JuMP.@variable(ø, u[t = 1:T, g = 1:G])
    JuMP.@variable(ø, v[t = 1:T, g = 1:G])
    JuMP.@variable(ø, x[t = 1:T, g = 1:G])
    if is01
        JuMP.set_binary.([u; v; x])
    else
        (JuMP.set_lower_bound.([u; v; x], 0.); JuMP.set_upper_bound.([u; v; x], 1.))
    end
    JuMP.@constraint(ø, Ϝ[r = 1:R1], 0 >= cnV1[r] + ip(px1V1[r], (u, v, x)))
    JuMP.@constraint(ø, ℶ[r = 1:R2], o >= cnV2[r] + ip(px1V2[r], (u, v, x)) + ip(pβV[r], β))
    JuMP.@constraint(ø, oβ >= ip(MY, β))
        JuMP.@constraint(ø, [g = 1:G],          x[1, g] - Gℷ["ZS"][g] == u[1, g] - v[1, g])
        JuMP.@constraint(ø, [t = 2:T, g = 1:G], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
        JuMP.@constraint(ø, [g = 1:G, t = 1:T-Gℷ["UT"][g]+1], sum(x[i, g] for i in t:t+Gℷ["UT"][g]-1)      >= Gℷ["UT"][g] * u[t, g])
        JuMP.@constraint(ø, [g = 1:G, t = T-Gℷ["UT"][g]+1:T], sum(x[i, g] - u[t, g] for i in t:T)          >= 0.)
        JuMP.@constraint(ø, [g = 1:G, t = 1:T-Gℷ["DT"][g]+1], sum(1. - x[i, g] for i in t:t+Gℷ["DT"][g]-1) >= Gℷ["DT"][g] * v[t, g])
        JuMP.@constraint(ø, [g = 1:G, t = T-Gℷ["DT"][g]+1:T], sum(1. - x[i, g] - v[t, g] for i in t:T)     >= 0.)
        JuMP.@expression(ø, CST[t = 1:T, g = 1:G], Gℷ["CST"][g] * u[t, g])
        JuMP.@expression(ø, CSH[t = 1:T, g = 1:G], Gℷ["CSH"][g] * v[t, g])
        JuMP.@expression(ø, COST1, sum([CST; CSH]))
    JuMP.@objective(ø, Min, COST1 + oβ + o) # if bounded, only `o` is dynamically refined
    @info "executing master..."
    # JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            (JuMP.set_lower_bound(o, 0.); JuMP.set_lower_bound(oβ, 0.))
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
            @assert status == JuMP.OPTIMAL " in master(): #22 $status"
            lb, olb = -Inf, -Inf
        else
            error(" in master(): $status ")
        end
    else
        lb, olb = JuMP.objective_value(ø), JuMP.value(o)
    end
    x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
    β1 = JuMP.value.(β)
    return lb, olb, x1, β1
end
function φ1(x1, β1, Y) # this value function is executed only when we are sure that x1_trial ensures the feasibility of THIS program
    x2 = x1, Y # 💡 disregard Ϝ2_2 because we are sure that x1_trial ensures the feasibility of THIS program
    const_obj = -ip(β1, Y) # ⚠️ Don't forget this
    cnV2, px2V2, R2, pβV = ℶ2_2["cn"], ℶ2_2["px"], length(ℶ2_2["cn"]), ℶ2_2["pβ"]
    ø = JumpModel(0)
    JuMP.@variable(ø, oβ)
    JuMP.@variable(ø, o)
    JuMP.@variable(ø, β[t = 1:T, l = 1:L]) # trade off between 'oβ' and 'o'
    JuMP.@constraint(ø, oβ >= ip(MZ, β))
    JuMP.@constraint(ø, ℶ[r = 1:R2], o >= cnV2[r] + ip(px2V2[r], x2) + ip(pβV[r], β))
    JuMP.@objective(ø, Min, oβ + o)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            (JuMP.set_lower_bound(o, 0.); JuMP.set_lower_bound(oβ, 0.))
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
            @assert status == JuMP.OPTIMAL " in value function φ1(;Y): #22 $status"
            lb = -Inf
        else
            error(" in value function φ1(;Y): $status ")
        end
    else
        lb = const_obj + JuMP.objective_value(ø) # ⚠️ Don't forget this
    end
    β2 = JuMP.value.(β)
    return lb, x2, β2
end
# c2g function to generate hazardous scenes
function maximize_φ2_over_Z(is01::Bool, x2, β2) # 😊
    (u, v, x), Y = x2
    ø = JumpModel(0)
    JuMP.@variable(ø, Z[t = 1:T, l = 1:L])
    if is01
        JuMP.set_binary.(Z)
    else
        [JuMP.set_upper_bound.(Z, 1.), JuMP.set_lower_bound.(Z, 0.)]
    end
    JuMP.@variable(ø, ℵQ1[t = 1:T, g = 1:G])
    JuMP.@variable(ø, ℵQ2[t = 1:T, g = 1:G])
    JuMP.@variable(ø, ℵQ3[t = 1:T, g = 1:G])
    JuMP.@variable(ø, ℵW[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ℵL[t = 1:T, l = 1:L] >= 0.)
    JuMP.@variable(ø, 0. <= ℵe[t = 1:T, g = 1:G] <= 1.) # RHS due to e >= 0
    JuMP.@variable(ø, ℵdl1[g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdl[t = 2:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdr1[g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdr[t = 2:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵPI[t = 1:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵPS[t = 1:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵbl[t = 1:T, b = 1:B] >= 0.)
    JuMP.@variable(ø, ℵbr[t = 1:T, b = 1:B] >= 0.)
    JuMP.@variable(ø, ℵR[t = 1:T] >= 0.)
    JuMP.@variable(ø, ℵ0[t = 1:T] >= 0.)
    JuMP.@constraint(ø,  ϖ[t = 1:T, w = 1:W],  ℵ0[t] + ℵW[t, w] + Wℷ["CW"][t, w] - PE + sum(F[b, Wℷ["n"][w]] * (ℵbl[t, b] - ℵbr[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(ø,  ζ[t = 1:T, l = 1:L], -ℵ0[t] + ℵL[t, l] + Lℷ["CL"][t, l] + PE + sum(F[b, Lℷ["n"][l]] * (ℵbr[t, b] - ℵbl[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(ø,  ρ[t = 1:T, g = 1:G], ℵPS[t, g] - ℵR[t] + Gℷ["CR"][g] >= 0.)
    JuMP.@constraint(ø, p²[t = 1:T, g = 1:G], Gℷ["C2"][g] * ℵe[t, g] - ℵQ2[t, g] - ℵQ1[t, g] == 0.)
    JuMP.@expression(ø,  pCommon[t = 1:T, g = 1:G], ℵPS[t, g] - ℵPI[t, g] - ℵ0[t] - 2. * ℵQ3[t, g] + Gℷ["C1"][g] * ℵe[t, g] + PE + sum((ℵbr[t, b] - ℵbl[t, b]) * F[b, Gℷ["n"][g]] for b in 1:B))
    JuMP.@constraint(ø,  pt1[g = 1:G], pCommon[1, g] + ℵdr1[g] - ℵdl1[g] + ℵdl[2, g] - ℵdr[2, g] == 0.)
    JuMP.@constraint(ø,  prest[t = 2:T-1, g = 1:G], pCommon[t, g] + ℵdr[t, g] - ℵdl[t, g] + ℵdl[t+1, g] - ℵdr[t+1, g] == 0.)
    JuMP.@constraint(ø,  ptT[g = 1:G], pCommon[T, g] + ℵdr[T, g] - ℵdl[T, g] == 0.)
    JuMP.@constraint(ø, [t = 1:T, g = 1:G], [ℵQ1[t, g], ℵQ2[t, g], ℵQ3[t, g]] in JuMP.SecondOrderCone())
    # ⚠️ don't forget the outer term
    JuMP.@objective(ø, Max, -ip(β2, Z)
        + PE * sum(sum(Wℷ["M"][w] * Y[t, w] for w in 1:W) - sum(Lℷ["M"][l] * Z[t, l]  for l in 1:L) for t in 1:T)
        + sum(ℵQ2 .- ℵQ1) + sum(ℵe[t, g] * (Gℷ["C0"][g] - (1 - x[t, g]) * Gℷ["M"][g]) for t in 1:T, g in 1:G)
        - sum(ℵW[t, w] * Wℷ["M"][w] * Y[t, w] for t in 1:T, w in 1:W) - sum(ℵL[t, l] * Lℷ["M"][l] * Z[t, l] for t in 1:T, l in 1:L)
        + SRD * sum(ℵR) + sum( ℵPI[t, g] * Gℷ["PI"][g] * x[t, g] - ℵPS[t, g] * Gℷ["PS"][g] * x[t, g]  for t in 1:T, g in 1:G)
        + sum((ℵbr[t, b] - ℵbl[t, b]) * (sum(F[b, Wℷ["n"][w]] * Wℷ["M"][w] * Y[t, w] for w in 1:W) - sum(F[b, Lℷ["n"][l]] * Lℷ["M"][l] * Z[t, l] for l in 1:L)) for t in 1:T, b in 1:B)
        + sum((ℵbl[t, b] + ℵbr[t, b]) * (-Bℷ["BC"][b]) for t in 1:T, b in 1:B)
        + sum( ℵ0[t] * (sum(Lℷ["M"][l] * Z[t, l] for l in 1:L) - sum(Wℷ["M"][w] * Y[t, w] for w in 1:W)) for t in 1:T)
        + sum(ℵdl1[g] * (Gℷ["ZP"][g] - Gℷ["RD"][g] * x[1, g] - Gℷ["SD"][g] * v[1, g]) - ℵdr1[g] * (Gℷ["RU"][g] * Gℷ["ZS"][g] + Gℷ["SU"][g] * u[1, g] + Gℷ["ZP"][g]) for g in 1:G)
        + sum( ℵdl[t, g] * (-Gℷ["RD"][g] * x[t, g] - Gℷ["SD"][g] * v[t, g]) - ℵdr[t, g] * (Gℷ["RU"][g] * x[t-1, g] + Gℷ["SU"][g] * u[t, g]) for t in 2:T, g in 1:G)
    )
    @info " maximizing φ2 over Z given (x2, β2) ... "
    # JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            return Inf
        else
            error(" in maximize_φ2_over_Z(): $status ")
        end
    else
        return worstObj, worstZ = JuMP.objective_value(ø), JuMP.value.(Z)
    end
end
function maximize_slacked_f_over_Z(is01::Bool, x2) # 😊
    (u, v, x), Y = x2
    ø = JumpModel(0)
    JuMP.@variable(ø, Z[t = 1:T, l = 1:L])
    if is01
        JuMP.set_binary.(Z)
    else
        [JuMP.set_upper_bound.(Z, 1.), JuMP.set_lower_bound.(Z, 0.)]
    end
    JuMP.@variable(ø, ℵW[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ℵL[t = 1:T, l = 1:L] >= 0.)
    JuMP.@variable(ø, ℵdl1[g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdl[t = 2:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, 0. <= ℵdr1[g = 1:G] <= 1.)
    JuMP.@variable(ø, 0. <= ℵdr[t = 2:T, g = 1:G] <= 1.)
    JuMP.@variable(ø, ℵPI[t = 1:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, 0. <= ℵPS[t = 1:T, g = 1:G] <= 1.)
    JuMP.@variable(ø, 0. <= ℵbl[t = 1:T, b = 1:B] <= 1.)
    JuMP.@variable(ø, 0. <= ℵbr[t = 1:T, b = 1:B] <= 1.)
    JuMP.@variable(ø, ℵR[t = 1:T] >= 0.)
    JuMP.@variable(ø, ℵ0[t = 1:T] >= 0.)
    JuMP.@constraint(ø,  ϖ[t = 1:T, w = 1:W],  ℵ0[t] + ℵW[t, w] + sum(F[b, Wℷ["n"][w]] * (ℵbl[t, b] - ℵbr[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(ø,  ζ[t = 1:T, l = 1:L], -ℵ0[t] + ℵL[t, l] + sum(F[b, Lℷ["n"][l]] * (ℵbr[t, b] - ℵbl[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(ø,  ρ[t = 1:T, g = 1:G], ℵPS[t, g] - ℵR[t] >= 0.)
    JuMP.@expression(ø,  pCommon[t = 1:T, g = 1:G], ℵPS[t, g] - ℵPI[t, g] - ℵ0[t] + sum((ℵbr[t, b] - ℵbl[t, b]) * F[b, Gℷ["n"][g]] for b in 1:B))
    JuMP.@constraint(ø,  pt1[g = 1:G], pCommon[1, g] + ℵdr1[g] - ℵdl1[g] + ℵdl[2, g] - ℵdr[2, g] == 0.)
    JuMP.@constraint(ø,  prest[t = 2:T-1, g = 1:G], pCommon[t, g] + ℵdr[t, g] - ℵdl[t, g] + ℵdl[t+1, g] - ℵdr[t+1, g] == 0.)
    JuMP.@constraint(ø,  ptT[g = 1:G], pCommon[T, g] + ℵdr[T, g] - ℵdl[T, g] == 0.)
    JuMP.@objective(ø, Max, -sum(ℵW[t, w] * Wℷ["M"][w] * Y[t, w] for t in 1:T, w in 1:W) - sum(ℵL[t, l] * Lℷ["M"][l] * Z[t, l] for t in 1:T, l in 1:L)
        + SRD * sum(ℵR) + sum( ℵPI[t, g] * Gℷ["PI"][g] * x[t, g] - ℵPS[t, g] * Gℷ["PS"][g] * x[t, g]  for t in 1:T, g in 1:G)
        + sum((ℵbr[t, b] - ℵbl[t, b]) * (sum(F[b, Wℷ["n"][w]] * Wℷ["M"][w] * Y[t, w] for w in 1:W) - sum(F[b, Lℷ["n"][l]] * Lℷ["M"][l] * Z[t, l] for l in 1:L)) for t in 1:T, b in 1:B)
        + sum((ℵbl[t, b] + ℵbr[t, b]) * (-Bℷ["BC"][b]) for t in 1:T, b in 1:B)
        + sum( ℵ0[t] * (sum(Lℷ["M"][l] * Z[t, l] for l in 1:L) - sum(Wℷ["M"][w] * Y[t, w] for w in 1:W)) for t in 1:T)
        + sum(ℵdl1[g] * (Gℷ["ZP"][g] - Gℷ["RD"][g] * x[1, g] - Gℷ["SD"][g] * v[1, g]) - ℵdr1[g] * (Gℷ["RU"][g] * Gℷ["ZS"][g] + Gℷ["SU"][g] * u[1, g] + Gℷ["ZP"][g]) for g in 1:G)
        + sum( ℵdl[t, g] * (-Gℷ["RD"][g] * x[t, g] - Gℷ["SD"][g] * v[t, g]) - ℵdr[t, g] * (Gℷ["RU"][g] * x[t-1, g] + Gℷ["SU"][g] * u[t, g]) for t in 2:T, g in 1:G)
    )
    @info "maximizing slacked f over Z given x2 ..."
    JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" status = $status; This is not possible because f is slacked ")
    else
        return slackWorstObj, slackWorstZ = JuMP.objective_value(ø), JuMP.value.(Z)
    end
end
function maximize_φ1_over_Y(is01::Bool, x1, β1) # 😊
    function inner_primal(Y)
        fV2, x2V2, R2, β2V = Δ2_2["f"], Δ2_2["x"], length(Δ2_2["f"]), Δ2_2["β"]
        JuMP.@variable(ø, λ[r = 1:R2] >= 0.)
        JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
        JuMP.@constraint(ø, ℵ1, sum(λ) == 1.)
        JuMP.@constraint(ø, ℵb[t = 1:T, l = 1:L], sum(λ[r] * β2V[r][t, l] for r in 1:R2) == β2[t, l])
        JuMP.@constraint(ø, ℵi[i = 1:3, t = 1:T, g = 1:G], sum(λ[r] * x2V2[r][1][i][t, g] for r in 1:R2) == x1[i][t, g])
        JuMP.@constraint(ø, ℵY[t = 1:T, w = 1:W], sum(λ[r] * x2V2[r][2][t, w] for r in 1:R2) == Y[t, w]) # Y is from outer layer
        JuMP.@objective(ø, Min, ip(MZ, β2) + ip(λ, fV2))
    end
    function inner_dual(Y)
        fV2, x2V2, R2, β2V = Δ2_2["f"], Δ2_2["x"], length(Δ2_2["f"]), Δ2_2["β"]
        JuMP.@variable(ø, ℵ1)
        JuMP.@variable(ø, ℵb[t = 1:T, l = 1:L])
        JuMP.@variable(ø, ℵi[i = 1:3, t = 1:T, g = 1:G])
        JuMP.@variable(ø, ℵY[t = 1:T, w = 1:W])
        JuMP.@constraint(ø, λ[r = 1:R2], fV2[r] - ℵ1
         - sum(ℵY[t, w] * x2V2[r][2][t, w] for t in 1:T, w in 1:W) 
         - sum(ℵb[t, l] * β2V[r][t, l] for t in 1:T, l in 1:L) 
         - sum(ℵi[i, t, g] * x2V2[r][1][i][t, g] for i in 1:3, t in 1:T, g in 1:G) >= 0.)
        JuMP.@constraint(ø, β2[t = 1:T, l = 1:L], ℵb[t, l] + MZ[t, l] == 0.)
        JuMP.@objective(ø, Max, ℵ1 + ip(ℵY, Y) + sum(ℵi[i, t, g] * x1[i][t, g] for i in 1:3, t in 1:T, g in 1:G))
    end
    fV2, x2V2, R2, β2V = Δ2_2["f"], Δ2_2["x"], length(Δ2_2["f"]), Δ2_2["β"]  
    @assert R2 >= 1 " Δ2_2 is empty, you could have skipped this subprocedure "
    ø = JumpModel(0)
    JuMP.@variable(ø, Y[t = 1:T, w = 1:W]) # outer layer
    if is01
        JuMP.set_binary.(Y)
    else
        (JuMP.set_lower_bound.(Y, 0.); JuMP.set_upper_bound.(Y, 1.))
    end
        JuMP.@variable(ø, ℵ1)
        JuMP.@variable(ø, ℵb[t = 1:T, l = 1:L])
        JuMP.@variable(ø, ℵi[i = 1:3, t = 1:T, g = 1:G])
        JuMP.@variable(ø, ℵY[t = 1:T, w = 1:W])
        JuMP.@constraint(ø, λ[r = 1:R2], fV2[r] - ℵ1
            - sum(ℵY[t, w] * x2V2[r][2][t, w] for t in 1:T, w in 1:W) 
            - sum(ℵb[t, l] * β2V[r][t, l] for t in 1:T, l in 1:L) 
            - sum(ℵi[i, t, g] * x2V2[r][1][i][t, g] for i in 1:3, t in 1:T, g in 1:G) >= 0.)
        JuMP.@constraint(ø, β2[t = 1:T, l = 1:L], ℵb[t, l] + MZ[t, l] == 0.)
    # ⚠️ don't forget the outer obj term
    JuMP.@objective(ø, Max, -ip(β1, Y) + ℵ1 + ip(ℵY, Y) + sum(ℵi[i, t, g] * x1[i][t, g] for i in 1:3, t in 1:T, g in 1:G))
    @info "maximizing φ1 over Y given x1, β1 ..."
    JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            return Inf
        else
            error(" maximize_φ1_over_Y() : $status ")
        end
    else
        return worstObj, worstY = JuMP.objective_value(ø), JuMP.value.(Y)
    end 
end
function maximize_φ1_dist_over_Y(is01::Bool, x1) # 📚 the main time consumer
    function inner_primal(Y) 
        (x2V2 = Δ2_2["x"]; R2 = length(x2V2))
        JuMP.@variable(ø, λ[r = 1:R2] >= 0.)
        JuMP.@variable(ø, x1Δ[i = 1:3, t = 1:T, g = 1:G]) # u, v, x by λ-conv-comb
        JuMP.@variable(ø, YΔ[t = 1:T, w = 1:W]) # Y by λ-conv-comb
        JuMP.@variable(ø, ai[i = 1:3, t = 1:T, g = 1:G])
        JuMP.@variable(ø, a2[t = 1:T, w = 1:W])
        JuMP.@constraint(ø, ℵ2l[t = 1:T, w = 1:W],             a2[t, w] >= YΔ[t, w] - Y[t, w])
        JuMP.@constraint(ø, ℵ2r[t = 1:T, w = 1:W],             a2[t, w] >= Y[t, w] - YΔ[t, w])
        JuMP.@constraint(ø, ℵil[i = 1:3, t = 1:T, g = 1:G], ai[i, t, g] >= x1Δ[i, t, g] - x1[i][t, g])
        JuMP.@constraint(ø, ℵir[i = 1:3, t = 1:T, g = 1:G], ai[i, t, g] >= x1[i][t, g] - x1Δ[i, t, g])
        JuMP.@constraint(ø, ℵi[i = 1:3, t = 1:T, g = 1:G], sum(λ[r] * x2V2[r][1][i][t, g] for r in 1:R2) == x1Δ[i, t, g])
        JuMP.@constraint(ø, ℵ2[t = 1:T, w = 1:W],          sum(λ[r] * x2V2[r][2][t, w]    for r in 1:R2) == YΔ[t, w])
        JuMP.@constraint(ø, ℵ1, sum(λ) == 1.)
        JuMP.@objective(ø, Min, sum(ai) + sum(a2))
    end
    function inner_dual(Y)
        JuMP.@variable(ø, ℵ2l[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ℵ2r[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ℵil[i = 1:3, t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, ℵir[i = 1:3, t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, ℵi[i = 1:3, t = 1:T, g = 1:G])
        JuMP.@variable(ø, ℵ2[t = 1:T, w = 1:W])
        JuMP.@variable(ø, ℵ1)
        JuMP.@constraint(ø, λ[r = 1:R2], -sum(ℵi[i, t, g] * x2V2[r][1][i][t, g] for i in 1:3, t in 1:T, g in 1:G) - sum(ℵ2[t, w] * x2V2[r][2][t, w] for t in 1:T, w in 1:W) - ℵ1 >= 0.)
        JuMP.@constraint(ø, x1Δ[i = 1:3, t = 1:T, g = 1:G], ℵi[i, t, g] + ℵil[i, t, g] - ℵir[i, t, g] == 0.)
        JuMP.@constraint(ø, YΔ[t = 1:T, w = 1:W], ℵ2[t, w] + ℵ2l[t, w] - ℵ2r[t, w] == 0.)
        JuMP.@constraint(ø, ai[i = 1:3, t = 1:T, g = 1:G], 1. - ℵil[i, t, g] - ℵir[i, t, g] == 0.)
        JuMP.@constraint(ø, a2[t = 1:T, w = 1:W], 1. - ℵ2l[t, w] - ℵ2r[t, w] == 0.)
        JuMP.@objective(ø, Max, ℵ1 + ip(Y, ℵ2r .- ℵ2l) + sum((ℵir[i, t, g] - ℵil[i, t, g]) * x1[i][t, g] for i in 1:3, t in 1:T, g in 1:G))
    end
    (x2V2 = Δ2_2["x"]; R2 = length(x2V2))
    @assert R2 >= 1 " Δ2_2 is empty, you could have skipped this subprocedure "
    ø = JumpModel(0)
    JuMP.@variable(ø, Y[t = 1:T, w = 1:W]) # outer layer
    if is01
        JuMP.set_binary.(Y)
    else
        (JuMP.set_lower_bound.(Y, 0.); JuMP.set_upper_bound.(Y, 1.))
    end
        JuMP.@variable(ø, ℵ2l[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ℵ2r[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ℵil[i = 1:3, t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, ℵir[i = 1:3, t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, ℵi[i = 1:3, t = 1:T, g = 1:G])
        JuMP.@variable(ø, ℵ2[t = 1:T, w = 1:W])
        JuMP.@variable(ø, ℵ1)
        JuMP.@constraint(ø, λ[r = 1:R2], -sum(ℵi[i, t, g] * x2V2[r][1][i][t, g] for i in 1:3, t in 1:T, g in 1:G) - sum(ℵ2[t, w] * x2V2[r][2][t, w] for t in 1:T, w in 1:W) - ℵ1 >= 0.)
        JuMP.@constraint(ø, x1Δ[i = 1:3, t = 1:T, g = 1:G], ℵi[i, t, g] + ℵil[i, t, g] - ℵir[i, t, g] == 0.)
        JuMP.@constraint(ø, YΔ[t = 1:T, w = 1:W], ℵ2[t, w] + ℵ2l[t, w] - ℵ2r[t, w] == 0.)
        JuMP.@constraint(ø, ai[i = 1:3, t = 1:T, g = 1:G], 1. - ℵil[i, t, g] - ℵir[i, t, g] == 0.)
        JuMP.@constraint(ø, a2[t = 1:T, w = 1:W], 1. - ℵ2l[t, w] - ℵ2r[t, w] == 0.)
        JuMP.@objective(ø, Max, ℵ1 + ip(Y, ℵ2r .- ℵ2l) + sum((ℵir[i, t, g] - ℵil[i, t, g]) * x1[i][t, g] for i in 1:3, t in 1:T, g in 1:G))
    @info "maximizing φ1_dist over Y given x1..."
    # JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" in maximize_φ1_dist_over_Y(): $status ")
    else
        return worstDistance, worstY = JuMP.objective_value(ø), JuMP.value.(Y)
    end
end
# cut generating function used in BWD pass
function gen_cut_for_ℶ2_2(x2, Z) # 💡 To gen a cut instead of eval, we don't need β2
    pβ2 = -Z # 💡 this is fixed
    ø = JumpModel(0)
    JuMP.@variable(ø, u[t = 1:T, g = 1:G])
    JuMP.@variable(ø, v[t = 1:T, g = 1:G])
    JuMP.@variable(ø, x[t = 1:T, g = 1:G])
    JuMP.@variable(ø, Y[t = 1:T, w = 1:W])
    JuMP.@constraint(ø, cpu[t = 1:T, g = 1:G], u[t, g] == x2[1][1][t, g])
    JuMP.@constraint(ø, cpv[t = 1:T, g = 1:G], v[t, g] == x2[1][2][t, g])
    JuMP.@constraint(ø, cpx[t = 1:T, g = 1:G], x[t, g] == x2[1][3][t, g])
    JuMP.@constraint(ø, cpY[t = 1:T, w = 1:W], Y[t, w] == x2[2][t, w])
        JuMP.@variable(ø,  ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø,  ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø,  ρ[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø,  p[t = 1:T, g = 1:G])
        JuMP.@variable(ø, p²[t = 1:T, g = 1:G])
        JuMP.@variable(ø,  e[t = 1:T, g = 1:G] >= 0.)
        JuMP.@constraint(ø, ℵW[t = 1:T, w = 1:W], Wℷ["M"][w] * Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, ℵL[t = 1:T, l = 1:L], Lℷ["M"][l] * Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, [t = 1:T, g = 1:G], [p²[t, g] + 1, p²[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone())
        JuMP.@constraint(ø, ℵe[t = 1:T, g = 1:G], e[t, g] >= Gℷ["C2"][g] * p²[t, g] + Gℷ["C1"][g] * p[t, g] + Gℷ["C0"][g] - (1 - x[t, g]) * Gℷ["M"][g])
        JuMP.@constraint(ø, ℵdl1[g = 1:G], p[1, g] - Gℷ["ZP"][g]       >= -Gℷ["RD"][g] * x[1, g] - Gℷ["SD"][g] * v[1, g])
        JuMP.@constraint(ø, ℵdl[t = 2:T, g = 1:G], p[t, g] - p[t-1, g] >= -Gℷ["RD"][g] * x[t, g] - Gℷ["SD"][g] * v[t, g])
        JuMP.@constraint(ø, ℵdr1[g = 1:G], Gℷ["RU"][g] * Gℷ["ZS"][g] + Gℷ["SU"][g] * u[1, g]       >= p[1, g] - Gℷ["ZP"][g])
        JuMP.@constraint(ø, ℵdr[t = 2:T, g = 1:G], Gℷ["RU"][g] * x[t-1, g] + Gℷ["SU"][g] * u[t, g] >= p[t, g] - p[t-1, g])
        JuMP.@constraint(ø, ℵPI[t = 1:T, g = 1:G], p[t, g] >= Gℷ["PI"][g] * x[t, g])
        JuMP.@constraint(ø, ℵPS[t = 1:T, g = 1:G], Gℷ["PS"][g] * x[t, g] >= p[t, g] + ρ[t, g])
        JuMP.@constraint(ø, ℵbl[t = 1:T, b = 1:B],
            sum(F[b, Gℷ["n"][g]] * p[t, g] for g in 1:G) + sum(F[b, Wℷ["n"][w]] * (Wℷ["M"][w] * Y[t, w] - ϖ[t, w]) for w in 1:W) - sum(F[b, Lℷ["n"][l]] * (Lℷ["M"][l] * Z[t, l] - ζ[t, l]) for l in 1:L) >= -Bℷ["BC"][b]
        )
        JuMP.@constraint(ø, ℵbr[t = 1:T, b = 1:B],
            Bℷ["BC"][b] >= sum(F[b, Gℷ["n"][g]] * p[t, g] for g in 1:G) + sum(F[b, Wℷ["n"][w]] * (Wℷ["M"][w] * Y[t, w] - ϖ[t, w]) for w in 1:W) - sum(F[b, Lℷ["n"][l]] * (Lℷ["M"][l] * Z[t, l] - ζ[t, l]) for l in 1:L)
        )
        JuMP.@constraint(ø, ℵR[t = 1:T], sum(ρ[t, :]) >= SRD)
        JuMP.@constraint(ø, ℵ0[t = 1:T], sum(Wℷ["M"][w] * Y[t, w] - ϖ[t, w] for w in 1:W) + sum(p[t, :]) - sum(Lℷ["M"][l] * Z[t, l] - ζ[t, l] for l in 1:L) >= 0.)
        JuMP.@expression(ø, CP[t = 1:T], sum(Wℷ["M"][w] * Y[t, w] - ϖ[t, w] for w in 1:W) + sum(p[t, :]) - sum(Lℷ["M"][l] * Z[t, l] - ζ[t, l] for l in 1:L))
        JuMP.@expression(ø, CW[t = 1:T, w = 1:W], Wℷ["CW"][t, w] * ϖ[t, w])
        JuMP.@expression(ø, CL[t = 1:T, l = 1:L], Lℷ["CL"][t, l] * ζ[t, l])
        JuMP.@expression(ø, CR[t = 1:T, g = 1:G], Gℷ["CR"][g]    * ρ[t, g])
        JuMP.@expression(ø, COST2, sum(CW) + sum(CL) + sum(CR) + sum(e) + PE * sum(CP))
    JuMP.@objective(ø, Min, COST2)
    JuMP.set_attribute(ø, "QCPDual", 1)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" in gen_cut_for_ℶ2_2(): $status ")
    else
        px1 = JuMP.dual.(cpu), JuMP.dual.(cpv), JuMP.dual.(cpx)
        pY = JuMP.dual.(cpY)
        px2 = px1, pY
        cn = JuMP.objective_value(ø) - ip(px2, x2)
        return cn, px2, pβ2
    end
end
function gen_cut_for_ℶ1_2(x1Γ, Y) # 💡 To gen a cut instead of eval, we don't need β1
    pβ1 = -Y # 💡 this is fixed
    cnV2, px2V2, R2, pβV = ℶ2_2["cn"], ℶ2_2["px"], length(ℶ2_2["cn"]), ℶ2_2["pβ"]
    ø = JumpModel(0)
    JuMP.@variable(ø, x1[i = 1:3, t = 1:T, g = 1:G]) # 'x1' as a part of x2
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
    JuMP.@variable(ø, o)
    JuMP.@constraint(ø, cp[i = 1:3, t = 1:T, g = 1:G], x1[i, t, g] == x1Γ[i][t, g])
    JuMP.@constraint(ø, ℶ[r = 1:R2], o >= cnV2[r] + ip( px2V2[r], ((x1[1, :, :], x1[2, :, :], x1[3, :, :]), Y) ) + ip(pβV[r], β2))
    JuMP.@objective(ø, Min, ip(MZ, β2) + o)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE # this may happen in nascent phase of iterations
            return -Inf
        else
            error(" in gen_cut_for_ℶ1_2: $status ")
        end
    else
        px1 = JuMP.value.(cp[1, :, :]), JuMP.value.(cp[2, :, :]), JuMP.value.(cp[3, :, :])
        cn = JuMP.objective_value(ø) - ip(px1, x1Γ)
        return cn, px1, pβ1
    end
end

Random.seed!(23)
tV = [t1]
x1V, β1V = [x1], [β1]
YV = [Y]
x2V, β2V = [x2], [β2]
lbV, cost = zeros(1), zeros(3)
dist0flag = falses(1)
while true
    𝚃 = tV[1]
    if 𝚃 == t1
        lb, olb, x1V[1], β1V[1] = master(true)
        oub = eval_Δ_at(Δ1_2, x1V[1], β1V[1])
        if oub == Inf
            @info " ▶ lb = $lb, (Δ1 $(length(Δ1_2["f"])),ℶ1 $(length(ℶ1_2["cn"])),Δ2 $(length(Δ2_2["f"])),ℶ2 $(length(ℶ2_2["cn"])))"
        else
            ub = lb - olb + oub
            @info " ▶⋅▶ lb = $lb | $ub = ub, (Δ1 $(length(Δ1_2["f"])),ℶ1 $(length(ℶ1_2["cn"])),Δ2 $(length(Δ2_2["f"])),ℶ2 $(length(ℶ2_2["cn"])))"
        end
        lbV[1], cost[1] = lb, lb - olb  
        tV[1] = t2f
    elseif 𝚃 == t2f
        if isempty(Ϝ2_2["cn"]) # means that the trial (x1, β1) will never be deemed infeasible
            if isempty(Δ2_2["f"]) # we are indifferent to a specific worst Y, since all evaluates to Inf
                worstY = 1. * rand(Bool, T, W)
            else
                worstDistance, worstY = maximize_φ1_dist_over_Y(true, x1V[1])
                dist_is_0 = worstDistance < 5e-5
                    @info "in t2f" worstDistance dist_is_0
                dist0flag[1] = false # RESET
                if dist_is_0 # 💡
                    ret = maximize_φ1_over_Y(true, x1V[1], β1V[1])
                    @assert length(ret) == 2 " check the worstDistance threshold "
                    dist0flag[1] = true
                    _, worstY = ret # worstY is updated
                end
            end
            YV[1] = worstY # ★ decide Y ★
            _, x2V[1], β2V[1] = φ1(x1V[1], β1V[1], worstY) # ★ decide trial (x2, β2) ★
            cost[2] = ip(MZ, β2V[1]) - ip(worstY, β1V[1])
            tV[1] = t3
        else
            error("TODO #1 execute slack_max_min program")
        end
    elseif 𝚃 == t3
        ret = maximize_φ2_over_Z(true, x2V[1], β2V[1]) # 💡 we do this before its slacked counterpart because it has the potential to save time
        if length(ret) == 1
            _, slackWorstZ = maximize_slacked_f_over_Z(true, x2V[1])
            error("TODO #2 generate feas. cut for Digamma2")
        else
            worstObj, worstZ = ret
            cost[3] = worstObj
            @info " lb = $(lbV[1]) | $(sum(cost)) = cost ◀"
            pushΔ(Δ2_2, worstObj, x2V[1], β2V[1])
            cn, px2, pβ2 = gen_cut_for_ℶ2_2(x2V[1], worstZ)
            pushℶ(ℶ2_2, cn, px2, pβ2)
            tV[1] = t2b
        end
    elseif 𝚃 == t2b
        if dist0flag[1]
            ret = maximize_φ1_over_Y(true, x1V[1], β1V[1])
                @assert length(ret) == 2 " ib t2b, worstDistance shouldn't > 0, please rethink "
            worstObj, worstY = ret
            pushΔ(Δ1_2, worstObj, x1V[1], β1V[1]) # 🥲  difficult to reach
        else
            worstY = YV[1] # reuse, to speed up
        end
        ret = gen_cut_for_ℶ1_2(x1V[1], worstY)
        if length(ret) == 1
            tV[1] = t2f
        else # ret = cn, px1, pβ1
            pushℶ(ℶ1_2, ret...) # UPD LOWER
            tV[1] = t1
        end
    end
end


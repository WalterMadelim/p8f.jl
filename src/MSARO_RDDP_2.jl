import JuMP
import Gurobi
import Random
import Distributions
import LinearAlgebra
using Logging

# MSARO with no RCR assumption
# case study: a single-item newsvendor problem
# algorithm in Supplemental material RDDP
# 18/10/24

function JumpModel(i)
    if i == 0
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    elseif i == 1
        ø = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2
        ø = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(ø) # JuMP.unset_silent(ø)
    return ø
end
function ip(x, y) LinearAlgebra.dot(x, y) end
global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env() # single item newsvendor problem to test incomplete-recourse

if true # data
    Random.seed!(87) 
    function randDext() return 5. + 20. * Int(rand() > .5) end # P_D_of_R_V = Distributions.Uniform(5., 25.)
    T = 7
    D1 = 24.9 # first stage's r.v. is deterministic
    H = [0, 0, 0, 0, 5, 5, 8.] # * by [y]^+
    C = [0, 1, 2, 3, 15, 15, 15.] # * by u
    B = [7, 7, 7, 7, 30, 30, 35.] # * by [y]^-
    M = [8. * 25, 6. * 25, 4. * 25, 3. * 25, 0, 25, 25] # hard constraint on UB of y
    y_at_0 = -1.2
    ℶempty = Dict(
        "py" => Float64[],
        "cn" => Float64[]
    )
    ℶ = [deepcopy(ℶempty) for _ in 1:T-1] # opt cut dict
    Ϝ = [deepcopy(ℶempty) for _ in 1:T-1] # feas cut-dict
    Δempty = Dict(
        "y" => Float64[],
        "f" => Float64[]
    )
    Δ = [deepcopy(Δempty) for _ in 1:T-1] # overestimator of c2g
end

function pre_c2g(yΓ, t) # used in stage T
    function slacked_subproblem_primal(D)
        JuMP.@variable(ø, s1 >= 0.)
        JuMP.@variable(ø, s2 >= 0.)
        JuMP.@variable(ø, s3 >= 0.)
        JuMP.@variable(ø, u >= 0.)
        JuMP.@variable(ø, y)
        JuMP.@constraint(ø, ℵD, s1 - s2 + y - u == yΓ - D)
        JuMP.@constraint(ø, ℵM, s3 - y >= -M[t])
        JuMP.@objective(ø, Min, s1 + s2 + s3)
    end
    function slacked_subproblem_dual(D)
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵD)
        JuMP.@constraint(ø, s1, 1. - ℵD >= 0.)
        JuMP.@constraint(ø, s2, 1. + ℵD >= 0.)
        JuMP.@constraint(ø, s3, 1. - ℵM >= 0.)
        JuMP.@constraint(ø, u, ℵD >= 0.)
        JuMP.@constraint(ø, y, ℵM - ℵD == 0.)
        JuMP.@objective(ø, Max, ℵD * (yΓ - D) - ℵM * M[t])
    end
    @assert t == T
    ø = JumpModel(0)
    JuMP.@variable(ø, 5. <= D <= 25.) # outer layer
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵD)
    JuMP.@constraint(ø, s1, 1. - ℵD >= 0.)
    JuMP.@constraint(ø, s2, 1. + ℵD >= 0.)
    JuMP.@constraint(ø, s3, 1. - ℵM >= 0.)
    JuMP.@constraint(ø, u, ℵD >= 0.)
    JuMP.@constraint(ø, y, ℵM - ℵD == 0.)
    JuMP.@objective(ø, Max, ℵD * (yΓ - D) - ℵM * M[t])
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    @assert status == JuMP.OPTIMAL "in pre_c2g(yΓ, t), $status"
    return slack_value, slack_D = JuMP.objective_value(ø), JuMP.value(D)
end
function c2g(yΓ, t) # used in stage T
    function subproblem_primal(D)
        JuMP.@variable(ø, u >= 0.)
        JuMP.@variable(ø, y)
        JuMP.@variable(ø, e)
        JuMP.@constraint(ø, ℵD, y - u == yΓ - D)
        JuMP.@constraint(ø, ℵH, e - H[t] * y >= 0.)
        JuMP.@constraint(ø, ℵB, e + B[t] * y >= 0.)
        JuMP.@constraint(ø, ℵM, -y >= -M[t])
        JuMP.@objective(ø, Min, e + C[t] * u)
    end
    function subproblem_dual(D)
        JuMP.@variable(ø, ℵH >= 0.)
        JuMP.@variable(ø, ℵB >= 0.)
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵD)
        JuMP.@constraint(ø, u, C[t] + ℵD >= 0.)
        JuMP.@constraint(ø, y, ℵM + H[t] * ℵH - B[t] * ℵB - ℵD == 0.)
        JuMP.@constraint(ø, e, 1. - ℵH - ℵB == 0.)
        JuMP.@objective(ø, Max, ℵD * (yΓ - D) - M[t] * ℵM)
    end
    @assert t == T
    ø = JumpModel(0)
    JuMP.@variable(ø, 5. <= D <= 25.) # outer layer
    JuMP.@variable(ø, ℵH >= 0.)
    JuMP.@variable(ø, ℵB >= 0.)
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵD)
    JuMP.@constraint(ø, u, C[t] + ℵD >= 0.)
    JuMP.@constraint(ø, y, ℵM + H[t] * ℵH - B[t] * ℵB - ℵD == 0.)
    JuMP.@constraint(ø, e, 1. - ℵH - ℵB == 0.)
    JuMP.@objective(ø, Max, ℵD * (yΓ - D) - M[t] * ℵM)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    @assert status == JuMP.OPTIMAL "in c2g(yΓ, t), $status"
    return worst_obj, worst_D = JuMP.objective_value(ø), JuMP.value(D)
end
function pre_c2g(yΓ, t, Ϝ)
    function slacked_primal_subproblem(yΓ, t, Ϝ, D)
        cnV, pyV, L = Ϝ["cn"], Ϝ["py"], length(Ϝ["cn"])
        ø = JumpModel(0)
        JuMP.@variable(ø, s1 >= 0.)
        JuMP.@variable(ø, s2 >= 0.)
        JuMP.@variable(ø, s3 >= 0.)
        JuMP.@variable(ø, u >= 0.)
        JuMP.@variable(ø, y) # 🏹
        JuMP.@constraint(ø, ℵL[l = 1:L], 0. >= cnV[l] + pyV[l] * y)
        JuMP.@constraint(ø, ℵM, s3 - y >= -M[t])
        JuMP.@constraint(ø, ℵD, y - u + s1 - s2 == yΓ - D)
        JuMP.@objective(ø, Min, s1 + s2 + s3)
    end
    function slacked_dual_subproblem(yΓ, t, Ϝ, D)
        cnV, pyV, L = Ϝ["cn"], Ϝ["py"], length(Ϝ["cn"])
        ø = JumpModel(0)
        JuMP.@variable(ø, ℵL[l = 1:L] >= 0.)
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵD)
        JuMP.@constraint(ø, s1, 1. - ℵD >= 0.)
        JuMP.@constraint(ø, s2, 1. + ℵD >= 0.)
        JuMP.@constraint(ø, s3, 1. - ℵM >= 0.)
        JuMP.@constraint(ø, u, ℵD >= 0.)
        JuMP.@constraint(ø, y, ℵM - ℵD + ip(ℵL, pyV) == 0.)
        JuMP.@objective(ø, Max, ℵD * (yΓ - D) + ip(ℵL, cnV) -ℵM * M[t])
    end
    cnV, pyV, L1 = Ϝ["cn"], Ϝ["py"], length(Ϝ["cn"])
    ø = JumpModel(0)
    JuMP.@variable(ø, 5. <= D <= 25.) # outer layer
    JuMP.@variable(ø, ℵL[l = 1:L1] >= 0.)
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵD)
    JuMP.@constraint(ø, s1, 1. - ℵD >= 0.)
    JuMP.@constraint(ø, s2, 1. + ℵD >= 0.)
    JuMP.@constraint(ø, s3, 1. - ℵM >= 0.)
    JuMP.@constraint(ø, u, ℵD >= 0.)
    JuMP.@constraint(ø, y, ℵM - ℵD + ip(ℵL, pyV) == 0.)
    JuMP.@objective(ø, Max, ℵD * (yΓ - D) + ip(ℵL, cnV) -ℵM * M[t])
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    @assert status == JuMP.OPTIMAL "in pre_c2g(yΓ, t, Ϝ), $status"
    return slack_value, slack_D = JuMP.objective_value(ø), JuMP.value(D)
end
function c2g_dist(yΓ, t, yV, L2)
    function c2g_dist_primal(D)
        # λ-method part
        JuMP.@variable(ø, λ[1:L2] >= 0.)
        JuMP.@variable(ø, yΔ)
        JuMP.@constraint(ø, ℵλ, sum(λ) == 1.)
        JuMP.@constraint(ø, ℵyΔ, yΔ == ip(yV, λ))
        # primal feasible region part
        JuMP.@variable(ø, u >= 0.)
        JuMP.@variable(ø, y)
        JuMP.@constraint(ø, ℵD, y - u == yΓ - D)
        JuMP.@constraint(ø, ℵM, -y >= -M[t])
        # minimize distance
        JuMP.@variable(ø, ey)
        JuMP.@constraint(ø, ℵe1, ey >= y - yΔ)
        JuMP.@constraint(ø, ℵe2, ey >= yΔ - y)
        JuMP.@objective(ø, Min, ey)
    end
    function c2g_dist_dual(D)
        JuMP.@variable(ø, ℵe1 >= 0.)
        JuMP.@variable(ø, ℵe2 >= 0.)
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵD)
        JuMP.@variable(ø, ℵyΔ)
        JuMP.@variable(ø, ℵλ)
        JuMP.@constraint(ø, λ[l = 1:L2], yV[l] * ℵyΔ - ℵλ >= 0.) 
        JuMP.@constraint(ø, u, ℵD >= 0.) 
        JuMP.@constraint(ø, yΔ, ℵe2 - ℵe1 - ℵyΔ == 0.) 
        JuMP.@constraint(ø, ey, 1. - ℵe1 - ℵe2 == 0.) 
        JuMP.@constraint(ø, y, ℵe1 - ℵe2 + ℵM - ℵD == 0.) 
        JuMP.@objective(ø, Max, ℵD * (yΓ - D) + ℵλ - M[t] * ℵM)  
    end
    ø = JumpModel(0)
    JuMP.@variable(ø, 5. <= D <= 25.) # outer max layer
    JuMP.@variable(ø, ℵe1 >= 0.)
    JuMP.@variable(ø, ℵe2 >= 0.)
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵD)
    JuMP.@variable(ø, ℵyΔ)
    JuMP.@variable(ø, ℵλ)
    JuMP.@constraint(ø, λ[l = 1:L2], yV[l] * ℵyΔ - ℵλ >= 0.) 
    JuMP.@constraint(ø, u, ℵD >= 0.) 
    JuMP.@constraint(ø, yΔ, ℵe2 - ℵe1 - ℵyΔ == 0.) 
    JuMP.@constraint(ø, ey, 1. - ℵe1 - ℵe2 == 0.) 
    JuMP.@constraint(ø, y, ℵe1 - ℵe2 + ℵM - ℵD == 0.) 
    JuMP.@objective(ø, Max, ℵD * (yΓ - D) + ℵλ - M[t] * ℵM)  
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    @assert status == JuMP.OPTIMAL " in c2g_dist(yΓ, t, yV, L2), $status "
    min_distance, Dworst = JuMP.objective_value(ø), JuMP.value(D)
    min_distance > 1e-5 && (min_distance = Inf) # current Δ is nascent, cannot provide a finite UB
    return min_distance, Dworst
end
function c2g(yΓ, t, Δ)
    @assert t in 2:T-1 # FW & BW
    if isempty(Δ["f"])
        return Inf, randDext()
    else
        yV, fV, L2 = Δ["y"], Δ["f"], length(Δ["f"])
        ret = c2g_dist(yΓ, t, yV, L2)
        ret[1] == Inf && return ret # only trial worst r.v. is useful  
    end
    function c2g_primal(D)
        JuMP.@variable(ø, λ[1:L2] >= 0.) 
        JuMP.@variable(ø, u >= 0.)
        JuMP.@variable(ø, y)
        JuMP.@variable(ø, o)
        JuMP.@variable(ø, e)
        JuMP.@constraint(ø, ℵo, o >= ip(fV, λ))
        JuMP.@constraint(ø, ℵH, e - H[t] * y >= 0.) 
        JuMP.@constraint(ø, ℵB, e + B[t] * y >= 0.) 
        JuMP.@constraint(ø, ℵM, -y >= -M[t]) 
        JuMP.@constraint(ø, ℵλ, sum(λ) == 1.) 
        JuMP.@constraint(ø, ℵyΔ, y == ip(yV, λ)) 
        JuMP.@constraint(ø, ℵy, y - u == yΓ - D)
        JuMP.@objective(ø, Min, e + C[t] * u + o)
    end
    function c2g_dual(D)
        JuMP.@variable(ø, ℵo >= 0.)
        JuMP.@variable(ø, ℵH >= 0.) 
        JuMP.@variable(ø, ℵB >= 0.) 
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵλ) 
        JuMP.@variable(ø, ℵyΔ)
        JuMP.@variable(ø, ℵy)
        JuMP.@constraint(ø, λ[l = 1:L2], fV[l] * ℵo + yV[l] * ℵyΔ - ℵλ >= 0.) 
        JuMP.@constraint(ø, u, C[t] + ℵy >= 0.)
        JuMP.@constraint(ø, y, ℵM - ℵyΔ - ℵy + ℵH * H[t] - ℵB * B[t] == 0.)
        JuMP.@constraint(ø, o, 1. - ℵo == 0.)
        JuMP.@constraint(ø, e, 1. - ℵH - ℵB == 0.)
        JuMP.@objective(ø, Max, ℵy * (yΓ - D) + ℵλ - M[t] * ℵM)
    end
    ø = JumpModel(0)
    JuMP.@variable(ø, 5. <= D <= 25.) # outer layer
    JuMP.@variable(ø, ℵo >= 0.)
    JuMP.@variable(ø, ℵH >= 0.) 
    JuMP.@variable(ø, ℵB >= 0.) 
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵλ) 
    JuMP.@variable(ø, ℵyΔ)
    JuMP.@variable(ø, ℵy)
    JuMP.@constraint(ø, λ[l = 1:L2], fV[l] * ℵo + yV[l] * ℵyΔ - ℵλ >= 0.) 
    JuMP.@constraint(ø, u, C[t] + ℵy >= 0.)
    JuMP.@constraint(ø, y, ℵM - ℵyΔ - ℵy + ℵH * H[t] - ℵB * B[t] == 0.)
    JuMP.@constraint(ø, o, 1. - ℵo == 0.)
    JuMP.@constraint(ø, e, 1. - ℵH - ℵB == 0.)
    JuMP.@objective(ø, Max, ℵy * (yΓ - D) + ℵλ - M[t] * ℵM)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    @assert status == JuMP.OPTIMAL "in c2g(yΓ, t, Δ), $status"
    return ub, Dworst = JuMP.objective_value(ø), JuMP.value(D)
end
function v(yΓ, t, Ϝ, ℶ, D)
    if t ∉ 1:T-1
        error(" v() is used in FW pass only ")
    else
        cnV1, pyV1, L1 = Ϝ["cn"], Ϝ["py"], length(Ϝ["cn"])
        cnV2, pyV2, L2 = ℶ["cn"], ℶ["py"], length(ℶ["cn"])
    end
    ø = JumpModel(0)
    JuMP.@variable(ø, u >= 0.)
    JuMP.@variable(ø, y) # 🏹
    JuMP.@variable(ø, e)
    JuMP.@variable(ø, o)
    JuMP.@constraint(ø, y - u == yΓ - D)
    JuMP.@constraint(ø, e - H[t] * y >= 0.)
    JuMP.@constraint(ø, e + B[t] * y >= 0.)
    JuMP.@constraint(ø, -y >= -M[t])
    JuMP.@constraint(ø, [l = 1:L1], 0. >= cnV1[l] + pyV1[l] * y)
    JuMP.@constraint(ø, [l = 1:L2], o >= cnV2[l] + pyV2[l] * y)
    JuMP.@objective(ø, Min, e + C[t] * u + o)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            JuMP.optimize!(ø)
            status = JuMP.termination_status(ø)
            if status == JuMP.DUAL_INFEASIBLE
                JuMP.set_lower_bound(o, 0.)
                JuMP.optimize!(ø)
                status = JuMP.termination_status(ø)
                if status == JuMP.OPTIMAL
                    return glb, oTrial, yTrial = -Inf, -Inf, JuMP.value(y)
                else
                    error(" TODO: in v() #86, $status ")
                end
            else
                error(" TODO: in v() #37, $status ")
            end
        end
    else
        return glb, oTrial, yTrial = JuMP.objective_value(ø), JuMP.value(o), JuMP.value(y)
    end
end
function gen_feas_cut(yyΓ, t, Ϝ, D)
    if t ∉ 2:T
        error(" gen_feas_cut() : range of t error ")
    else
        cnV1, pyV1, L1 = Ϝ["cn"], Ϝ["py"], length(Ϝ["cn"])
    end
    ø = JumpModel(0)
    JuMP.@variable(ø, u >= 0.)
    JuMP.@variable(ø, y)
    JuMP.@variable(ø, yΓ)
    JuMP.@variable(ø, s1 >= 0.)
    JuMP.@variable(ø, s2 >= 0.)
    JuMP.@variable(ø, s3 >= 0.)
    JuMP.@constraint(ø, ℵyΓ, yΓ == yyΓ) # copy constr
    JuMP.@constraint(ø, s1 - s2 + y - u == yΓ - D)
    JuMP.@constraint(ø, s3 - y >= -M[t])
    JuMP.@constraint(ø, [l = 1:L1], 0. >= cnV1[l] + pyV1[l] * y)
    JuMP.@objective(ø, Min, s1 + s2 + s3)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        error(" in gen_feas_cut(yyΓ, t, Ϝ, D) || $status || you NEED to add more slack variables. ")
    else
        return JuMP.objective_value(ø), JuMP.dual(ℵyΓ)
    end
end
function gen_cut(yyΓ, t, Ϝ, ℶ, D)
    if t ∉ T:-1:2
        error(" gen_cut() is used in BW pass only ")
    else
        cnV1, pyV1, L1 = Ϝ["cn"], Ϝ["py"], length(Ϝ["cn"])
        cnV2, pyV2, L2 = ℶ["cn"], ℶ["py"], length(ℶ["cn"])
    end
    ø = JumpModel(0)
    JuMP.@variable(ø, yΓ)
    JuMP.@variable(ø, u >= 0.)
    JuMP.@variable(ø, y)
    JuMP.@variable(ø, e)
    JuMP.@variable(ø, o)
    isempty(ℶ["cn"]) && JuMP.set_lower_bound(o, 0.)
    JuMP.@constraint(ø, ℵyΓ, yΓ == yyΓ) # copy constr
    JuMP.@constraint(ø, y - u == yΓ - D)
    JuMP.@constraint(ø, e - H[t] * y >= 0.)
    JuMP.@constraint(ø, e + B[t] * y >= 0.)
    JuMP.@constraint(ø, -y >= -M[t])
    JuMP.@constraint(ø, [l = 1:L1], 0. >= cnV1[l] + pyV1[l] * y)
    JuMP.@constraint(ø, [l = 1:L2], o  >= cnV2[l] + pyV2[l] * y)
    JuMP.@objective(ø, Min, e + C[t] * u + o)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            JuMP.optimize!(ø)
            status = JuMP.termination_status(ø)
            error(" gen_cut() #5: $status ")
        else
            error(" gen_cut() #2: $status ")
        end
    else
        return JuMP.objective_value(ø), JuMP.dual(ℵyΓ)
    end
end
function eval_Δ_at(y, Δ) # used in termination assessment
    isempty(Δ["f"]) && return Inf
    yL, fL, L = Δ["y"], Δ["f"], length(Δ["f"])
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:L] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, ip(yL, λ) == y)
    JuMP.@objective(ø, Min, ip(fL, λ))
    JuMP.optimize!(ø)
    return (JuMP.termination_status(ø) == JuMP.OPTIMAL ? JuMP.objective_value(ø) : Inf)
end

tV = [1]
D = NaN * ones(T)
objF, oF, yF = NaN * ones(T-1), NaN * ones(T-1), NaN * ones(T-1)
termination_status = falses(1)
for ı in 1:typemax(Int)
    t = tV[1]
    if t in (1, T)
        if t == 1
            D[1], D[T] = D1, NaN
            objF[t], oF[t], yF[t] = v(y_at_0, t, Ϝ[t], ℶ[t], D[t])
            lb, ub = objF[1], objF[1] - oF[1] + eval_Δ_at(yF[1], Δ[1])
            @info "▶ transition cnt = $ı, lb = $lb | $ub = ub"
            if lb + 1e-4 > ub
                @info " 😊 Gap is closed "
                termination_status[1] = true
            end
            tV[1] += 1
        else # t == T
            slack_value, slack_D = pre_c2g(yF[t-1], t)
            if slack_value < 5e-6 # FWD --> ✅ --> BWD
                objWorst, D[t] = c2g(yF[t-1], t)
                termination_status[1] && break
                objWorst < Inf && [push!(Δ[t-1]["y"], yF[t-1]), push!(Δ[t-1]["f"], objWorst)]
                raw_cn, py = gen_cut(yF[t-1], t, deepcopy(ℶempty), deepcopy(ℶempty), D[t]) # stage T has no feasCut nor c2g 
                [push!(ℶ[t-1]["cn"], raw_cn - py * yF[t-1]), push!(ℶ[t-1]["py"], py)]
            else # FWD --> ❌ --> BWD
                raw_cn, py = gen_feas_cut(yF[t-1], t, deepcopy(ℶempty), slack_D)
                [push!(Ϝ[t-1]["cn"], raw_cn - py * yF[t-1]), push!(Ϝ[t-1]["py"], py)]
                @info " add a feas cut at stage $t in FWD pass"
            end
            tV[1] -= 1
        end
    else
        if D[T] === NaN # naive fwd
            slack_value, slack_D = pre_c2g(yF[t-1], t, Ϝ[t])
            if slack_value < 5e-6 # === Having RCR, thus proceed
                objWorst, D[t] = c2g(yF[t-1], t, Δ[t])
                objF[t], oF[t], yF[t] = v(yF[t-1], t, Ϝ[t], ℶ[t], D[t])
                tV[1] += 1
            else
                raw_cn, py = gen_feas_cut(yF[t-1], t, Ϝ[t], slack_D)
                [push!(Ϝ[t-1]["cn"], raw_cn - py * yF[t-1]), push!(Ϝ[t-1]["py"], py)]
                @info " add a feas cut at stage $t in FWD pass"
                tV[1] -= 1
            end
        else # naive bwd
            objWorst, D[t] = c2g(yF[t-1], t, Δ[t])
            objWorst < Inf && [push!(Δ[t-1]["y"], yF[t-1]), push!(Δ[t-1]["f"], objWorst)]
            raw_cn, py = gen_cut(yF[t-1], t, Ϝ[t], ℶ[t], D[t]) 
            [push!(ℶ[t-1]["cn"], raw_cn - py * yF[t-1]), push!(ℶ[t-1]["py"], py)]
            tV[1] -= 1
        end
    end
end

[ Info: ▶ transition cnt = 1, lb = -Inf | NaN = ub
[ Info: ▶ transition cnt = 13, lb = -75.0 | Inf = ub
[ Info:  add a feas cut at stage 2 in FWD pass
[ Info: ▶ transition cnt = 15, lb = -30.0 | Inf = ub
[ Info:  add a feas cut at stage 3 in FWD pass
[ Info:  add a feas cut at stage 2 in FWD pass
[ Info: ▶ transition cnt = 19, lb = 15.0 | Inf = ub
[ Info:  add a feas cut at stage 4 in FWD pass
[ Info:  add a feas cut at stage 3 in FWD pass
[ Info:  add a feas cut at stage 2 in FWD pass
[ Info: ▶ transition cnt = 25, lb = 35.0 | Inf = ub
[ Info:  add a feas cut at stage 5 in FWD pass
[ Info:  add a feas cut at stage 4 in FWD pass
[ Info:  add a feas cut at stage 3 in FWD pass
[ Info:  add a feas cut at stage 2 in FWD pass
[ Info: ▶ transition cnt = 33, lb = 105.0 | Inf = ub
[ Info: ▶ transition cnt = 45, lb = 1170.0 | 1170.0 = ub
[ Info:  😊 Gap is closed

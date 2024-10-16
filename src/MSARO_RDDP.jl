import JuMP
import Gurobi
import Random
import Distributions
import LinearAlgebra
using Logging

# Multistage continuous ARO via RDDP under Relative Complete Recourse assumption
# Test case: a multi-item inventory problem with a fixed budget
# bilinear program can be solved efficiently with Gurobi, you need to write dual program in a disciplined manner
# 16/10/24

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
GRB_ENV = Gurobi.Env()

# state variable from last stage has suffix Γ
# cutting plane dict has suffix ℶ
# inner simplex dict has suffix Δ
# accumulation price [state variable] a
# backlog cost B
# budget price C
# r.v. demand D
# epigraph variable e
# holding cost H
# length of the lambda model L
# budget total M
# ctrl variable u 💡 relative complete recourse holds becasue u = 0 is always a feasible solution
# inventory level [state variable] y

T, I = 7, 3
Random.seed!(87)
ℙ = Distributions.Uniform(5., 25.) # mean = 15
function randDext()
    i = Int(rand() > .5)
    return 5. + 20. * i
end
D = rand(ℙ, T, I)
ℙ = Distributions.Uniform(2., 4.)
H = rand(ℙ, T, I)
ℙ = Distributions.Uniform(3., 6.)
B = rand(ℙ, T, I)
B[end, :] *= 2
ℙ = Distributions.Uniform(1., 2.)
C = rand(ℙ, T, I)
M = 412. # budget
D1 = D[1, :] # first stage's r.v. is deterministic
a_at_0 = 0.5
y_at_0 = [-1. * i for i in 1:I]

ℶempty = Dict(
    "pa" => Float64[],
    "py" => Vector{Float64}[],
    "cn" => Float64[]
)
ℶ = [deepcopy(ℶempty) for _ in 1:T-1]
Δempty = Dict(
    "a" => Float64[],
    "y" => Vector{Float64}[],
    "f" => Float64[]
)
Δ = [deepcopy(Δempty) for _ in 1:T-1]

function v(aΓ, yΓ, t, ℶ, D) # used in FWD pass to generate trial x's
    function v(t, aΓ, yΓ, D) # checked 😊 value function Template
        function v_dual(t, aΓ, yΓ, D)
            ø = JumpModel(0)
            JuMP.@variable(ø, ℵa)
            JuMP.@variable(ø, ℵM >= 0.)
            JuMP.@variable(ø, ℵD[1:I])
            JuMP.@variable(ø, ℵH[1:I] >= 0.)
            JuMP.@variable(ø, ℵB[1:I] >= 0.)
            JuMP.@constraint(ø, a, ℵM - ℵa == 0.)
            JuMP.@constraint(ø, e[i = 1:I], 1. - (ℵB[i] + ℵH[i]) == 0.)
            JuMP.@constraint(ø, y[i = 1:I], ℵH[i] *  H[t, i] - ℵB[i] * B[t, i] - ℵD[i] == 0.)
            JuMP.@constraint(ø, u[i = 1:I], ℵD[i] + ℵa * C[t, i] >= 0.)  
            JuMP.@objective(ø, Max, aΓ * ℵa - M * ℵM + ip(yΓ, ℵD) - ip(ℵD, D))
            JuMP.optimize!(ø)
            status = JuMP.termination_status(ø)
            if status != JuMP.OPTIMAL
                error("in master_plus_RCYZ, status = $status")
            end
            JuMP.objective_value(ø)
        end
        ø = JumpModel(0) # t = 1
        JuMP.@variable(ø, y[1:I]) # 🏹
        JuMP.@variable(ø, a) # 🏹
        JuMP.@variable(ø, e[1:I])
        JuMP.@variable(ø, u[1:I] >= 0.)
        JuMP.@constraint(ø, ℵH[i = 1:I], e[i] >=  H[t, i] * y[i])
        JuMP.@constraint(ø, ℵB[i = 1:I], e[i] >= -B[t, i] * y[i])
        JuMP.@constraint(ø, ℵM, M >= a)
        JuMP.@constraint(ø, ℵa, a == aΓ + ip(C[t, :], u))
        JuMP.@constraint(ø, ℵD[i = 1:I], y[i] == yΓ[i] - D[i] + u[i]) # yΓ (free), aΓ (in [0, M]), realization of D[t = 1] (in [5, 25])
        JuMP.@objective(ø, Min, sum(e))
        # JuMP.set_attribute(υ, "DualReductions", 0)
        JuMP.optimize!(ø)
        status = JuMP.termination_status(ø)
        if status != JuMP.OPTIMAL
            error("in master_plus_RCYZ, status = $status")
        end
        JuMP.objective_value(ø)
    end
    @assert t in 1:T-1
    ø = JumpModel(0) # t = 1
    JuMP.@variable(ø, y[1:I]) # 🏹
    JuMP.@variable(ø, a) # 🏹
    JuMP.@variable(ø, e[1:I])
    JuMP.@variable(ø, u[1:I] >= 0.)
    JuMP.@variable(ø, o) # epi variable for ℶ
    JuMP.@constraint(ø, ℵH[i = 1:I], e[i] >=  H[t, i] * y[i])
    JuMP.@constraint(ø, ℵB[i = 1:I], e[i] >= -B[t, i] * y[i])
    JuMP.@constraint(ø, ℵM, M >= a)
    JuMP.@constraint(ø, ℵa, a == aΓ + ip(C[t, :], u))
    JuMP.@constraint(ø, ℵD[i = 1:I], y[i] == yΓ[i] - D[i] + u[i]) # yΓ (free), aΓ (in [0, M]), realization of D[t = 1] (in [5, 25])
    for (cn, pa, py) in zip(ℶ["cn"], ℶ["pa"], ℶ["py"])
        JuMP.@constraint(ø, o >= cn + pa * a + ip(py, y))
    end
    JuMP.@objective(ø, Min, sum(e) + o)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        @assert status == JuMP.INFEASIBLE_OR_UNBOUNDED " in v(aΓ, yΓ, t, ℶ, D): #1 $status "
        JuMP.set_attribute(ø, "DualReductions", 0)
        JuMP.optimize!(ø)
        status = JuMP.termination_status(ø)
        @assert status == JuMP.DUAL_INFEASIBLE "  in v(aΓ, yΓ, t, ℶ, D): #2 $status"
        JuMP.set_lower_bound(o, 0.)
        JuMP.optimize!(ø)
        status = JuMP.termination_status(ø)
        @assert status == JuMP.OPTIMAL " in v(aΓ, yΓ, t, ℶ, D): #3 $status"
        return -Inf, -Inf, JuMP.value(a), JuMP.value.(y)
    else
        return JuMP.objective_value(ø), JuMP.value(o), JuMP.value(a), JuMP.value.(y)
    end
end
function gen_cut(aΓtri, yΓtri, t, ℶ, D)
    @assert t in T:-1:2
    ø = JumpModel(0)
    # copy part
    JuMP.@variable(ø, aΓ)
    JuMP.@variable(ø, yΓ[1:I])
    JuMP.@constraint(ø, ℵaΓ, aΓ == aΓtri)
    JuMP.@constraint(ø, ℵyΓ[i = 1:I], yΓ[i] == yΓtri[i])
    # normal part follows
    JuMP.@variable(ø, y[1:I]) # 🏹
    JuMP.@variable(ø, a) # 🏹
    JuMP.@variable(ø, e[1:I])
    JuMP.@variable(ø, u[1:I] >= 0.)
    JuMP.@variable(ø, o) # epi variable for ℶ
    t == T && JuMP.set_lower_bound(o, 0.) # And ℶ is empty
    JuMP.@constraint(ø, ℵH[i = 1:I], e[i] >=  H[t, i] * y[i])
    JuMP.@constraint(ø, ℵB[i = 1:I], e[i] >= -B[t, i] * y[i])
    JuMP.@constraint(ø, ℵM, M >= a)
    JuMP.@constraint(ø, ℵa, a == aΓ + ip(C[t, :], u))
    JuMP.@constraint(ø, ℵD[i = 1:I], y[i] == yΓ[i] - D[i] + u[i])
    for (cn, pa, py) in zip(ℶ["cn"], ℶ["pa"], ℶ["py"])
        JuMP.@constraint(ø, o >= cn + pa * a + ip(py, y))
    end
    JuMP.@objective(ø, Min, sum(e) + o)
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        error(" in gen_cut(): $status")
    else
        return JuMP.objective_value(ø), JuMP.dual(ℵaΓ), JuMP.dual.(ℵyΓ)
    end
end
function c2g(aΓ, yΓ, t) # for end_stage;;;
    @assert t == T
    ø = JumpModel(0)
    JuMP.@variable(ø, 5. <= D[1:I] <= 25.) # outer MAX layer
    JuMP.@variable(ø, ℵa)
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵD[1:I])
    JuMP.@variable(ø, ℵH[1:I] >= 0.)
    JuMP.@variable(ø, ℵB[1:I] >= 0.)
    JuMP.@constraint(ø, a, ℵM - ℵa == 0.)
    JuMP.@constraint(ø, e[i = 1:I], 1. - (ℵB[i] + ℵH[i]) == 0.)
    JuMP.@constraint(ø, y[i = 1:I], ℵH[i] *  H[t, i] - ℵB[i] * B[t, i] - ℵD[i] == 0.)
    JuMP.@constraint(ø, u[i = 1:I], ℵD[i] + ℵa * C[t, i] >= 0.)  
    JuMP.@objective(ø, Max, aΓ * ℵa - M * ℵM + ip(yΓ, ℵD) - ip(ℵD, D))
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    @assert status == JuMP.OPTIMAL " in c2g(end_stage): $status "
    cost_worst = JuMP.objective_value(ø)
    Dworst = JuMP.value.(D)
    bv1 = isapprox.(Dworst, 5.; atol = 1e-5)
    bv2 = isapprox.(Dworst, 25.; atol = 1e-5)
    if !(all(bv1 .|| bv2))
        error("in c2g(end_stage): worst D is not Ext Point: $Dworst")
    else
        return cost_worst, Dworst
    end 
end
function c2g(aΓ, yΓ, t, Δ) # used both in FW and BW
    function c2g_feas(aΓ, yΓ, t, aL, yL, L)
        function c2g_primal_feas(aΓ, yΓ, t, aL, yL, D) # primal max-min formulation for reference
            # λ-method part
            JuMP.@variable(ø, λ[1:L] >= 0.)
            JuMP.@variable(ø, aΔ)
            JuMP.@variable(ø, yΔ[1:I])
            JuMP.@constraint(ø, ℵλ, sum(λ) == 1.)
            JuMP.@constraint(ø, ℵaΔ, aΔ == ip(aL, λ))
            JuMP.@constraint(ø, ℵyΔ[i = 1:I], yΔ[i] == sum(yL[l][i] * λ[l] for l in 1:L))
            # primal feasible region part
            JuMP.@variable(ø, a) # 🏹
            JuMP.@variable(ø, y[1:I]) # 🏹
            JuMP.@variable(ø, u[1:I] >= 0.)
            JuMP.@constraint(ø, ℵM, M >= a)
            JuMP.@constraint(ø, ℵa, a == aΓ + ip(C[t, :], u))
            JuMP.@constraint(ø, ℵD[i = 1:I], y[i] == yΓ[i] - D[i] + u[i])
            # minimize distance
            JuMP.@variable(ø, ea)
            JuMP.@variable(ø, ey[1:I])
            JuMP.@constraint(ø, ℵea1, ea >= a - aΔ)
            JuMP.@constraint(ø, ℵea2, ea >= aΔ - a)
            JuMP.@constraint(ø, ℵey1[i = 1:I], ey[i] >= y[i] - yΔ[i])
            JuMP.@constraint(ø, ℵey2[i = 1:I], ey[i] >= yΔ[i] - y[i])
            JuMP.@objective(ø, Min, ea + sum(ey))
        end
        function c2g_dual_feas(aΓ, yΓ, t, aL, yL, D)
            JuMP.@variable(ø, ℵea1 >= 0.)
            JuMP.@variable(ø, ℵea2 >= 0.)
            JuMP.@variable(ø, ℵey1[1:I] >= 0.)
            JuMP.@variable(ø, ℵey2[1:I] >= 0.)
            JuMP.@variable(ø, ℵM >= 0.)
            JuMP.@variable(ø, ℵλ)
            JuMP.@variable(ø, ℵaΔ)
            JuMP.@variable(ø, ℵyΔ[1:I])
            JuMP.@variable(ø, ℵa)
            JuMP.@variable(ø, ℵD[1:I])
            JuMP.@constraint(ø, λ[l = 1:L], ℵaΔ * aL[l] - ℵλ + sum( ℵyΔ[i] * yL[l][i] for i in 1:I ) >= 0.)
            JuMP.@constraint(ø, u[i = 1:I], ℵa * C[t, i] + ℵD[i] >= 0.)
            JuMP.@constraint(ø, yΔ[i = 1:I], ℵey2[i] - ℵey1[i] - ℵyΔ[i] == 0.)
            JuMP.@constraint(ø, ey[i = 1:I], 1. - ℵey1[i] - ℵey2[i] == 0.)
            JuMP.@constraint(ø, y[i = 1:I], ℵey1[i] - ℵey2[i] - ℵD[i] == 0.)
            JuMP.@constraint(ø, aΔ, ℵea2 - ℵea1 - ℵaΔ == 0.)
            JuMP.@constraint(ø, ea, 1. - ℵea1 - ℵea2 == 0.)
            JuMP.@constraint(ø, a, ℵM - ℵa + ℵea1 - ℵea2 == 0.)
            JuMP.@objective(ø, Max, ℵλ - M * ℵM + ℵa * aΓ + sum(ℵD[i] * (yΓ[i] - D[i]) for i in 1:I))
        end
        ø = JumpModel(0)
        JuMP.@variable(ø, 5. <= D[1:I] <= 25.) # outer MAX layer
        JuMP.@variable(ø, ℵea1 >= 0.)
        JuMP.@variable(ø, ℵea2 >= 0.)
        JuMP.@variable(ø, ℵey1[1:I] >= 0.)
        JuMP.@variable(ø, ℵey2[1:I] >= 0.)
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵλ)
        JuMP.@variable(ø, ℵaΔ)
        JuMP.@variable(ø, ℵyΔ[1:I])
        JuMP.@variable(ø, ℵa)
        JuMP.@variable(ø, ℵD[1:I])
        JuMP.@constraint(ø, λ[l = 1:L], ℵaΔ * aL[l] - ℵλ + sum( ℵyΔ[i] * yL[l][i] for i in 1:I ) >= 0.)
        JuMP.@constraint(ø, u[i = 1:I], ℵa * C[t, i] + ℵD[i] >= 0.)
        JuMP.@constraint(ø, yΔ[i = 1:I], ℵey2[i] - ℵey1[i] - ℵyΔ[i] == 0.)
        JuMP.@constraint(ø, ey[i = 1:I], 1. - ℵey1[i] - ℵey2[i] == 0.)
        JuMP.@constraint(ø, y[i = 1:I], ℵey1[i] - ℵey2[i] - ℵD[i] == 0.)
        JuMP.@constraint(ø, aΔ, ℵea2 - ℵea1 - ℵaΔ == 0.)
        JuMP.@constraint(ø, ea, 1. - ℵea1 - ℵea2 == 0.)
        JuMP.@constraint(ø, a, ℵM - ℵa + ℵea1 - ℵea2 == 0.)
        JuMP.@objective(ø, Max, ℵλ - M * ℵM + ℵa * aΓ + sum(ℵD[i] * (yΓ[i] - D[i]) for i in 1:I))
        JuMP.optimize!(ø)
        status = JuMP.termination_status(ø)
        @assert status == JuMP.OPTIMAL "in c2g_feas: #1 status"
        min_distance, Dworst = JuMP.objective_value(ø), JuMP.value.(D)
        if min_distance > 1e-5
            bv1 = isapprox.(Dworst, 5.; atol = 1e-5)
            bv2 = isapprox.(Dworst, 25.; atol = 1e-5)
            if !(all(bv1 .|| bv2))
                error("in c2g_feas: #2 worst D is not Ext Point: $Dworst")
            else
                return Inf, Dworst
            end
        else
            return min_distance, Dworst
        end
    end
    function c2g_opt_primal()
        JuMP.@variable(ø, λ[1:L] >= 0.)
        JuMP.@variable(ø, u[1:I] >= 0.)
        JuMP.@variable(ø, e[1:I])
        JuMP.@variable(ø, y[1:I])
        JuMP.@variable(ø, a)
        JuMP.@variable(ø, o)
        JuMP.@constraint(ø, ℵH[i = 1:I], e[i] >=  H[t, i] * y[i])
        JuMP.@constraint(ø, ℵB[i = 1:I], e[i] >= -B[t, i] * y[i])
        JuMP.@constraint(ø, ℵo, o >= ip(fL, λ))
        JuMP.@constraint(ø, ℵM, M >= a)
        JuMP.@constraint(ø, ℵyΔ[i = 1:I], y[i] == sum(yL[l][i] * λ[l] for l in 1:L))
        JuMP.@constraint(ø, ℵD[i = 1:I], y[i] == yΓ[i] - D[i] + u[i])
        JuMP.@constraint(ø, ℵa, a == aΓ + ip(C[t, :], u))
        JuMP.@constraint(ø, ℵaΔ, a == ip(aL, λ))
        JuMP.@constraint(ø, ℵλ, sum(λ) == 1.)
        JuMP.@objective(ø, Min, sum(e) + o)
    end  
    function c2g_opt_dual()
        JuMP.@variable(ø, ℵH[1:I] >= 0.)
        JuMP.@variable(ø, ℵB[1:I] >= 0.)
        JuMP.@variable(ø, ℵo >= 0.)
        JuMP.@variable(ø, ℵM >= 0.)
        JuMP.@variable(ø, ℵyΔ[1:I])
        JuMP.@variable(ø, ℵD[1:I])
        JuMP.@variable(ø, ℵa)
        JuMP.@variable(ø, ℵaΔ)
        JuMP.@variable(ø, ℵλ)
        JuMP.@constraint(ø, λ[l = 1:L], ℵaΔ * aL[l] - ℵλ + sum(ℵyΔ[i] * yL[l][i] for i in 1:I) + ℵo * fL[l] >= 0.)
        JuMP.@constraint(ø, u[i = 1:I], ℵa * C[t, i] + ℵD[i] >= 0.)
        JuMP.@constraint(ø, e[i = 1:I], 1. - ℵH[i] - ℵB[i] == 0.)
        JuMP.@constraint(ø, y[i = 1:I], ℵH[i] * H[t, i] - ℵB[i] * B[t, i] - ℵyΔ[i] - ℵD[i] == 0.)
        JuMP.@constraint(ø, a, ℵM - ℵa - ℵaΔ == 0.)
        JuMP.@constraint(ø, o, 1. - ℵo == 0.)
        JuMP.@objective(ø, Max, ℵλ + ℵa * aΓ - ℵM * M + sum(ℵD[i] * (yΓ[i] - D[i]) for i in 1:I))
    end
    @assert t in 2:T-1
    if isempty(Δ["f"])
        return Inf, [randDext() for i in 1:I]
    else
        aL, yL, fL = Δ["a"], Δ["y"], Δ["f"]
        L = length(fL)
    end
    ret = c2g_feas(aΓ, yΓ, t, aL, yL, L)
    ret[1] == Inf && return ret
    ø = JumpModel(0) # 📚 main (reformulated) Max-min program
    JuMP.@variable(ø, 5. <= D[1:I] <= 25.)
    JuMP.@variable(ø, ℵH[1:I] >= 0.)
    JuMP.@variable(ø, ℵB[1:I] >= 0.)
    JuMP.@variable(ø, ℵo >= 0.)
    JuMP.@variable(ø, ℵM >= 0.)
    JuMP.@variable(ø, ℵyΔ[1:I])
    JuMP.@variable(ø, ℵD[1:I])
    JuMP.@variable(ø, ℵa)
    JuMP.@variable(ø, ℵaΔ)
    JuMP.@variable(ø, ℵλ)
    JuMP.@constraint(ø, λ[l = 1:L], ℵaΔ * aL[l] - ℵλ + sum(ℵyΔ[i] * yL[l][i] for i in 1:I) + ℵo * fL[l] >= 0.)
    JuMP.@constraint(ø, u[i = 1:I], ℵa * C[t, i] + ℵD[i] >= 0.)
    JuMP.@constraint(ø, e[i = 1:I], 1. - ℵH[i] - ℵB[i] == 0.)
    JuMP.@constraint(ø, y[i = 1:I], ℵH[i] * H[t, i] - ℵB[i] * B[t, i] - ℵyΔ[i] - ℵD[i] == 0.)
    JuMP.@constraint(ø, a, ℵM - ℵa - ℵaΔ == 0.)
    JuMP.@constraint(ø, o, 1. - ℵo == 0.)
    JuMP.@objective(ø, Max, ℵλ + ℵa * aΓ - ℵM * M + sum(ℵD[i] * (yΓ[i] - D[i]) for i in 1:I))
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        error("in c2g main: #1 $status")
    else
        ub, Dworst = JuMP.objective_value(ø), JuMP.value.(D)
        bv1 = isapprox.(Dworst, 5.; atol = 1e-5)
        bv2 = isapprox.(Dworst, 25.; atol = 1e-5)
        if !(all(bv1 .|| bv2))
            error("in c2g main: #2 worst D is not Ext Point: $Dworst")
        end
        return ub, Dworst
    end
end
function eval_Δ_at(a, y, Δ) # used in termination assessment
    isempty(Δ["f"]) && return Inf
    aL, yL, fL = Δ["a"], Δ["y"], Δ["f"]
    L = length(fL)
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:L] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, a == ip(aL, λ))
    JuMP.@constraint(ø, [i = 1:I], y[i] == sum(yL[l][i] * λ[l] for l in 1:L))
    JuMP.@objective(ø, Min, ip(fL, λ))
    JuMP.optimize!(ø)
    status = JuMP.termination_status(ø)
    if status != JuMP.OPTIMAL
        return Inf
    else
        return JuMP.objective_value(ø)
    end
end

# containers for iteration
D = NaN * ones(I, T) # (approximate) worst-case distributions
aFv, yFv = NaN * ones(T-1), NaN * ones(I, T-1) # state variable
lbFv  = NaN * ones(T-1)
ubFv  = NaN * ones(T-1)
glbFv = NaN * ones(T-1) # is only used at [1]
termination_flag = falses(1)
for ite in 1:typemax(Int)
    t = 1
    D[:, t] = D1
    glbFv[t], lbFv[t], aFv[t], yFv[:, t] = v(a_at_0, y_at_0, t, ℶ[t], D[:, t])
    lb = glbFv[1]
    ub = lb - lbFv[1] + eval_Δ_at(aFv[1], yFv[:, 1], Δ[1])
    @info "▶ ite = $ite, lb = $lb | $ub = ub"
    if lb + 1e-4 > ub
        @info "▶▶▶ algorithm converges"
        termination_flag[1] = true
    end
    for t in 2:T-1
        ubFv[t-1], D[:, t] = c2g(aFv[t-1], yFv[:, t-1], t, Δ[t])
        glbFv[t], lbFv[t], aFv[t], yFv[:, t] = v(aFv[t-1], yFv[:, t-1], t, ℶ[t], D[:, t])
    end
    # ---- backward starts ----
    t = T # t = 7
    ubFv[t-1], D[:, t] = c2g(aFv[t-1], yFv[:, t-1], t)
    termination_flag[1] && break
    push!(Δ[t-1]["a"], aFv[t-1])
    push!(Δ[t-1]["y"], yFv[:, t-1])
    push!(Δ[t-1]["f"], ubFv[t-1]) # don't have to check, as t = T is always accurate
    raw_obj, pa, py = gen_cut(aFv[t-1], yFv[:, t-1], t, ℶempty, D[:, t])
    push!(ℶ[t-1]["cn"], raw_obj - pa * aFv[t-1] - ip(py, yFv[:, t-1]))
    push!(ℶ[t-1]["pa"], pa)
    push!(ℶ[t-1]["py"], py)
    for t in T-1:-1:2
        ubBv_tm1, D[:, t] = c2g(aFv[t-1], yFv[:, t-1], t, Δ[t])
        if ubBv_tm1 < Inf
            push!(Δ[t-1]["a"], aFv[t-1])
            push!(Δ[t-1]["y"], yFv[:, t-1])
            push!(Δ[t-1]["f"], ubBv_tm1)
        end
        ubFv[t-1] = ubBv_tm1
        raw_obj, pa, py = gen_cut(aFv[t-1], yFv[:, t-1], t, ℶ[t], D[:, t])
        push!(ℶ[t-1]["cn"], raw_obj - pa * aFv[t-1] - ip(py, yFv[:, t-1]))
        push!(ℶ[t-1]["pa"], pa)
        push!(ℶ[t-1]["py"], py)
    end
end

[ Info: ▶ ite = 1, lb = -Inf | NaN = ub
[ Info: ▶ ite = 2, lb = 214.30379145575182 | Inf = ub
[ Info: ▶ ite = 3, lb = 548.0091574110922 | Inf = ub
[ Info: ▶ ite = 4, lb = 589.7783540851352 | Inf = ub
[ Info: ▶ ite = 5, lb = 885.9578558388391 | Inf = ub
[ Info: ▶ ite = 6, lb = 1024.4393024712954 | Inf = ub
[ Info: ▶ ite = 7, lb = 1377.0750217650593 | Inf = ub
[ Info: ▶ ite = 8, lb = 1377.0750217650593 | Inf = ub
[ Info: ▶ ite = 9, lb = 1377.0750217650593 | Inf = ub
[ Info: ▶ ite = 10, lb = 1638.9276163189734 | Inf = ub
[ Info: ▶ ite = 11, lb = 1638.9276163189734 | Inf = ub
[ Info: ▶ ite = 12, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 13, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 14, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 15, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 16, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 17, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 18, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 19, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 20, lb = 2192.674893919585 | Inf = ub
[ Info: ▶ ite = 21, lb = 2290.5402064776827 | Inf = ub
[ Info: ▶ ite = 22, lb = 2290.5402064776827 | Inf = ub
[ Info: ▶ ite = 23, lb = 2290.5402064776827 | Inf = ub
[ Info: ▶ ite = 24, lb = 2553.1688523446023 | Inf = ub
[ Info: ▶ ite = 25, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 26, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 27, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 28, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 29, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 30, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 31, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 32, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 33, lb = 2932.300170470358 | Inf = ub
[ Info: ▶ ite = 34, lb = 2971.73321194687 | Inf = ub
[ Info: ▶ ite = 35, lb = 2976.9644935875804 | Inf = ub
[ Info: ▶ ite = 36, lb = 2976.9644935875817 | Inf = ub
[ Info: ▶ ite = 37, lb = 2976.9644935875817 | Inf = ub
[ Info: ▶ ite = 38, lb = 2976.9644935875817 | Inf = ub
[ Info: ▶ ite = 39, lb = 2976.9644935875817 | Inf = ub
[ Info: ▶ ite = 40, lb = 2976.9644935875817 | 2976.964493587589 = ub
[ Info: ▶▶▶ algorithm converges







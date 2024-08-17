import LinearAlgebra
import Distributions
import Random
import JuMP
import Gurobi
using Logging

# DRO problem min_{x in 𝒳} sup_{ℙ in ℱ} <ℙ, f(x, Z)>
# Depiction:
# ℱ is characterized by a set of (potentially ∞-ly many) q's
# The ambiguity set characterized by qMat (signified by ℱ_finite) is the current surrogate of ℱ
# [★] The DRO specified by ℱ_finite is solvable, leading to a trial_x, then a worst case PD
# if this PD is in ℱ, then this trial_x is Optimal, thus return
# if this PD is not in ℱ, we embellish ℱ_finite by finding a `q` who can cut this PD off, goto [★]
# Instance: SOCP
# 17/08/24

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function JumpModel(i)
    if i == 0
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        # m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # for callbacks
    elseif i == 1
        m = JuMP.Model(MosekTools.Optimizer)
    end
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function initial_trial_x_generation() # x in 𝒳
    m = JumpModel(0)
    JuMP.@variable(m, x[1:M] >= 0.)
    JuMP.@constraint(m, sum(x) == 1.)
    JuMP.optimize!(m) # without an obj
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value.(x)
end
function quadForm(q, M)
    q' * M * q
end
function club_suit(x, qMat)
    J = size(qMat)[2]
    m = JumpModel(0) # a conic optimization
    JuMP.@variable(m, ξ[1:N, 1:K])
    JuMP.@variable(m, ζ[1:J, 1:K])
    JuMP.@variable(m, η[1:K] >= 0.)
    JuMP.@constraint(m, sum(η) == 1.)
    JuMP.@constraint(m, [n = 1:N], sum(ξ[n, k] for k in 1:K) == mu[n])
    JuMP.@constraint(m, [j = 1:J], sum(ζ[j, k] for k in 1:K) <= sigma' * qMat[:, j].^2)
    JuMP.@constraint(m, [n = 1:N, k = 1:K], ξ[n, k] <= rd_BND * η[k] )
    JuMP.@constraint(m, [n = 1:N, k = 1:K], -rd_BND * η[k] <= ξ[n, k])
    JuMP.@constraint(m, [j = 1:J, k = 1:K], [ζ[j, k] + η[k]; ζ[j, k] - η[k]; 2. * qMat[:, j]' * (ξ[:, k] .- η[k] * mu)] in JuMP.SecondOrderCone())
    JuMP.@objective(m, Max, sum((S[:, :, k] * x .+ tS[:, k])' * ξ[:, k] + (s[:, k]' * x + ts[k]) * η[k] for k in 1:K))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m), JuMP.value.(ξ), JuMP.value.(η)
end
function worst_pd(x, qMat) # after the accomplishment of the cutting plane method
    _, ξ, η = club_suit(x, qMat)
    bitVec = η .> 7e-5 # omit those events with negligible Pr
    ξ, η = ξ[:, bitVec], η[bitVec]
    Dict(
        "z" => [ξ[:, i] / η[i] for i in eachindex(η)],
        "P" => η
    )
end
function cut_at(x, qMat)
    obj, ξ, η = club_suit(x, qMat)
    px::Vector{Float64} = sum(S[:, :, k]' * ξ[:, k] .+ η[k] * s[:, k] for k in 1:K)
    p0::Float64 = sum(tS[:, k]' * ξ[:, k] + ts[k] * η[k] for k in 1:K)
    obj, px, p0
end
function outer_surrogate(cutDict)
    m = JumpModel(0) # cutting plane optimization
    JuMP.@variable(m, x[1:M] >= 0.)
    JuMP.@constraint(m, sum(x) == 1.)
    JuMP.@variable(m, th)
    for (px, p0) in zip(cutDict["px"], cutDict["p0"])
        JuMP.@constraint(m, th >= px' * x + p0)
    end
    JuMP.@objective(m, Min, th)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(th), JuMP.value.(x)
end
function outer_solve(qMat)::Vector{Float64}
    xt =  initial_trial_x_generation()
    ub, px, p0 = cut_at(xt, qMat) # add an initial cut to assure lower boundedness
    cutDict = Dict( # initialize cutDict
        "id" => Int[1],
        "at" => Vector{Float64}[xt],
        "px" => Vector{Float64}[px],
        "p0" => Float64[p0]
    )
    for ite in 1:typemax(Int)
        lb, xt = outer_surrogate(cutDict)
        ub, px, p0 = cut_at(xt, qMat)
        @assert lb <= ub "lb > ub"
        gap = abs((ub-lb)/ub)
        @debug "▶ ▶ ▶ ite = $ite" lb ub gap=(ub-lb)/abs(ub)
        if gap < 1e-4
            @info " 😊 gap < 0.01% " ub
            return xt
        end
        push!(cutDict["id"], 1+length(cutDict["id"]))
        push!(cutDict["at"], xt)
        push!(cutDict["px"], px)
        push!(cutDict["p0"], p0)
    end
end
function find_q(pd::Dict)
    function my_callback_function(cb_data, cb_where::Cint)
        if cb_where == Gurobi.GRB_CB_MIPSOL
            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            qSol = JuMP.callback_value.(cb_data, q)
            if quadForm(qSol, objMat) < -1e-5
                if LinearAlgebra.norm(qSol) <= 1. + 1e-5
                    qt .= qSol # a solution q that violates is found
                    Gurobi.GRBterminate(JuMP.backend(m)) 
                    return nothing
                end
            end
        end
        return nothing
    end
    objMat = deepcopy(Σ)
    for (z, P) in zip(pd["z"], pd["P"])
        objMat -= P * (z-mu) * (z-mu)'
    end
    @debug LinearAlgebra.eigvals(objMat)
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # direct with callback, might be non-convex QCQP
    JuMP.set_silent(m)
    JuMP.@variable(m, q[1:N])
    JuMP.@constraint(m, sum(q[n]^2 for n in 1:N) <= 1.) # quad constr
    JuMP.@objective(m, Min, quadForm(q, objMat))
    JuMP.MOI.set(m, Gurobi.CallbackFunction(), my_callback_function)
    qt = 3. * ones(N)
    JuMP.optimize!(m)
    if qt[1] < 2.
        return quadForm(qt, objMat), qt
    elseif JuMP.termination_status(m) == JuMP.JuMP.OPTIMAL
        if JuMP.objective_value(m) < -1e-5
            error("we have a q that violates, but is not found!")
        else
            @info " 😊😊 opt-pbj is nonnegative, thus the current q's characterizing the ambiguity set are sufficient. "
            return missing, missing
        end
    else
        error("we haven't either find a violating q, or solving the nonconvex program to optimal.")
    end
end
function main()
    for Ite in 1:typemax(Int)
        global qMat, xt
        xt = outer_solve(qMat)
        x_incumbent .= xt # update
        pd = worst_pd(xt, qMat)
        val, q = find_q(pd)
        if typeof(val) == Missing
            @info "the optimal solution is" xt
            @info "check the q's" qMat
            return xt
        end
        qMat = [qMat q] # update
    end
end
if true # problem data gen
    # K is the num of obj-pieces; M = len(x); N = len(z) = len(q)
    # K, M, N = 3, 4, 5 
    K, M, N = 6, 8, 10
    mu = zeros(N) # mean gen
    rd_BND = 50. # support gen
    # obj f data gen
    d = Distributions.Normal()
    Random.seed!(996)
    S = rand(d, N, M, K)
    tS = rand(d, N, K)
    s = rand(d, M, K)
    ts = rand(d, K)
    # moment constrs data gen
    d = Distributions.Uniform(0., 7.071)
    sigma = rand(d, N)
    Σ = LinearAlgebra.Diagonal(sigma.^2)
    # d = Distributions.Uniform(1., 2.)
    # epsilon = rand(d, N)
    # kappa = epsilon .* sigma # used when 4th moment
    if true # global containers
        qMat = one(ones(N, N)) # initial q's gen
        x_incumbent = initial_trial_x_generation()
    end
end

main()


import LinearAlgebra
import Distributions
import Random
import JuMP
import Gurobi
using Logging

# 4th-moment DRO problem min_{x in 𝒳} sup_{ℙ in ℱ} <ℙ, f(x, Z)>
# SOCP + nonconvex QCQP is enough to deal with 4th moment, therefore Gurobi per se is enough
# although we can cast ^4 terms as QCQP, Gurobi may print some warnings. Nonetheless it can still solve, therefore we can omit this warning.
# These are some potential improving methods:
# we don't have to seek both 2nd new q and 4th new q at every iteration, in that the latter requires much more time
# we can firstly seek new q's for 2nd constraint to locate the trial_x into a more steady region, then start add new q for 4th.
# For solving the non-convex QCQP problem: we can terminate when a feasible q with < 0 obj is found.
# we can also retrieve the MIPSOL_OBJBND ( current best objective bound ) at my callback function. when this is very high, e.g., 1e-5, then we can guess that it's hopeless to find a feasible q with < 0 obj, thus continue the main loop.
# if no more q's can be found, then we accomplish this algorithm.
# 18/08/24

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
function club_suit(x, qs) #  🫠
    function q(r, j) qs[r][:, j] end # pick the q vector in row r, number j
    J2, J4 = size(qs[1])[2], size(qs[2])[2]
    m = JumpModel(0) # a conic optimization
    JuMP.@variable(m, ξ[1:N, 1:K])
    JuMP.@variable(m, η[1:K] >= 0.)
    JuMP.@constraint(m, sum(η) == 1.)
    JuMP.@variable(m, ζ[1:J2, 1:K]) # 2nd
    JuMP.@variable(m, ϑ[1:J4, 1:K]) # 4th
    JuMP.@variable(m, ψ[1:J4, 1:K]) # ancillary of 4th
    JuMP.@constraint(m, [n = 1:N], sum(ξ[n, :]) == 0.)
    JuMP.@constraint(m, [n = 1:N, k = 1:K], -rd_BND * η[k] <= ξ[n, k])
    JuMP.@constraint(m, [n = 1:N, k = 1:K], ξ[n, k] <= rd_BND * η[k] )
    JuMP.@constraint(m, [j = 1:J2], sum(ζ[j, :]) <= quadForm(q(1, j), Σ))
    JuMP.@constraint(m, [j = 1:J2, k = 1:K], [ζ[j, k] + η[k]; ζ[j, k] - η[k]; 2. * q(1, j)' * ξ[:, k]] in JuMP.SecondOrderCone())
    JuMP.@constraint(m, [j = 1:J4], sum(ϑ[j, :]) <= phi(q(2, j)) )
    JuMP.@constraint(m, [j = 1:J4, k = 1:K], [ϑ[j, k] + η[k]; ϑ[j, k] - η[k]; 2. * ψ[j, k]] in JuMP.SecondOrderCone())
    JuMP.@constraint(m, [j = 1:J4, k = 1:K], [ψ[j, k] + η[k]; ψ[j, k] - η[k]; 2. * q(2, j)' * ξ[:, k]] in JuMP.SecondOrderCone())
    JuMP.@objective(m, Max, sum((S[:, :, k] * x .+ tS[:, k])' * ξ[:, k] + (s[:, k]' * x + ts[k]) * η[k] for k in 1:K))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL " ♣ SOCP: $(JuMP.termination_status(m))"
    JuMP.objective_value(m), JuMP.value.(ξ), JuMP.value.(η)
end
function worst_pd(x, qs) # 🫠 after the accomplishment of the cutting plane method
    _, ξ, η = club_suit(x, qs)
    bitVec = η .> 7e-5 # omit those events with negligible Pr
    ξ, η = ξ[:, bitVec], η[bitVec]
    Dict(
        "z" => [ξ[:, i] / η[i] for i in eachindex(η)],
        "P" => η
    )
end
function cut_at(x, qs) # 🫠
    obj, ξ, η = club_suit(x, qs)
    px::Vector{Float64} = sum(S[:, :, k]' * ξ[:, k] .+ η[k] * s[:, k] for k in 1:K)
    p0::Float64 = sum(tS[:, k]' * ξ[:, k] + ts[k] * η[k] for k in 1:K)
    obj, px, p0
end
function outer_surrogate(cutDict) # 🫠
    m = JumpModel(0) # cutting plane optimization
    JuMP.@variable(m, x[1:M] >= 0.)
    JuMP.@constraint(m, sum(x) == 1.)
    JuMP.@variable(m, th)
    for (px, p0) in zip(cutDict["px"], cutDict["p0"])
        JuMP.@constraint(m, th >= px' * x + p0)
    end
    JuMP.@objective(m, Min, th)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL " outer_surrogate_LP: $(JuMP.termination_status(m))"
    JuMP.value(th), JuMP.value.(x)
end
function outer_solve(qs)::Vector{Float64} # 🫠
    xt =  initial_trial_x_generation()
    ub, px, p0 = cut_at(xt, qs) # add an initial cut to assure lower boundedness
    cutDict = Dict( # initialize cutDict
        "id" => Int[1],
        "at" => Vector{Float64}[xt],
        "px" => Vector{Float64}[px],
        "p0" => Float64[p0]
    )
    for ite in 1:typemax(Int)
        lb, xt = outer_surrogate(cutDict)
        ub, px, p0 = cut_at(xt, qs)
        @assert lb <= ub "lb > ub"
        gap = abs((ub-lb)/ub)
        @debug "▶ ▶ ▶ ite = $ite" lb ub gap=(ub-lb)/abs(ub)
        if gap < 1e-4
            @info " 😊 outer problem gap < 0.01% " ub
            return xt
        end
        push!(cutDict["id"], 1+length(cutDict["id"]))
        push!(cutDict["at"], xt)
        push!(cutDict["px"], px)
        push!(cutDict["p0"], p0)
    end
end
function find_q_2nd(pd::Dict)
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
        objMat -= P * z * z'
    end
    @debug LinearAlgebra.eigvals(objMat)
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # direct with callback, might be non-convex QCQP
    JuMP.set_silent(m)
    JuMP.@variable(m, q[1:N])
    JuMP.@constraint(m, sum(q[n]^2 for n in 1:N) <= 1.) # quad constr
    JuMP.@objective(m, Min, quadForm(q, objMat))
    JuMP.MOI.set(m, Gurobi.CallbackFunction(), my_callback_function)
    qt = 3. * ones(N)
    @info "2nd non-convex"
    JuMP.optimize!(m)
    if qt[1] < 2.
        return quadForm(qt, objMat), qt
    elseif JuMP.termination_status(m) == JuMP.OPTIMAL
        if JuMP.objective_value(m) < -1e-4
            error("we have a q that violates, but is not found!")
        else
            @info " 😊😊 opt-pbj is nonnegative, thus the current q's characterizing the ambiguity set are sufficient. "
            return missing, missing
        end
    else
        error("we haven't either find a violating q, or solving the nonconvex program to optimal.")
    end
end
function find_q_4th(pd::Dict)
    function objfun(q, pd)
        phi(q) - sum(P * (z' * q)^4 for (z, P) in zip(pd["z"], pd["P"])) 
    end
    function my_callback_function(cb_data, cb_where::Cint)
        if cb_where == Gurobi.GRB_CB_MIPSOL
            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            qSol = JuMP.callback_value.(cb_data, q)
            if objfun(qSol, pd) < -1e-5
                if LinearAlgebra.norm(qSol) <= 1. + 1e-5
                    qt .= qSol # a solution q that violates is found
                    Gurobi.GRBterminate(JuMP.backend(m)) 
                    return nothing
                end
            end
        end
        return nothing
    end
    C = length(pd["P"]) # there are C points in the worst_PD
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # direct with callback, might be non-convex QCQP
    # JuMP.set_silent(m)
    JuMP.@variable(m, q[1:N])
    JuMP.@variable(m, qsquare[1:N]) # can be used only in part1
    JuMP.@constraint(m, [n = 1:N], q[n]^2 <= qsquare[n]) 
    JuMP.@variable(m, objpart1_2[1:N]) # >= sth
    JuMP.@constraint(m, [n in 1:N], sum((sigma[n] * sigma[i])^2 * qsquare[i] * qsquare[n] for i in 1:N if i != n) <= objpart1_2[n])
    JuMP.@variable(m, objpart1_1[1:N]) # >= sth
    JuMP.@constraint(m, [n in 1:N], (kappa[n]^2 * qsquare[n])^2 <= objpart1_1[n])
    JuMP.@variable(m, part2middle[1:C]) # to break 4th down
    JuMP.@constraint(m, [c in 1:C], part2middle[c] == (pd["z"][c]' * q) ^ 2)
    JuMP.@variable(m, objpart2[1:C]) # == sth
    JuMP.@constraint(m, [c in 1:C], objpart2[c] == pd["P"][c] * part2middle[c] ^ 2)
    JuMP.@objective(m, Min, sum(objpart1_1[n] + 6. * objpart1_2[n] for n in 1:N) - sum(objpart2)) # 💡
    JuMP.@constraint(m, sum(q[n]^2 for n in 1:N) <= 1.) # norm <= 1
    JuMP.MOI.set(m, Gurobi.CallbackFunction(), my_callback_function)
    qt = 3. * ones(N)
    @info "4th non-convex"
    JuMP.optimize!(m)
    if qt[1] < 2.
        return objfun(qt, pd), qt
    elseif JuMP.termination_status(m) == JuMP.OPTIMAL
        if JuMP.objective_value(m) < -1e-4
            error("we have a q that violates, but is not found!")
        else
            @info " 😊😊😊😊 opt-pbj is nonnegative, thus the current q's characterizing the ambiguity set are sufficient. "
            return missing, missing
        end
    else
        error("we haven't either find a violating q, or solving the nonconvex program to optimal.")
    end
end
function main()
    for ite in 1:typemax(Int)
        global qs, xt
        xt = outer_solve(qs)
        x_incumbent .= xt # update
        pd = worst_pd(xt, qs)
        val2, q2 = find_q_2nd(pd)
        val4, q4 = find_q_4th(pd)
        if typeof(val4) == Missing
            if typeof(val2) == Missing
                @info "the optimal solution is" xt
                return xt
            elseif val2 < -1e-5
                qs[1] = [qs[1] q2]
            else
                error("4-2")
            end
        elseif val4 < -1e-5
            qs[2] = [qs[2] q4]
            if typeof(val2) != Missing && val2 < -1e-5
                qs[1] = [qs[1] q2]
            end
        else
            error("out of 4th")
        end
        @info " ▶ ▶ ▶ main ite = $(ite)" num_qs = (size(qs[1])[2], size(qs[2])[2])
    end
end
if true # problem data gen
    function quadForm(q, M) q' * M * q end
    function phi(q) sum((kappa[n] * q[n])^4 + 6. * sum((sigma[n] * sigma[m] * q[n] * q[m])^2 for m in 1:N if m != n) for n in 1:N) end
    function initial_trial_x_generation() # x in 𝒳
        m = JumpModel(0)
        JuMP.@variable(m, x[1:M] >= 0.)
        JuMP.@constraint(m, sum(x) == 1.)
        JuMP.optimize!(m) # without an obj
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL " initial_trial_x_gen: $(JuMP.termination_status(m))"
        JuMP.value.(x)
    end
    # K is the num of obj-pieces; M = len(x); N = len(z) = len(q)
    K, M, N = 3, 4, 5
    # K, M, N = 6, 8, 10
    rd_BND = 10. # support gen
    # obj f data gen
    d = Distributions.Normal()
    Random.seed!(996)
    S = rand(d, N, M, K)
    tS = rand(d, N, K)
    s = rand(d, M, K)
    ts = rand(d, K)
    # moment constrs data gen
    d = Distributions.Uniform(0., 5.)
    sigma = rand(d, N)
    Σ = LinearAlgebra.Diagonal(sigma.^2)
    d = Distributions.Uniform(1., 2.)
    epsilon = rand(d, N)
    kappa = epsilon .* sigma # used when 4th moment
    if true # global containers
        qs = Matrix{Float64}[one(ones(N, N)) for _ in 1:2] # qs[1] = qMat for 2nd moment whereas qs[2] = qMat for 4th moment
        x_incumbent = initial_trial_x_generation()
    end
end

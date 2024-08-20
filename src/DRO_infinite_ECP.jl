import JuMP
import Gurobi
import MosekTools
import Optim
import Random
using Logging

# DRO_infinite_ECP
# result:
# soltuion theta = 0.066711, x = [1.000000097388703, 0.0, -1.362894644522967e-7, 1.5082664962619764e-8, 0.0, -2.938014240752205e-8, 0.0, -2.0546616284617336e-8, 0.0, 0.0, 0.0, 0.0, -2.4691652869586892e-8, -2.0275509139942546e-8, 0.0, 4.383556681780431e-8, 1.581166875212968e-8, 0.0, 3.311643113653387e-8, 2.2815141114671097e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.632146075355219e-8, 1.9456542150000502e-8, 4.567272366355702e-9, -3.5054654924732855e-8, 0.0, 5.939321857563046e-9, 0.0, 0.0, 0.0, 2.3580824942963586e-8, 0.0, -5.2778413015427796e-9, 0.0, -2.5162411013659196e-8, 0.0, -2.0685635166732063e-8, 0.0, 0.0, 8.72307921269387e-9, -7.022114511968672e-9, 2.607041580577255e-8, 0.0, -1.778942393393379e-8, 0.0, 0.06671067234576002]
# At the 1st iteration, a violating `q` is found, but there's no more in later iterations.
# objective is 0.06671. the interesting point is that this value is virtually the value of theta
# 2024/8/20

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
if true # data gen
    function initial_trial_x_generation()
        a = zeros(N+1)
        a[1] = 1.
        a
    end
    function a1(formal_x) -sigma .* formal_x[begin:end-1] end
    function b1(formal_x) -[mu; 1]' * formal_x end
    N = 50
    CTPLN_INI_BND = -53.
    CTPLN_GAP_TOL = 5e-5
    VIO_BRINK = -5e-5
    mu = [n/250 for n in 1:N]
    sigma = [N * sqrt(2 * n) for n in 1:N]/1000
    ε = .1
    if true # global containers
        qs = one(ones(N, N))
        x_incumbent = initial_trial_x_generation() # the global xt (the incumbent)
    end
end
function JumpModel(i)
    if i == 0 # the most frequently used
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    elseif i == 1 # generic convex conic program
        m = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 # if you need Gurobi callback
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function ECP_sup_subprogram(formal_x, qs)
    K = 2
    J = size(qs)[2]
    m = JumpModel(1) # a conic optimization
    JuMP.@variable(m, η[1:K] >= 0.)
    JuMP.@constraint(m, sum(η) == 1.)
    JuMP.@variable(m, ξ[1:N, 1:K])
    JuMP.@constraint(m, [n = 1:N], sum(ξ[n, :]) == 0.)
    JuMP.@variable(m, ζ[1:J, 1:K])
    JuMP.@constraint(m, [j = 1:J], sum(ζ[j, :]) <= 1.)
    JuMP.@constraint(m, [n = 1:N, k = 1:K], -η[k] <= ξ[n, k])
    JuMP.@constraint(m, [n = 1:N, k = 1:K], ξ[n, k] <= η[k])
    JuMP.@constraint(m, [j = 1:J, k = 1:K], [qs[:, j]' * ξ[:, k] - η[k] * .5 * (qs[:, j]' * qs[:, j]), η[k], ζ[j, k]] in JuMP.MOI.ExponentialCone())
    JuMP.@objective(m, Max, a1(formal_x)' * ξ[:, 1] + b1(formal_x) * η[1])
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    if status == JuMP.OPTIMAL
        return JuMP.objective_value(m), JuMP.value.(ξ), JuMP.value.(η)
    elseif status == JuMP.SLOW_PROGRESS
        @warn "♣ ECP terminates with JuMP.SLOW_PROGRESS" ξ=JuMP.value.(ξ) objval=JuMP.objective_value(m) η=JuMP.value.(η) 
        return JuMP.objective_value(m), JuMP.value.(ξ), JuMP.value.(η)
    else
        error("the ♣ ECP terminate with $(status)")
    end
end
function cut_at(formal_x, qs)
    Q_value, ξ, η = ECP_sup_subprogram(formal_x, qs)
    ξ, η = ξ[:, 1], η[1]
    px::Vector{Float64} = -[ξ[n] * sigma[n] + η * mu[n] for n in 1:N] 
    pth::Float64 = -η
    p0::Float64 = 0.
    Q_value, [px; pth], p0
end
function outer_surrogate(cutDict)
    m = JumpModel(0) # cutting plane optimization
    JuMP.@variable(m, x[1:N] >= 0.)
    JuMP.@constraint(m, sum(x) == 1.)
    JuMP.@variable(m, θ)
    JuMP.@variable(m, th)
    for (px, p0) in zip(cutDict["px"], cutDict["p0"])
        JuMP.@constraint(m, th >= px' * [x; θ] + p0)
    end
    JuMP.@variable(m, obj)
    JuMP.@constraint(m, obj >= θ + th / ε)
    JuMP.@objective(m, Min, obj)
    JuMP.optimize!(m)
    status, cutDict_sufficient_flag = JuMP.termination_status(m), true
    if status in [JuMP.INFEASIBLE_OR_UNBOUNDED, JuMP.DUAL_INFEASIBLE]
        cutDict_sufficient_flag = false
        JuMP.set_lower_bound(obj, CTPLN_INI_BND)
        JuMP.optimize!(m)
        status = JuMP.termination_status(m)
    end
    if status != JuMP.OPTIMAL
        error(" flag=($cutDict_sufficient_flag), outer_surrogate: status = $status ")
    end
    if cutDict_sufficient_flag
        return JuMP.value(obj), JuMP.value.([x; θ])
    else
        return -Inf, JuMP.value.([x; θ])
    end
end
function outer_solve(qs)::Vector{Float64}
    xt = initial_trial_x_generation()
    _, px, p0 = cut_at(xt, qs) # add an initial cut to assure lower boundedness
    cutDict = Dict( # initialize cutDict
        "id" => Int[1],
        "at" => Vector{Float64}[xt],
        "px" => Vector{Float64}[px],
        "p0" => Float64[p0]
    )
    for ite in 1:typemax(Int)
        lb, xt = outer_surrogate(cutDict)
        θ = xt[end] # record current 1st stage solution
        Q_value, px, p0 = cut_at(xt, qs)
        ub = θ + Q_value / ε
        @assert lb <= ub + CTPLN_GAP_TOL "lb($(lb)) > ub($(ub)), check whether they are specified correctly!"
        gap = ub-lb
        @debug "▶ ▶ ▶ ite = $ite" lb ub gap
        if gap < CTPLN_GAP_TOL
            @info " 😊 outer problem ub - lb = $gap < $CTPLN_GAP_TOL, ub = $(ub)"
            return xt
        end
        push!(cutDict["id"], 1+length(cutDict["id"]))
        push!(cutDict["at"], xt)
        push!(cutDict["px"], px)
        push!(cutDict["p0"], p0)
    end
end
function worst_pd(x, qs) # after the accomplishment of the cutting plane method
    _, ξ, η = ECP_sup_subprogram(x, qs)
    @info "worst_pd: the probability is $η"
    # bitVec = η .> 7e-5 # omit those events with negligible Pr
    # ξ, η = ξ[:, bitVec], η[bitVec]
    Dict(
        "z" => [ξ[:, i] / η[i] for i in eachindex(η)],
        "P" => η
    )
end
function objfun(q, pd) .5 * q' * q - log(sum(P * exp(q' * z) for (z, P) in zip(pd["z"], pd["P"]))) end

Random.seed!(86)
for mainIte in 1:typemax(Int)
    xt = outer_solve(qs)
    x_incumbent .= xt
    pd = worst_pd(xt, qs)
    gen_res_at = q0 -> Optim.optimize(q -> objfun(q, pd), q0, Optim.NewtonTrustRegion())
    find_already = [false]
    vio_q = Inf * ones(N)
    for ite in 1:400
        q0 = 10 * rand() * (2 * (rand(N) .- .5))
        res = gen_res_at(q0)
        qt, vt = Optim.minimizer(res), Optim.minimum(res)
        if vt < VIO_BRINK
            vio_q .= qt
            find_already[1] = true
            break
        end
    end
    if find_already[1] == false
        @error "we can't find a vio_q at this stage"
        return xt
    else
        qs = [qs vio_q]
    end
end



############################################## deprecated ##############################################
function ERRoneous_find_q(pd::Dict) # ⚠️⚠️⚠️ Gurobi's NL function is revolting, If it's not QP, Don't use Gurobi
    column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
    function objfun(q, pd) .5 * q' * q - log(sum(P * exp(q' * z) for (z, P) in zip(pd["z"], pd["P"]))) end
    function my_callback_function(cb_data, cb_where::Cint)
        if cb_where == Gurobi.GRB_CB_MIPSOL
            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            qSol = JuMP.callback_value.(cb_data, q) # retrieve q prematurely
            objBnd = Ref{Cdouble}(-Inf)
            if objfun(qSol, pd) < -0.1 # a favorable q is already found
                qt .= qSol # a solution q that violates is found
                Gurobi.GRBterminate(JuMP.backend(m))
                return nothing
            else
                errorcode = Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJBND, objBnd)
                @assert errorcode == 0 "OBJBND: GRB_errorcode != 0"
                if objBnd[] > CRITICAL_VAL # recognized as obj(q) >= 0, for all q
                    @error "objBnd = $(objBnd[])"
                    qt .= -Inf # no violating q could be found
                    Gurobi.GRBterminate(JuMP.backend(m))
                    return nothing
                else # carry on optimizing
                    return nothing
                end
            end
        else
            return nothing
        end
    end
    CRITICAL_VAL = -0.9 # a lower value means more strict condition for a `q` who violates
    qt = Inf * ones(N) # initial is Inf, if objBnd >= 0, set it to -Inf then, elseif a vio q is found, set it to the q.
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # 💡Gurobi as a black box optimizer, we shall check the solution afterwards.
    # JuMP.set_silent(m)
    JuMP.@variable(m, q[1:N])
    JuMP.@variable(m, obj1)
    JuMP.@variable(m, expectation_)
    JuMP.@objective(m, Min, obj1 - expectation_)
    JuMP.@variable(m, phi)
    JuMP.@constraint(m, phi == .5 * q' * q)
    errcode = Gurobi.GRBaddgenconstrExp(JuMP.backend(m), "", column(phi), column(obj1), "")
    @assert errcode == 0 "GRB_errcode != 0"
    C = 2
    JuMP.@variable(m, expResult[1:C])
    JuMP.@constraint(m, expectation_ == sum(pd["P"][c] * expResult[c] for c in 1:C)) # we specify the `expectation_`
    JuMP.@variable(m, exponent_[1:C])
    for c in 1:C
        errcode = Gurobi.GRBaddgenconstrExp(JuMP.backend(m), "", column(exponent_[c]), column(expResult[c]), "")
        @assert errcode == 0 "GRB_errcode != 0"
    end
    JuMP.@constraint(m, [c = 1:C], exponent_[c] == q' * pd["z"][c])
    # JuMP.set_attribute(m, JuMP.MOI.RawOptimizerAttribute("FuncNonlinear"), 1) # crummy
    # JuMP.set_attribute(m, JuMP.MOI.RawOptimizerAttribute("FuncPieceError"), 1e-4) # crummy
    JuMP.set_attribute(m, JuMP.MOI.RawOptimizerAttribute("FuncPieces"), 60000)
    # JuMP.set_attribute(m, JuMP.MOI.RawOptimizerAttribute("FuncPieceLength"), 1e-2)
    # JuMP.MOI.set(m, Gurobi.CallbackFunction(), my_callback_function)
    JuMP.optimize!(m)
    JuMP.get_attribute(m, Gurobi.ModelAttribute("MaxVio"))
    if JuMP.termination_status(m) == JuMP.INTERRUPTED # this entry is for the more favorable cases
        @assert qt[1] != Inf "qt has not been modified by callback"
        @error "123"
        return qt
    elseif JuMP.termination_status(m) == JuMP.OPTIMAL # this entry is more ambiguous and thus slower, but it's definitive
        @error "here456"
        qt .= JuMP.value.(q) # we obtained a solution after all
        @error "qt" qt JuMP.objective_value(m)
        objval = objfun(qt, pd) # check its value
        if objval < CRITICAL_VAL
            return qt
        else
            return -Inf * ones(N)
        end
    else
        error(" NLP hell : $(JuMP.termination_status(m))")
    end
end







import JuMP
import Gurobi
import MosekTools
import Optim
import Random
import Distributions
import LinearAlgebra
import Symbolics
using Logging

# (Dual) ECP conic programs + NonConvex unconstrained NLP solving
# You don't have to manually input the gradient and hessian matrix, the black-box solver with AutoDiff is reliable
# This program can verify the validity of the GIP algorithm
# But the numerical case is not very well-designed, i.e., the design of `mu` and `sigma`
# To attain the desired result, you need to tune:
# 1. ε: affectes the trade-off relation ship between the 1st and 2nd stage cost
# 2. DEN: if under DEN = 2 you cannot find vio-q at an early stage, you might want to increase DEN
# 3. the initial set of q's, i.e. `qs`. If you involve those crucial q's at the initial stage, you might have attain a very good cost initially, thus cannot see the cost descend along the iterations.
# 25/8/24

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function ip(x, y) LinearAlgebra.dot(x, y) end # this function is more versatile than x' * y
if true # data gen
    function initial_trial_x_generation()
        a = zeros(N+1)
        a[5] = 1.
        a
    end
    function a1(formal_x) -sigma .* formal_x[begin:end-1] end
    function b1(formal_x) -[mu; 1]' * formal_x end
    N = 50
    K = 2
    CTPLN_INI_BND = -5.
    CTPLN_GAP_TOL = 5e-5
    mu = [n/250 for n in 1:N]
    sigma = [N * sqrt(2 * n) for n in 1:N]/1000
    ε = 0.05
    DEN = 4
    if true # global containers
        qs = -one(ones(N, N))
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

function oneshot_inf_program(qs) # 🫠
    J = size(qs)[2]
    ι = JumpModel(1)
    JuMP.@variable(ι, θ)
    JuMP.@variable(ι, x[1:N] >= 0.)
    JuMP.@constraint(ι, sum(x) == 1.)
    formal_x = [x; θ]
    if true # sub block of constrs and variables
        JuMP.@variable(ι, α)
        JuMP.@variable(ι, β[1:N])
        JuMP.@variable(ι, γ[1:J] >= 0.)
        JuMP.@variable(ι, l[1:K, 1:J])
        JuMP.@variable(ι, m[1:K, 1:J])
        JuMP.@variable(ι, n[1:K, 1:J])
        JuMP.@constraint(ι, [k=1:K, j=1:J], [-m[k,j], -l[k,j], exp(1) * n[k,j]] in JuMP.MOI.ExponentialCone()) # dual(EXP cone)
        JuMP.@variable(ι, w1[1:K, 1:N] >= 0.)                                       # dual( K(W) ) begins
        JuMP.@variable(ι, w2[1:K, 1:N] >= 0.)
        JuMP.@variable(ι, r[1:K, 1:N])
        JuMP.@variable(ι, t[1:K])
        JuMP.@constraint(ι, [k=1:K, n=1:N], w1[k, n] - w2[k, n] + r[k, n] == 0.)
        JuMP.@constraint(ι, [k=1:K], t[k] >= sum(w1[k, :]) + sum(w2[k, :]))         # dual( K(W) ) ends
        k = 1
        JuMP.@constraint(ι, α - b1(formal_x) + sum(l[k, j] * ip(qs[:, j]/2, qs[:, j]) for j in 1:J) - sum(m[k, :]) - t[k] >= 0.)
        JuMP.@constraint(ι, [n=1:N], β[n] - a1(formal_x)[n] - r[k, n] - sum(l[k, j] * qs[n, j] for j in 1:J) == 0.)
        JuMP.@constraint(ι, [j=1:J], γ[j] - n[k, j] == 0.)
        k = 2
        JuMP.@constraint(ι, α + sum(l[k, j] * ip(qs[:, j]/2, qs[:, j]) for j in 1:J) - sum(m[k, :]) - t[k] >= 0.)
        JuMP.@constraint(ι, [n=1:N], β[n] - r[k, n] - sum(l[k, j] * qs[n, j] for j in 1:J) == 0.)
        JuMP.@constraint(ι, [j=1:J], γ[j] - n[k, j] == 0.)
    end
    JuMP.@objective(ι, Min, θ + (α + sum(γ))/ε)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    ub = JuMP.objective_value(ι)
    formal_x = JuMP.value.(formal_x)
    return ub, formal_x
end
function ECP_sub_inf_program(formal_x, qs) # 🫠 used to derive η and ξ
    # function ECP_sub_sup_program(formal_x, qs) # we don't need this function anymore
    #     J = size(qs)[2]
    #     m = JumpModel(1) # a conic optimization
    #     JuMP.@variable(m, η[1:K] >= 0.)
    #     JuMP.@constraint(m, sum(η) == 1.)
    #     JuMP.@variable(m, ξ[1:N, 1:K])
    #     JuMP.@constraint(m, [n = 1:N], sum(ξ[n, :]) == 0.)
    #     JuMP.@variable(m, ζ[1:J, 1:K])
    #     JuMP.@constraint(m, [j = 1:J], sum(ζ[j, :]) <= 1.)
    #     JuMP.@constraint(m, [n = 1:N, k = 1:K], -η[k] <= ξ[n, k])
    #     JuMP.@constraint(m, [n = 1:N, k = 1:K], ξ[n, k] <= η[k])
    #     JuMP.@constraint(m, [j = 1:J, k = 1:K], [qs[:, j]' * ξ[:, k] - η[k] * .5 * (qs[:, j]' * qs[:, j]), η[k], ζ[j, k]] in JuMP.MOI.ExponentialCone())
    #     JuMP.@objective(m, Max, a1(formal_x)' * ξ[:, 1] + b1(formal_x) * η[1])
    #     JuMP.optimize!(m)
    #     status = JuMP.termination_status(m)
    #     if status == JuMP.OPTIMAL
    #         return JuMP.objective_value(m), JuMP.value.(η), JuMP.value.(ξ)
    #     elseif status == JuMP.SLOW_PROGRESS
    #         @warn "♣ ECP terminates with JuMP.SLOW_PROGRESS" ξ=JuMP.value.(ξ) objval=JuMP.objective_value(m) η=JuMP.value.(η) 
    #         return JuMP.objective_value(m), JuMP.value.(η), JuMP.value.(ξ)
    #     else
    #         error("the ♣ ECP terminate with $(status)")
    #     end
    # end  
    J = size(qs)[2]
    ι = JumpModel(1)
    if true # sub block of constrs and variables
        JuMP.@variable(ι, α)
        JuMP.@variable(ι, β[1:N])
        JuMP.@variable(ι, γ[1:J] >= 0.)
        JuMP.@variable(ι, l[1:K, 1:J])
        JuMP.@variable(ι, m[1:K, 1:J])
        JuMP.@variable(ι, n[1:K, 1:J])
        JuMP.@constraint(ι, [k=1:K, j=1:J], [-m[k,j], -l[k,j], exp(1) * n[k,j]] in JuMP.MOI.ExponentialCone()) # dual(EXP cone)
        JuMP.@variable(ι, w1[1:K, 1:N] >= 0.)                                       # dual( K(W) ) begins
        JuMP.@variable(ι, w2[1:K, 1:N] >= 0.)
        JuMP.@variable(ι, r[1:K, 1:N])
        JuMP.@variable(ι, t[1:K])
        JuMP.@constraint(ι, [k=1:K, n=1:N], w1[k, n] - w2[k, n] + r[k, n] == 0.)
        JuMP.@constraint(ι, [k=1:K], t[k] >= sum(w1[k, :]) + sum(w2[k, :]))         # dual( K(W) ) ends
        k = 1
        JuMP.@constraint(ι, eta1, α - b1(formal_x) + sum(l[k, j] * ip(qs[:, j]/2, qs[:, j]) for j in 1:J) - sum(m[k, :]) - t[k] >= 0.)
        JuMP.@constraint(ι, xi1[n=1:N], β[n] - a1(formal_x)[n] - r[k, n] - sum(l[k, j] * qs[n, j] for j in 1:J) == 0.)
        JuMP.@constraint(ι, zeta1[j=1:J], γ[j] - n[k, j] == 0.)
        k = 2
        JuMP.@constraint(ι, eta2, α + sum(l[k, j] * ip(qs[:, j]/2, qs[:, j]) for j in 1:J) - sum(m[k, :]) - t[k] >= 0.)
        JuMP.@constraint(ι, xi2[n=1:N], β[n] - r[k, n] - sum(l[k, j] * qs[n, j] for j in 1:J) == 0.)
        JuMP.@constraint(ι, zeta2[j=1:J], γ[j] - n[k, j] == 0.)
    end
    JuMP.@objective(ι, Min, α + sum(γ))
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    η = JuMP.dual.([eta1, eta2])
    ξ = JuMP.dual.([xi1 xi2])
    return JuMP.objective_value(ι), η, ξ
end

function egrand(q, ξ, k) ip(q, ξ[:, k]) end
function lgrand(q, ξ, η) sum(η[i] * exp( egrand(q, ξ, i) ) for i in 1:K) end
function objfun(q, ξ, η) ip(q/DEN, q) - log(lgrand(q, ξ, η)) end

Random.seed!(86)

for GIP_ite in 1:typemax(Int)
    ub, formal_x = oneshot_inf_program(qs)
    x, theta = formal_x[1:end-1], formal_x[end]
    x = round.(x; digits = 4)
    theta = round(theta; digits = 6)
    # @info ">>> GIP_ite = $(GIP_ite), ub = $ub, x = $x, θ = $theta"
    @info ">>> GIP_ite = $(GIP_ite), ub = $ub, θ = $theta, and x is shown as"
    @info x
    x_incumbent .= formal_x
    _, η, ξ = ECP_sub_inf_program(formal_x, qs)
    for k in eachindex(η)
        ξ[:, k] .= ξ[:, k] ./ η[k]
    end
    gen_res_at = q0 -> Optim.optimize(q -> objfun(q, ξ, η), q0, Optim.NewtonTrustRegion(); autodiff = :forward) # 
    find_already = [false]
    vio_q = NaN * ones(N)
    d = Distributions.Uniform(-5., 5.)
    for ite in 1:100
        q0 = rand(d, N)
        @debug "before opt, val = $(objfun(q0, ξ, η))"
        res = gen_res_at(q0)
        qt, vt = Optim.minimizer(res), Optim.minimum(res) # get the optimization result
        if vt < -1e-6
            @info "find the q with vio val = $vt"
            vio_q .= qt
            find_already[1] = true
            break
        else
            @debug "this trial fails with value $vt, fails at" qt
        end
    end
    if find_already[1] == false
        @info "we can't find a vio_q at this stage, thus leave the GIP algorithm"
        return x_incumbent
    else
        qs = [qs vio_q]
    end
end

# function cut_at(formal_x, qs)
#     Q_value, ξ, η = ECP_sup_subprogram(formal_x, qs)
#     ξ, η = ξ[:, 1], η[1]
#     px::Vector{Float64} = -[ξ[n] * sigma[n] + η * mu[n] for n in 1:N] 
#     pth::Float64 = -η
#     p0::Float64 = 0.
#     Q_value, [px; pth], p0
# end
# function outer_surrogate(cutDict)
#     m = JumpModel(0) # cutting plane optimization
#     JuMP.@variable(m, x[1:N] >= 0.)
#     JuMP.@constraint(m, sum(x) == 1.)
#     JuMP.@variable(m, θ)
#     JuMP.@variable(m, th)
#     for (px, p0) in zip(cutDict["px"], cutDict["p0"])
#         JuMP.@constraint(m, th >= px' * [x; θ] + p0)
#     end
#     JuMP.@variable(m, obj)
#     JuMP.@constraint(m, obj >= θ + th / ε)
#     JuMP.@objective(m, Min, obj)
#     JuMP.optimize!(m)
#     status, cutDict_sufficient_flag = JuMP.termination_status(m), true
#     if status in [JuMP.INFEASIBLE_OR_UNBOUNDED, JuMP.DUAL_INFEASIBLE]
#         cutDict_sufficient_flag = false
#         JuMP.set_lower_bound(obj, CTPLN_INI_BND)
#         JuMP.optimize!(m)
#         status = JuMP.termination_status(m)
#     end
#     if status != JuMP.OPTIMAL
#         error(" flag=($cutDict_sufficient_flag), outer_surrogate: status = $status ")
#     end
#     if cutDict_sufficient_flag
#         return JuMP.value(obj), JuMP.value.([x; θ])
#     else
#         return -Inf, JuMP.value.([x; θ])
#     end
# end
# function outer_solve(qs)::Vector{Float64}
#     xt = initial_trial_x_generation()
#     _, px, p0 = cut_at(xt, qs) # add an initial cut to assure lower boundedness
#     cutDict = Dict( # initialize cutDict
#         "id" => Int[1],
#         "at" => Vector{Float64}[xt],
#         "px" => Vector{Float64}[px],
#         "p0" => Float64[p0]
#     )
#     for ite in 1:typemax(Int)
#         lb, xt = outer_surrogate(cutDict)
#         θ = xt[end] # record current 1st stage solution
#         Q_value, px, p0 = cut_at(xt, qs)
#         ub = θ + Q_value / ε
#         @assert lb <= ub + CTPLN_GAP_TOL "lb($(lb)) > ub($(ub)), check whether they are specified correctly!"
#         gap = ub-lb
#         @debug "▶ ▶ ▶ ite = $ite" lb ub gap
#         if gap < CTPLN_GAP_TOL
#             @debug " 😊 outer problem ub - lb = $gap < $CTPLN_GAP_TOL, ub = $(ub)"
#             xtmp = round.(xt[1:end-1]; digits = 4)
#             @info "x at $xtmp, θ at $θ, inducing cost $ub"
#             return xt
#         end
#         push!(cutDict["id"], 1+length(cutDict["id"]))
#         push!(cutDict["at"], xt)
#         push!(cutDict["px"], px)
#         push!(cutDict["p0"], p0)
#     end
# end
# function worst_pd(x, qs) # after the accomplishment of the cutting plane method
#     _, ξ, η = ECP_sup_subprogram(x, qs)
#     @debug "worst_pd: the probability is $η"
#     # bitVec = η .> 7e-5 # omit those events with negligible Pr
#     # ξ, η = ξ[:, bitVec], η[bitVec]
#     Dict(
#         "z" => [ξ[:, i] / η[i] for i in eachindex(η)],
#         "P" => η
#     )
# end
# function expobjfun(q, ξ, η) 1 - sum( η[k] * exp(ip(q, ξ[:, k]) - ip(q, q)/2) for k in 1:K ) end
# function cmlgrand(q, ξ, η, m) sum(ξ[m, i] * η[i] * exp( egrand(q, ξ, i) ) for i in 1:K) end
# function cnmlgrand(q, ξ, η, n, m) sum(ξ[n, i] * ξ[m, i] * η[i] * exp( egrand(q, ξ, i) ) for i in 1:K) end





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







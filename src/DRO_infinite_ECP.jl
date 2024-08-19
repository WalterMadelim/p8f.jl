import LinearAlgebra
import Random
import JuMP
import Gurobi
import MosekTools
using Logging
# import Distributions

# DRO_inf_ECP
# temporary version
# but shows how the Gurobi callback works and how the ECP uses
# the style is useful
# 19/8/24

column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function JumpModel(i)
    if i == 0 # the most useful
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    elseif i == 1 # generic convex conic program
        m = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 # if you need Gurobi callback
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    # JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end

function club_suit(formal_x, qs) # 🫠
    J = size(qs)[2]
    m = JumpModel(1) # a conic optimization
    JuMP.set_silent(m)
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
    # print("\r ♣ ECP starts >")
    # @warn "trial x for the ECP is" x = formal_x[begin:end-1] theta = formal_x[end]
    JuMP.optimize!(m)
    # print("\r ♣ ECP ends successfully <")
    if JuMP.termination_status(m) == JuMP.OPTIMAL
        return JuMP.objective_value(m), JuMP.value.(ξ), JuMP.value.(η)
    elseif JuMP.termination_status(m) == JuMP.SLOW_PROGRESS
        @warn "♣ ECP terminates with JuMP.SLOW_PROGRESS" objval = JuMP.objective_value(m) η = JuMP.value.(η)
        @info "ξ in detail" ξ = JuMP.value.(ξ)
        return JuMP.objective_value(m), JuMP.value.(ξ), JuMP.value.(η)
    else
        error("the ♣ ECP terminate with $(JuMP.termination_status(m))")
    end
end

function cut_at(formal_x, qs) # 🫠
    th_ub, ξ, η = club_suit(formal_x, qs)
    ξ, η = ξ[:, 1], η[1]
    px::Vector{Float64} = -[ξ[n] * sigma[n] + η * mu[n] for n in 1:N]
    pth::Float64 = -η
    p0::Float64 = 0. #  ⚠️ we don't have const term in this instance
    th_ub, [px; pth], p0
end
function outer_surrogate(cutDict) # 🫠🫠
    m = JumpModel(0) # cutting plane optimization
    JuMP.set_silent(m)
    JuMP.@variable(m, θ)
    JuMP.@variable(m, x[1:N] >= 0.)
    JuMP.@constraint(m, sum(x) == 1.)
    JuMP.@variable(m, th)
    for (px, p0) in zip(cutDict["px"], cutDict["p0"])
        JuMP.@constraint(m, th >= px' * [x; θ] + p0)
    end
    JuMP.@variable(m, obj >= CTPLN_INI_BND)
    JuMP.@constraint(m, obj >= θ + th / ε)
    JuMP.@objective(m, Min, obj)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL " outer_surrogate_LP: $(JuMP.termination_status(m))"
    JuMP.value(obj), JuMP.value.([x; θ])
end
function outer_solve(qs)::Vector{Float64} # 🫠🫠
    xt = initial_trial_x_generation()
    th_ub, px, p0 = cut_at(xt, qs) # add an initial cut to assure lower boundedness
    cutDict = Dict( # initialize cutDict
        "id" => Int[1],
        "at" => Vector{Float64}[xt],
        "px" => Vector{Float64}[px],
        "p0" => Float64[p0]
    )
    for ite in 1:typemax(Int)
        lb, xt = outer_surrogate(cutDict)
        ite >= 2 && lb <= CTPLN_INI_BND + 1.1 && @error "lb is not leaving its BND yet"
        x, θ = xt[begin:end-1], xt[end] # record current 1st stage solution
        th_ub, px, p0 = cut_at(xt, qs)
        ub = θ + th_ub / ε
        @assert lb <= ub + 5e-5 "lb($(lb)) > ub($(ub)), check whether they are specified correctly!"
        gap = abs(ub-lb)
        @debug "▶ ▶ ▶ ite = $ite" lb ub gap
        if gap < 5e-5
            @assert ite >= 2 "gap closed at the very first iteration."
            @info " 😊 outer problem gap < 5e-5 " lb ub
            return xt
        end
        push!(cutDict["id"], 1+length(cutDict["id"]))
        push!(cutDict["at"], xt)
        push!(cutDict["px"], px)
        push!(cutDict["p0"], p0)
    end
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
function find_q(pd::Dict)
    function my_callback_function(cb_data, cb_where::Cint)
        if cb_where == Gurobi.GRB_CB_MIPSOL
            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            qSol = JuMP.callback_value.(cb_data, q)
            objSol = Ref{Cdouble}(NaN)
            errorcode = Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJ, objSol)
            @assert errorcode == 0 "OBJ: GRB_errorcode != 0"
            @assert !(objSol[] === NaN) "objSol still holds the element NaN after GRBcbget()"
            if objSol[] < -0.1 # a violating q is already found
                @info "Gurobi NLP obj val by early termination" objSol[]
                qt .= qSol # a solution q that violates is found
                Gurobi.GRBterminate(JuMP.backend(m)) 
                return nothing
            else # objSol here should be renamed as objBnd, but we omit this process
                errorcode = Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJBND, objSol)
                @assert errorcode == 0 "OBJBND: GRB_errorcode != 0"
                if objSol[] > -8e-5 # recognized as obj(q) >= 0, for all q
                    qt .= -Inf
                    Gurobi.GRBterminate(JuMP.backend(m))
                    return nothing
                else # we haven't find a violating q, nor have we recognize that obj(q) >= 0 for all q
                    return nothing
                end
            end
        else
            return nothing
        end
    end
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # direct with callback, might be non-convex QCQP
    JuMP.set_silent(m)
    JuMP.@variable(m, q[1:N])
    JuMP.@variable(m, logarith)
    JuMP.@objective(m, Min, .5 * q' * q - logarith)
    JuMP.@variable(m, expectatio)
    errcode = Gurobi.GRBaddgenconstrLog(JuMP.backend(m), "", column(expectatio), column(logarith), "")
    @assert errcode == 0 "GRB_errcode != 0"
    C = length(pd["P"]) # there are C points in the worst_PD
    JuMP.@variable(m, expresult[1:C])
    JuMP.@constraint(m, expectatio == sum(pd["P"][c] * expresult[c] for c in 1:C)) # we specify the `expectatio`
    JuMP.@variable(m, exponen[1:C])
    for c in 1:C
        Gurobi.GRBaddgenconstrExp(JuMP.backend(m), "", column(exponen[c]), column(expresult[c]), "")
    end
    JuMP.@constraint(m, [c = 1:C], exponen[c] == q' * pd["z"][c])
    qt = Inf * ones(N) # initial is Inf, if objBnd >= 0, set it to -Inf then, elseif a vio q is found, set it to the q.
    JuMP.MOI.set(m, Gurobi.CallbackFunction(), my_callback_function)
    # print("\r NLP starts >")
    JuMP.optimize!(m)
    # print("\r NLP ends successfully <")
    if JuMP.termination_status(m) == JuMP.INTERRUPTED
        @assert qt[1] != Inf "qt has not been modified by callback"
        return qt
    elseif JuMP.termination_status(m) ==JuMP.OPTIMAL
        if JuMP.objective_value(m) < -8e-5
            qt .= JuMP.value.(q)
        else
            qt .= -Inf
        end
        return qt
    else
        error(" hell : $(JuMP.termination_status(m))")
    end
end

if true # problem data gen
    function initial_trial_x_generation() # x in 𝒳
        m = JumpModel(0)
        JuMP.set_silent(m)
        JuMP.@variable(m, θ)
        JuMP.@variable(m, x[1:N] >= 0.)
        JuMP.@constraint(m, sum(x) == 1.)
        JuMP.optimize!(m) # without an obj
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL " initial_trial_x_gen: $(JuMP.termination_status(m))"
        JuMP.value.([x; θ])
    end
    function a1(formal_x) -sigma .* formal_x[begin:end-1] end
    function b1(formal_x) -[mu; 1]' * formal_x end
    CTPLN_INI_BND = -15. # ⚠️ make sure that after optimization, the obj is distant from this bound
    K = 2
    N = 8
    # obj f data gen
    mu = [n/4. for n in 1:N]
    sigma = [N * sqrt(2 * n) for n in 1:N]/3
    ε = .05
    if true # global containers
        qs = one(ones(N, N))
        x_incumbent = initial_trial_x_generation() # the global xt (the incumbent)
    end
end

function main()
    for ite in 1:typemax(Int)
        global qs, x_incumbent
        xt = outer_solve(qs)
        x_incumbent .= xt # record the incumbent
        pd = worst_pd(xt, qs)
        qt = find_q(pd)
        if qt[1] == -Inf
            @info " 😊 the current qs::Matrix is already sufficient, thus GIP algorithm returns the best solution. "
            return xt
        else
            qs = [qs qt]
            @info " ▶ ▶ ▶ Main ite = $(ite), num of q = $(size(qs)[2])"
            @info "new added qt" qt
        end
    end
end

xt = main()




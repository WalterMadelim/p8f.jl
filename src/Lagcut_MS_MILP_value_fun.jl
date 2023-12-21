using CairoMakie
using OffsetArrays
using Logging
import JuMP
import Gurobi
import LinearAlgebra
import Distributions

function udrv() # uniformly distributed rv
    rand(Distributions.Uniform(-5,7.3))
end
function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
function get_solu_chain(train_trials_record)
    [train_trials_record[i]["x_dm1_trial"][end] for i in eachindex(train_trials_record)]
end
function get_x_d_decision_bound(boundDict, d::Int, t::Int, upper::Bool)
    d == t-2 && error("x[d] is given by parameters")
    @assert d in t-1:T # from the first decision to the last decision
    m = JumpModel()
    JuMP.@variable(m, x[t-2:d]) # the beginning one is input, thus a fixed value
    for i in t-2:d-1 # when i == t-2, this bound should be a fixed value = the global parameter x_input_tm2
        JuMP.set_lower_bound(x[i], boundDict["xl"][i])
        JuMP.set_upper_bound(x[i], boundDict["xu"][i])
    end
    JuMP.@variable(m, cb[i = t-1:d], Bin) # the formal decision range
    JuMP.@constraint(m, [i = t-1:d], x[i] == x[i-1] + (2. * cb[i] - 1.) + xi[i])
    if upper
        JuMP.@objective(m, Max, x[d]) # the decision variable whose bound is decided in this function
    else
        JuMP.@objective(m, Min, x[d])
    end
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(x[d])
end
function _Q_kernel(d, arg_x_dm1, fix_decision_x_d::Bool, fixed_x_d_value)
    m = JumpModel()
    JuMP.@variable(m, boundDict["xl"][i] <= x[i = d-1:T] <= boundDict["xu"][i])
    JuMP.fix(x[d-1], arg_x_dm1; force = true)
    fix_decision_x_d && JuMP.fix(x[d], fixed_x_d_value; force = true)
    JuMP.@variable(m, cb[i = d:T], Bin)
    JuMP.@variable(m, boundDict["fl"][i] <= f_t[i = d:T] <= boundDict["fu"][i])
    JuMP.@variable(m, abs_x[i = d:T] >= 0.) # can only used in Min-obj
    JuMP.@constraint(m, [i = d:T], abs_x[i] >= x[i])
    JuMP.@constraint(m, [i = d:T], abs_x[i] >= -x[i])
    JuMP.@constraint(m, [i = d:T], x[i] == x[i-1] + (2. * cb[i] - 1.) + xi[i])
    JuMP.@constraint(m, [i = d:T], f_t[i] == beta^(i-1) * abs_x[i])
    JuMP.@objective(m, Min, sum(f_t[i] for i in d:T))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    if d == t-1
        for i in d:T
            println("$(JuMP.value(x[i]))")
        end
    end
    return JuMP.objective_value(m)
end
function Q(d, arg_x_dm1)
    d == T+1 && (println("Evaluating the dummy Q_{T+1}"); return 0.)
    @assert d in t-1:T # the first index is valid for Q but not for Q_hat
    @assert boundDict["xl"][d-1] <= arg_x_dm1 <= boundDict["xu"][d-1]
    return _Q_kernel(d, arg_x_dm1, false, NaN)
end
function f_d(d, x)
    @assert d in t-1:T
    return beta^(d-1) * abs(x)
end
function Q_hat(d, trial_xdm1)
    d == T+1 && (println("Evaluating the dummy Q_hat_{T+1}"); return 0.)
    d == t-1 && error("we did NOT establish the model Q_{$d}, because x_{$(d-1)} is fixed.")
    @assert d in t:T
    @assert boundDict["xl"][d-1] <= trial_xdm1 <= boundDict["xu"][d-1] 
    m = JumpModel()
    JuMP.@variable(m, boundDict["hatQl"][d] <= tha <= boundDict["hatQu"][d])
    lD = hatQ[d]
    for (cx, ct, rhs, old) in zip(lD["cx"], lD["ct"], lD["rhs"], lD["is_inferior"])
        old || JuMP.@constraint(m, cx * trial_xdm1 + ct * tha >= rhs)
    end
    JuMP.@objective(m, Min, tha)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(tha)
end
function idq(d, x_input_dm2) # √ √ √ step function in fwd. 
    # `idq` is the acronym for ★ input-decision-trailing_Q ★ a paradigm for MSP
    @assert boundDict["xl"][d-2] <= x_input_dm2 <= boundDict["xu"][d-2] # ★ d-2
    m = JumpModel()
    JuMP.@variable(m, boundDict["xl"][d-1] <= trial_xdm1 <= boundDict["xu"][d-1]) # ★ d-1
    JuMP.@variable(m, boundDict["hatQl"][d] <= tha <= boundDict["hatQu"][d]) # ★ d
    JuMP.@variable(m, boundDict["fl"][d-1] <= f_dm1 <= boundDict["fu"][d-1])
    JuMP.@variable(m, a >= 0.)
    JuMP.@variable(m, cb_dm1, Bin)
    JuMP.@constraint(m, a >=  trial_xdm1)
    JuMP.@constraint(m, a >= -trial_xdm1)
    JuMP.@constraint(m, f_dm1 == beta^(d-2) * a)
    JuMP.@constraint(m, trial_xdm1 == x_input_dm2 + (2. * cb_dm1 - 1.) + xi[d-1])
    lD = hatQ[d]
    for (cx, ct, rhs, old) in zip(lD["cx"], lD["ct"], lD["rhs"], lD["is_inferior"])
        old || JuMP.@constraint(m, cx * trial_xdm1 + ct * tha >= rhs)
    end
    JuMP.@objective(m, Min, f_dm1 + tha)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(trial_xdm1), JuMP.value(f_dm1), JuMP.value(tha), JuMP.objective_value(m) # those in train_trials_record
end
if true #【Functions】algorithm1 and its affiliates, t in 2:T
    function Q_ast(d, pai, pai0) # see (12)
        m = JumpModel()
        JuMP.@variable(m, boundDict["xl"][d-1] <= x_tm1 <= boundDict["xu"][d-1])
        JuMP.@variable(m, boundDict["xl"][d] <= x <= boundDict["xu"][d])
        JuMP.@variable(m, boundDict["fl"][d] <= f_t <= boundDict["fu"][d])
        JuMP.@variable(m, abs_x >= 0.) # can only used in Min-obj
        JuMP.@variable(m, cb, Bin)
        JuMP.@variable(m, boundDict["hatQl"][d+1] <= tha <= boundDict["hatQu"][d+1]) # cutting plane model of Q_{d+1}(x[d])
        JuMP.@constraint(m, abs_x >= x)
        JuMP.@constraint(m, abs_x >= -x)
        JuMP.@constraint(m, x == x_tm1 + (2. * cb - 1.) + xi[d])
        lD = hatQ[d+1] # ■ hatQ_{t+1} is involved in the objective function of the def of hatQ_{t}. THIS is where MSP is more involved than 2SP
        for (cx, ct, rhs, old) in zip(lD["cx"], lD["ct"], lD["rhs"], lD["is_inferior"]) # this is the cutting plane model of Q_t(x[d-1])
            old || JuMP.@constraint(m, cx * x + ct * tha >= rhs)
        end
        JuMP.@constraint(m, f_t == beta^(d-1) * abs_x)
        coef_pai0 = f_t + tha
        JuMP.@objective(m, Min, pai * x_tm1 + pai0 * coef_pai0)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        JuMP.objective_value(m), JuMP.value(x_tm1), JuMP.value(coef_pai0) # [1]: value, [2][3]: coeffs of pai and pai0
    end
    function algo1_ini(d, x_hat, tha_hat)
        m = JumpModel()
        JuMP.@variable(m, pai) # bound implicitly enforced later
        JuMP.@variable(m, pai0 >= 0.)
        # |pai| + pai0 <= 1 
        JuMP.@constraint(m, 1. - pai0 >= pai)
        JuMP.@constraint(m, 1. - pai0 >= -pai)
        d != T && ( hatQast[d] = Dict("cp" => Float64[],"cp0" => Float64[]) ) # initialization
        if isempty(hatQast[d]["cp"]) # build Q̂* from scratch; this `if` is used to ensure that at least 1 cut exists for Q̂*
            JuMP.@objective(m, Max, - x_hat * pai - tha_hat * pai0) 
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL
            pai_ini, pai0_ini = JuMP.value(pai), JuMP.value(pai0) # get the initial feas. solu of (17): 1.0, 0.0
            @assert pai0_ini >= 0. # otherwise it's Gurobi's fault
            val, c_pai, c_pai0 = Q_ast(d, pai_ini, pai0_ini)
            push!(hatQast[d]["cp"], c_pai)
            push!(hatQast[d]["cp0"], c_pai0)
        end
        JuMP.@variable(m, phi)
        JuMP.@objective(m, Max, phi - x_hat * pai - tha_hat * pai0) # the surrogate objective of (17)
        lD = hatQast[d]
        for (cp, cp0) in zip(lD["cp"], lD["cp0"])
            JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0)
        end
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        ub = JuMP.objective_value(m)
        pai, pai0 = JuMP.value(pai), JuMP.value(pai0) # a feas. solu of (17)
        @assert pai0 >= 0. # otherwise it's Gurobi's fault
        val, c_pai, c_pai0 = Q_ast(d, pai, pai0) # this is a new one compared to the initializing above
        lb = val - pai * x_hat - pai0 * tha_hat # obj value corr to the feas. solu just derived
        return ub, lb, pai, pai0, val, c_pai, c_pai0
    end
    function algorithm1(ite, d, x_hat, tha_hat, delta = 0.9) # generate a (13) cut for Q_t(⋅) that cut off trial (x̂t-1, curr_hat_Q_t(x̂t-1))
        icbt = zeros(4) # [1]=lb  [2]=pai  [3]=pai0 [4]=Q^ast_t(pai, pai0)  All incumbent 
        ub, icbt[1], icbt[2], icbt[3], icbt[4], c_pai, c_pai0 = algo1_ini(d, x_hat, tha_hat) # (3.8263752000000006, 0.0, 0.0, 1.0, 1.4450970583895657, 0.0)
        while true
            if ub <= 1e-6
                # @info "This trail point cannot be cut off with algorithm1 (is saturated)." d ite x_hat tha_hat ub
                return false
            elseif icbt[1] > (1.0 - delta) * ub + 1e-6 # cut off: by algo design
                if icbt[3] < 0.
                    error("Algo1: very serious wrong!")
                elseif icbt[3] == 0.
                    error("Algo1: intend to generate feas. cut, which is abnormal!")
                else # give birth to a normal opt. cut
                    cut_off_dist = icbt[4] - (icbt[2] * x_hat + icbt[3] * tha_hat)
                    @assert cut_off_dist > 1e-6 # cut off: the authentic certificate 
                    if isempty(hatQ[d]["cut_id"])
                        push!(hatQ[d]["cut_id"], 1)
                    else
                        push!(hatQ[d]["cut_id"], 1 + maximum(hatQ[d]["cut_id"]))
                    end
                    push!(hatQ[d]["gen_in_ite"], ite)
                    push!(hatQ[d]["x_trial"], x_hat)
                    push!(hatQ[d]["t_trial"], tha_hat)
                    push!(hatQ[d]["cx"], icbt[2])
                    push!(hatQ[d]["ct"], icbt[3])
                    push!(hatQ[d]["rhs"], icbt[4])
                    push!(hatQ[d]["is_inferior"], false)
                    @debug "algo1 SUCCESS: cut off!" cut_off_dist strengthed_Qt_ind=d icbt
                    return true
                end
            else
                push!(hatQast[d]["cp"], c_pai)
                push!(hatQast[d]["cp0"], c_pai0)
                m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
                JuMP.set_silent(m)
                JuMP.@variable(m, pai)
                JuMP.@variable(m, pai0 >= 0.)
                JuMP.@constraint(m, 1. - pai0 >=  pai) # |pai| + pai0 <= 1 
                JuMP.@constraint(m, 1. - pai0 >= -pai) # |pai| + pai0 <= 1 
                JuMP.@variable(m, phi)
                JuMP.@objective(m, Max, phi - x_hat * pai - tha_hat * pai0) # the surrogate objective of (17)
                lD = hatQast[d]
                for (cp, cp0) in zip(lD["cp"], lD["cp0"])
                    JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0)
                end
                JuMP.optimize!(m)
                @assert JuMP.termination_status(m) == JuMP.OPTIMAL
                ub = JuMP.objective_value(m)
                pai, pai0 = JuMP.value(pai), JuMP.value(pai0) # a feas. solu of (17)
                @assert pai0 >= 0. # otherwise it's Gurobi's fault
                val, c_pai, c_pai0 = Q_ast(d, pai, pai0) # this is a new one compared to the initializing above
                lb = val - pai * x_hat - pai0 * tha_hat
                if lb > icbt[1] # the new feasible point (pai, pai0) is superior
                    icbt .= lb, pai, pai0, val
                end
            end
        end
    end
end
function getData(num_decisions, input_fixed_parameter)
    ind_first_decision = T + 1 - num_decisions
    t = ind_first_decision + 1
    x_input_tm2 = input_fixed_parameter
    boundDict = Dict(
        "x_Ref" => [i for i in t-2:T],
        "xl" => OffsetVector([0. for _ in t-2:T], t-2:T), # lb of decision x's EXCEPT for the beginning one is fixed
        "xu" => OffsetVector([0. for _ in t-2:T], t-2:T), # ub's, the beginning one is dummy
        # t-1:T is the index range of the formal decision variables
        "f_Ref" => [i for i in t-1:T],
        "fl" => OffsetVector([0. for _ in t-1:T], t-1:T), # real zeros
        "fu" => OffsetVector([0. for _ in t-1:T], t-1:T), 
        "Q_Ref" => [i for i in t:T+1],
        # t:T is the index range of the formal hatQ's
        "hatQl"=> OffsetVector([0. for _ in t:T+1], t:T+1), # real zeros; the end is fixed to 0.
        "hatQu"=> OffsetVector([0. for _ in t:T+1], t:T+1) # the end is fixed to 0.
    )
    boundDict["xl"][t-2], boundDict["xu"][t-2] = x_input_tm2, x_input_tm2 # the fixed value
    for d in t-1:T # must step forward sequentially
        boundDict["xl"][d] = get_x_d_decision_bound(boundDict, d, t, false)
        boundDict["xu"][d] = get_x_d_decision_bound(boundDict, d, t, true)
    end
    boundDict["fu"][t-1:T] .= Float64[beta^(d-1) * max(abs(boundDict["xl"][d]), abs(boundDict["xu"][d])) for d in t-1:T]
    boundDict["hatQu"][t:T] .= Float64[sum(boundDict["fu"][d] for d in i:T) for i in t:T]
    hatQ = OffsetVector([Dict(
        "cut_id" => Int[],
        "gen_in_ite" => Int[],
        "x_trial" => Float64[],
        "t_trial" => Float64[],
        "cx" => Float64[],
        "ct" => Float64[],
        "rhs" => Float64[],
        "is_inferior" => Bool[] # the cut is deemed inferior compared with a later-generated one
        ) for _ in t:T+1], t:T+1)
    hatQast = OffsetVector([Dict("cp" => Float64[],"cp0" => Float64[]) for _ in t:T], t:T)
    train_trials_record = OffsetVector([Dict(
        "ite" => Int[], # linear currently
        "x_dm1_trial" => Float64[],
        "f_dm1_induced" => Float64[],
        "tha_trial" => Float64[],
        "lb" => Float64[], # "f_tm1_indecud" + "tha_trial"
    ) for _ in t:T+1], t:T+1)
    return t, boundDict, hatQ, hatQast, train_trials_record
end
function train(u, x_input_um2)
    ite, ub = 1, boundDict["fu"][u-1] + boundDict["hatQu"][u] # initial bound of Q_t for a reference only
    if ite == 1 # √ √ √ initialization
        x_dm1_trial = x_input_um2 # to ignite
        for d in u:T+1
            x_dm1_trial, f_dm1_induced, tha_trial, lb = idq(d, x_dm1_trial) # ignite
            if d == u && lb >= ub - 1e-6
                @warn "😅 Gap closed at the First iterate!" ite ub lb solution=x_dm1_trial
                return 1, x_dm1_trial, ub, NaN
            end
            push!(train_trials_record[d]["ite"], ite)
            push!(train_trials_record[d]["x_dm1_trial"], x_dm1_trial)
            push!(train_trials_record[d]["f_dm1_induced"], f_dm1_induced)
            push!(train_trials_record[d]["tha_trial"], tha_trial)
            push!(train_trials_record[d]["lb"], lb)
        end
        ub = sum(train_trials_record[d]["f_dm1_induced"][ite] for d in u:T+1)
        for d in T:-1:u
            x_dm1_trial = train_trials_record[d]["x_dm1_trial"][ite]
            algorithm1(ite, d, x_dm1_trial, Q_hat(d, x_dm1_trial)) || @warn "At ite = 1, bwd phase, there is at least one Q_hat not updated."
        end
    end
    x_close_TOL, value_TOL = 1e-5, 1e-6
    for ite in 2:typemax(Int) # Main Iterate
        x_dm1_trial = x_input_um2 # to ignate
        for d in u:T+1
            x_dm1_trial, f_dm1_induced, tha_trial, lb = idq(d, x_dm1_trial) # ignite
            if d == u # summary / exit session (for the last ite)
                if lb >= ub - 1e-6
                    @info "😊 Gap closed." ite ub lb solution=x_dm1_trial
                    return 2, x_dm1_trial, ub, NaN
                elseif !any([ite-1 in hatQ[d]["gen_in_ite"] for d in u:T]) # the last ite doesn'u generate a cut for any Q_t
                    @warn "Quit training process due to saturation of cutting planes" ite ub lb gap="$((ub-lb)/ub*100)%" candidate_solution=x_dm1_trial
                    return 3, x_dm1_trial, ub, (ub-lb)/ub
                else
                    @info "▶ ▶ ▶ ite = $ite" ub lb
                end
            end
            push!(train_trials_record[d]["ite"], ite)
            push!(train_trials_record[d]["x_dm1_trial"], x_dm1_trial)
            push!(train_trials_record[d]["f_dm1_induced"], f_dm1_induced)
            push!(train_trials_record[d]["tha_trial"], tha_trial)
            push!(train_trials_record[d]["lb"], lb)
        end
        ub = sum(train_trials_record[d]["f_dm1_induced"][ite] for d in u:T+1) # update directly to save effort
        # @info " ♠ Start bwd pass"
        for d in T:-1:u
            x_dm1_trial = train_trials_record[d]["x_dm1_trial"][ite]
            t_trial = Q_hat(d, x_dm1_trial)
            cut_is_gened = algorithm1(ite, d, x_dm1_trial, t_trial) # use an ad hoc theta_trial
            if cut_is_gened
                bitvec2 = hatQ[d]["t_trial"] .< t_trial - value_TOL
                @assert bitvec2[end] == false # because @assert hatQ[d]["t_trial"][end] == t_trial
                bitvec = abs.( hatQ[d]["x_trial"] .- x_dm1_trial ) .< x_close_TOL
                hatQ[d]["is_inferior"][bitvec .&& bitvec2] .= true
            end
        end
    end
end

# we should give an `x0` for a formal MSP proposition
# but for generality we use the name `x_input_tm2`, if we want `x0`, we can just set `t = 2`
# x_input_tm2 = 2. # that induces a `v_prim`, which is the theoretical opt value of the MSP problem, it can be solved by a brute force one-large-scale deterministic programming 
# an MSP problem is fully decided by `x_input_tm2` and `xi`
# ■ v_prim = Q(t-1, x_input_tm2)
# T is the index of Q such that Q_T is the last formal value function, and Q_{T+1} is the ending dummy ≡ 0.
# 2 is the index of Q such that Q_2 is the first value function. We do not study Q_1 although it exists as a const Q_1(x_0) = `v_prim` where x_0 is `x_input_tm2`
# t is the index of Q such that 2 <= t, and it decides the DATA STRUCTURE we generate: x_{t-1} is the first formal decision that we store its bounds, we also store x_{t-2} as a dummy decision whose bounds are dummy and fixed to `x_input_tm2`
# u is the index of Q such that 2 <= t <= u, and it decides that x_{u-1} is the first formal decision that we decide in the TRAINING PROCESS train(u, x_input_um2)
# X_T is ALWAYS the last formal decision variable, both in data structure generation and trainning
# ■ t in 2:T or t == T+1
# ■ u in t:T or u == T+1
# example: t = 2, T = 3
#         Q1(fixed) =            Q2 =               Q3 =                Q4(dummy)
# x0(fix) ->        xi[1] x1            xi[2] x2            xi[3] x3



global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
beta = .9
xi = Float64[-4.059602000926438, 6.460695538605968, 3.4660403334855836, -4.019347551700105, 0.2318883638616276, 0.7330941144225198, -3.161166574781623, 1.5241577283751582, -3.583872523402287, 4.028527482440921, -2.2545896848589253, 6.328237145802639, 1.1858619742681293, 4.389687037306254, 0.1230529065657695, -3.0406688715249794, 6.42923699891918, -3.905337771397673, 1.339902685987914, 0.12369683624259231, 0.64663272549571, 3.2090725980247417, 2.728039391229948, -3.786741752921084]
T = length(xi)

check_vec = []

num_decisions = T # you choose before trainning
input_fixed_parameter = 2.0 # this is the base for bound tightening

t, boundDict, hatQ, hatQast, train_trials_record = getData(num_decisions, input_fixed_parameter)
flag, x_tm1_opt, ub, gap = train(t, input_fixed_parameter)
solu_chain = get_solu_chain(train_trials_record)
push!(check_vec, solu_chain) # store

while num_decisions >= 2
    num_decisions -= 1
    input_fixed_parameter = solu_chain[begin]

    t, boundDict, hatQ, hatQast, train_trials_record = getData(num_decisions, input_fixed_parameter)
    flag, x_tm1_opt, ub, gap = train(t, input_fixed_parameter)
    solu_chain = get_solu_chain(train_trials_record)
    println("Ite [$num_decisions]")
    println(solu_chain)
    push!(check_vec, solu_chain) # store

    tmp = popfirst!(check_vec) # recover old

    solu_chain, tmp = solu_chain[begin:end], tmp[begin+1:end]
    @assert isapprox(solu_chain, tmp; atol = 1e-4, norm = x -> LinearAlgebra.norm(x, Inf))
end


# ub 
# v_prim = Q(t-1, x_input_tm2)
# difference_between_ub____v_prim = ub - v_prim
# @info "concluding session" ub v_prim difference_between_ub____v_prim


if false #【Plots】
    function L_cut(lD::Dict, ind::Int)::Function  # distil the cut function 
        cx, ct, rhs = lD["cx"][ind], lD["ct"][ind], lD["rhs"][ind]
        @assert ct > 0.
        x -> (rhs - cx * x)/ct
    end    

    f = Figure();
    ax = Axis(f[1, 1]); # ,limits = (-4.3, 7.5, 0, 10));
    num_points = 300
    for d in t:t # draw the Q functions, through large scale programmings 
        xdm1 = range(boundDict["xl"][d-1], boundDict["xu"][d-1]; length = num_points);
        val = f_d.(d, xdm1) + Q.(d, xdm1);
        lines!(ax, xdm1, val)
        text!(ax, xdm1[num_points ÷ 2], val[num_points ÷ 2]; text = "Q$d")
        val = f_d.(d, xdm1) + Q_hat.(d, xdm1);
        lines!(ax, xdm1, val)
        text!(ax, xdm1[1], val[1]; text = "hatQ$d")
    end

    d = 18
    xdm1 = range(boundDict["xl"][s-1], boundDict["xu"][s-1]; length = num_points);
    for c in [3, 5, 6] # draw some cuts
        fc = L_cut(hatQ[s], c)
        val = fc.(xdm1)
        lines!(ax, xdm1, val)
        text!(ax, xdm1[1], val[1]; text = "cut$c")
    end
end

if false # helper functions
    function cut_viewer(s::Int)
        for (cx, ct, rhs, old) in zip(hatQ[s]["cx"], hatQ[s]["ct"], hatQ[s]["rhs"], hatQ[s]["is_inferior"])
            old || println(cx," ",ct," ",rhs)
        end
    end
end











[-0.8242400486573089, -3.4081125720595953, -0.3795850896186739, -3.634174774477599, 1.6940623713250398, 1.8799243455931691, 5.2696113828994235, 4.392664289465193, 0.3519954179402136, 5.781232416859393, 0.8758946454617202, 1.2157973314496342, 0.3394941676922265, -0.01387310681206344, 2.195199491212678, 3.923238882442626, -0.8635028704784578]
[-1.348397777032467, -0.8242400486573089, -5.408112572059595, -2.379585089618674, -5.634174774477599, -0.3059376286749602, -0.12007565440683088, 3.2696113828994235, 2.392664289465193, -1.6480045820597864, 3.7812324168593934, -1.1241053545382798, -0.7842026685503658, 0.3394941676922265, -0.01387310681206344, 2.195199491212678, 3.923238882442626, -0.8635028704784578]





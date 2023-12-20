using CairoMakie
using OffsetArrays
using Logging
import JuMP
import Gurobi
import LinearAlgebra
import Distributions

# use Lag cut (13) to generate (very tight, sometimes precise) cutting plane model for MS-MILP value functions, thus solve the MSP problem
# 20/12/23
# Design Std:
# 0, whenever we say time stage `s`, `s` indicates the subsript of a Q_hat, thus `s-1` is the index of the arg of Q_hat_s
# 1, at each ite, each stage, == 1 trial point (x, tha) must be generated and recorded, we allow the recurrence of x, but `tha` MUST weekly improve.
# 2, at each ite, each stage, <= 1 cut is generated
# 3, the crude MSP problem has x[0]:fixed_input, x[1], ..., x[T]; thus those Q_hat which needs study ranges from 2:T+1 with the ending one dummy.
# 4, the programming horizon is truncated to be [s in t:T+1], where t>=2. By T+1 we mean that x[T] is included, see tip 0.

function get_x_bound(t::Int, upper::Bool) # used for initialization, get the lb of state variables.
    @assert 1 <= t <= T # indices of x
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    JuMP.@variable(m, x[s = 0:t])
    for s in 0:t-1
        JuMP.set_lower_bound(x[s], boundDict["xl"][s])
        JuMP.set_upper_bound(x[s], boundDict["xu"][s])
    end
    JuMP.@variable(m, cb[s = 1:t], Bin)
    JuMP.@constraint(m, [s = 1:t], x[s] == x[s-1] + (2. * cb[s] - 1.) + xi[s])
    if upper
        JuMP.@objective(m, Max, x[t])
    else
        JuMP.@objective(m, Min, x[t])
    end
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(x[t])
end
function fill_boundDict(T, x0)
    boundDict["xl"][0], boundDict["xu"][0] = x0, x0 # the dummy input
    for s in 1:T # go forward
        boundDict["xl"][s] = get_x_bound(s, false)
        boundDict["xu"][s] = get_x_bound(s, true)
    end
    boundDict["fu"] .= [beta^(s-1) * max(abs(boundDict["xl"][s]), abs(boundDict["xu"][s])) for s in 1:T]
    boundDict["hatQu"][T] = boundDict["fu"][T]
    for s in T-1:-1:2 # go backward
        boundDict["hatQu"][s] = boundDict["fu"][s] + boundDict["hatQu"][s+1]
    end
end


if true #【Functions】Q related, i.e., use Gurobi's brute force to solve large-scale MILP
    function Q_kernel(t::Int, eval_mode::Bool, input_xm1::Float64)
        @assert 1 <= t <= T
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, boundDict["xl"][s] <= x[s = t-1:T] <= boundDict["xu"][s])
        if eval_mode
            @assert boundDict["xl"][t-1] <= input_xm1 <= boundDict["xu"][t-1]
            JuMP.fix(x[t-1], input_xm1; force = true)
        end
        JuMP.@variable(m, cb[t:T], Bin)
        JuMP.@variable(m, boundDict["fl"][s] <= f_t[s = t:T] <= boundDict["fu"][s])
        JuMP.@variable(m, abs_x[t:T] >= 0.) # can only used in Min-obj
        JuMP.@constraint(m, [s = t:T], abs_x[s] >= x[s])
        JuMP.@constraint(m, [s = t:T], abs_x[s] >= -x[s])
        JuMP.@constraint(m, [s = t:T], x[s] == x[s-1] + (2. * cb[s] - 1.) + xi[s])
        JuMP.@constraint(m, [s = t:T], f_t[s] == beta^(s-1) * abs_x[s])
        JuMP.@objective(m, Min, sum(f_t[s] for s in t:T))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        # @info "ctrl begin with x0 = $x0"
        # for t in 1:T
        #     println("x[$t] = $(JuMP.value(x[t])) = $(JuMP.value(x[t-1])) + $(JuMP.value(c[t])) + $(xi[t])" * ", with cost $(beta^(t-1) * JuMP.value(abs_x[t]))")
        # end
        # @info "total cost: $(JuMP.objective_value(m))"
        JuMP.value(x[t-1]), JuMP.objective_value(m)
    end
    function argmin_Q(t::Int) # which is derived by delete 1 line from Q(t::Int, x_tm1)
        argmin_Q_t = Q_kernel(t, false, NaN)[1]
    end
    function min_Q(t::Int)
        min_val_of_Q_t = Q_kernel(t, false, NaN)[2]
    end
    function Q(t::Int, input_xm1::Float64)
        Q_at_input_xm1 = Q_kernel(t, true, input_xm1)[2]
    end
    function Q(t::Int)::Function
        x -> Q(t, x)
    end
    function value_pri()
        the_obj_val_of_MSP = Q(1, x0)
    end
end

function Q_hat(t::Int, trial_xtm1::Float64)::Float64 # revised 20/12
    t == T + 1 && return 0. # the dummy function with value 0
    @assert 2 <= t <= T
    @assert boundDict["xl"][t-1] <= trial_xtm1 <= boundDict["xu"][t-1] 
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m) # actually we can solve directly, but for convenience we import Gurobi
    JuMP.@variable(m, boundDict["hatQl"][t] <= tha <= boundDict["hatQu"][t])
    lD = hatQ[t]
    for (cx, ct, rhs, old) in zip(lD["cx"], lD["ct"], lD["rhs"], lD["is_inferior"]) # this is the cutting plane model of Q_t(x[t-1])
        old || JuMP.@constraint(m, cx * trial_xtm1 + ct * tha >= rhs)
    end
    JuMP.@objective(m, Min, tha)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    return JuMP.value(tha)
end
function gen_trial_xtm1(t::Int, x_input_tm2::Float64)
    @assert 2 <= t <= T+1 # obj = f_1(x1) + Q_2(x1) when t = 2 ;; obj = f_T(xT) when t = T+1
    @assert boundDict["xl"][t-2] <= x_input_tm2 <= boundDict["xu"][t-2]
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    JuMP.@variable(m, boundDict["hatQl"][t] <= tha <= boundDict["hatQu"][t])
    JuMP.@variable(m, boundDict["xl"][t-1] <= x_tm1 <= boundDict["xu"][t-1])
    JuMP.@variable(m, boundDict["fl"][t-1] <= f_tm1 <= boundDict["fu"][t-1])
    JuMP.@variable(m, a >= 0.)
    JuMP.@variable(m, cb_tm1, Bin)
    JuMP.@constraint(m, a >=  x_tm1)
    JuMP.@constraint(m, a >= -x_tm1)
    JuMP.@constraint(m, f_tm1 == beta^(t-2) * a)
    JuMP.@constraint(m, x_tm1 == x_input_tm2 + (2. * cb_tm1 - 1.) + xi[t-1])
    lD = hatQ[t]
    for (cx, ct, rhs, old) in zip(lD["cx"], lD["ct"], lD["rhs"], lD["is_inferior"])
        old || JuMP.@constraint(m, cx * x_tm1 + ct * tha >= rhs)
    end
    JuMP.@objective(m, Min, f_tm1 + tha)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(x_tm1), JuMP.value(f_tm1), JuMP.value(tha), JuMP.objective_value(m) # those in train_trials_record
end

if true #【Functions】algorithm1 and its affiliates, t in 2:T
    function Q_ast(t, pai, pai0) # see (12)
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, boundDict["xl"][t-1] <= x_tm1 <= boundDict["xu"][t-1]) # x[t-1] as input param
        JuMP.@variable(m, boundDict["xl"][t] <= x <= boundDict["xu"][t]) # x[t]
        JuMP.@variable(m, boundDict["fl"][t] <= f_t <= boundDict["fu"][t])
        JuMP.@variable(m, abs_x >= 0.) # can only used in Min-obj
        JuMP.@variable(m, cb, Bin)
        JuMP.@variable(m, boundDict["hatQl"][t+1] <= tha <= boundDict["hatQu"][t+1]) # cutting plane model of Q_{t+1}(x[t])
        JuMP.@constraint(m, abs_x >= x)
        JuMP.@constraint(m, abs_x >= -x)
        JuMP.@constraint(m, x == x_tm1 + (2. * cb - 1.) + xi[t])
        lD = hatQ[t+1]
        for (cx, ct, rhs, old) in zip(lD["cx"], lD["ct"], lD["rhs"], lD["is_inferior"]) # this is the cutting plane model of Q_t(x[t-1])
            old || JuMP.@constraint(m, cx * x + ct * tha >= rhs)
        end
        JuMP.@constraint(m, f_t == beta^(t-1) * abs_x)
        coef_pai0 = f_t + tha
        JuMP.@objective(m, Min, pai * x_tm1 + pai0 * coef_pai0)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        JuMP.objective_value(m), JuMP.value(x_tm1), JuMP.value(coef_pai0) # [1]: value, [2][3]: coeffs of pai and pai0
    end
    function algo1_ini(t, x_hat, tha_hat)
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, pai) # bound implicitly enforced later
        JuMP.@variable(m, pai0 >= 0.)
        # |pai| + pai0 <= 1 
        JuMP.@constraint(m, 1. - pai0 >= pai)
        JuMP.@constraint(m, 1. - pai0 >= -pai)
        if t != T
            hatQast[t] = Dict("cp" => Float64[],"cp0" => Float64[]) # initialization
        end
        if isempty(hatQast[t]["cp"]) # build Q̂* from scratch; this `if` is used to ensure that at least 1 cut exists for Q̂*
            JuMP.@objective(m, Max, - x_hat * pai - tha_hat * pai0) 
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL
            pai_ini, pai0_ini = JuMP.value(pai), JuMP.value(pai0) # get the initial feas. solu of (17): 1.0, 0.0
            @assert pai0_ini >= 0. # otherwise it's Gurobi's fault
            val, c_pai, c_pai0 = Q_ast(t, pai_ini, pai0_ini)
            push!(hatQast[t]["cp"], c_pai)
            push!(hatQast[t]["cp0"], c_pai0)
        end
        JuMP.@variable(m, phi)
        JuMP.@objective(m, Max, phi - x_hat * pai - tha_hat * pai0) # the surrogate objective of (17)
        lD = hatQast[t]
        for (cp, cp0) in zip(lD["cp"], lD["cp0"])
            JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0)
        end
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        ub = JuMP.objective_value(m)
        pai, pai0 = JuMP.value(pai), JuMP.value(pai0) # a feas. solu of (17)
        @assert pai0 >= 0. # otherwise it's Gurobi's fault
        val, c_pai, c_pai0 = Q_ast(t, pai, pai0) # this is a new one compared to the initializing above
        lb = val - pai * x_hat - pai0 * tha_hat # obj value corr to the feas. solu just derived
        return ub, lb, pai, pai0, val, c_pai, c_pai0
    end
    function algorithm1(ite, t, x_hat, tha_hat, delta = 0.9) # generate a (13) cut for Q_t(⋅) that cut off trial (x̂t-1, curr_hat_Q_t(x̂t-1))
        @assert 2 <= t <= T
        icbt = zeros(4) # [1]=lb  [2]=pai  [3]=pai0 [4]=Q^ast_t(pai, pai0)  All incumbent 
        ub, icbt[1], icbt[2], icbt[3], icbt[4], c_pai, c_pai0 = algo1_ini(t, x_hat, tha_hat) # (3.8263752000000006, 0.0, 0.0, 1.0, 1.4450970583895657, 0.0)
        while true
            if ub <= 1e-6
                @info "This trail point cannot be cut off with algorithm1 (is saturated)." t ite x_hat tha_hat ub
                return false
            elseif icbt[1] > (1.0 - delta) * ub + 1e-6 # cut off: by algo design
                if icbt[3] < 0.
                    error("Algo1: very serious wrong!")
                elseif icbt[3] == 0.
                    error("Algo1: intend to generate feas. cut, which is abnormal!")
                else # give birth to a normal opt. cut
                    cut_off_dist = icbt[4] - (icbt[2] * x_hat + icbt[3] * tha_hat)
                    @assert cut_off_dist > 1e-6 # cut off: the authentic certificate 
                    if isempty(hatQ[t]["cut_id"])
                        push!(hatQ[t]["cut_id"], 1)
                    else
                        push!(hatQ[t]["cut_id"], 1 + maximum(hatQ[t]["cut_id"]))
                    end
                    push!(hatQ[t]["gen_in_ite"], ite)
                    push!(hatQ[t]["x_trial"], x_hat)
                    push!(hatQ[t]["t_trial"], tha_hat)
                    push!(hatQ[t]["cx"], icbt[2])
                    push!(hatQ[t]["ct"], icbt[3])
                    push!(hatQ[t]["rhs"], icbt[4])
                    push!(hatQ[t]["is_inferior"], false)
                    @debug "algo1 SUCCESS: cut off!" cut_off_dist strengthed_Qt_ind=t icbt
                    return true
                end
            else
                push!(hatQast[t]["cp"], c_pai)
                push!(hatQast[t]["cp0"], c_pai0)
                m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
                JuMP.set_silent(m)
                JuMP.@variable(m, pai)
                JuMP.@variable(m, pai0 >= 0.)
                JuMP.@constraint(m, 1. - pai0 >=  pai) # |pai| + pai0 <= 1 
                JuMP.@constraint(m, 1. - pai0 >= -pai) # |pai| + pai0 <= 1 
                JuMP.@variable(m, phi)
                JuMP.@objective(m, Max, phi - x_hat * pai - tha_hat * pai0) # the surrogate objective of (17)
                lD = hatQast[t]
                for (cp, cp0) in zip(lD["cp"], lD["cp0"])
                    JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0)
                end
                JuMP.optimize!(m)
                @assert JuMP.termination_status(m) == JuMP.OPTIMAL
                ub = JuMP.objective_value(m)
                pai, pai0 = JuMP.value(pai), JuMP.value(pai0) # a feas. solu of (17)
                @assert pai0 >= 0. # otherwise it's Gurobi's fault
                val, c_pai, c_pai0 = Q_ast(t, pai, pai0) # this is a new one compared to the initializing above
                lb = val - pai * x_hat - pai0 * tha_hat
                if lb > icbt[1] # the new feasible point (pai, pai0) is superior
                    icbt .= lb, pai, pai0, val
                end
            end
        end
    end
end

function train(t::Int = t, x_tm2_input::Float64 = x0) # to solve the MSP problem
    if 2 < t <= T # welcome info
        @assert boundDict["xl"][t-2] <= x_tm2_input <= boundDict["xu"][t-2]
        @warn "study MSP partially: the first decision is x_t where" t=t-1 stages_of_x_need_decision=T-(t-2)
    elseif t == 2
        @info "MSP start training with" x_0=x_tm2_input stages_of_x_need_decision=T
    else
        error("argument t is invalid!")
    end
    ite, ub = 1, boundDict["fu"][t-1] + boundDict["hatQu"][t] # initial bound of Q_t for a reference only
    if ite == 1 # initialization
        x_tm1_trial = x_tm2_input # for the sake of tidiness
        for s in t:T+1
            x_tm1_trial, f_tm1_induced, tha_trial, lb = gen_trial_xtm1(s, x_tm1_trial) # ignite the fwd process
            if s == t && lb >= ub - 1e-6
                @warn "😅 Gap closed at the First iterate!" ite ub lb x1_solu=x_tm1_trial
                return x_sm1_trial, ub
            end
            push!(train_trials_record[s]["ite"], ite)
            push!(train_trials_record[s]["x_tm1_trial"], x_tm1_trial)
            push!(train_trials_record[s]["f_tm1_induced"], f_tm1_induced)
            push!(train_trials_record[s]["tha_trial"], tha_trial)
            push!(train_trials_record[s]["lb"], lb)
        end
        ub = sum(train_trials_record[s]["f_tm1_induced"][ite] for s in t:T+1) # update directly to save effort
        for s in T:-1:t # x1 - Q2 ;; x8 - Q9 ;; Q9 dummy, thus bwd process is one stage less than fwd [notice]
            x_sm1_trial = train_trials_record[s]["x_tm1_trial"][ite]
            algorithm1(ite, s, x_sm1_trial, Q_hat(s, x_sm1_trial)) || @warn "At ite = 1, bwd phase, there is at least one Q_hat not updated."
        end
    end
    for ite in 2:typemax(Int) # Main Iterate
        x_tm1_trial = x_tm2_input # for the sake of tidiness
        recur_xtrial_stages, corr_cut_ind_vec_vec = Int[], Vector{Int64}[] # recurrence check related
        x_close_tol, value_improve_tol = 1e-5, 1e-6 # recurrence check related
        for s in t:T+1
            x_tm1_trial, f_tm1_induced, tha_trial, lb = gen_trial_xtm1(s, x_tm1_trial) # ignite the fwd process
            if s == t # summary session (for the last ite)
                if lb >= ub - 1e-6
                    @info "😊 Gap closed." ite ub lb x1_solu=x_tm1_trial
                    return x_sm1_trial, ub
                elseif !any([ite-1 in hatQ[stage]["gen_in_ite"] for stage in t:T]) # the last ite doesn't generate a cut for any Q_t
                    @warn "Quit training process due to saturation of cutting planes" ite ub lb gap="$((ub-lb)/ub*100)%" x1_candidate=x_tm1_trial
                    return x_tm1_trial, ub
                else
                    @info ">>>" ite ub lb
                end
            end
            old_ite_ind_vec = findall(x -> x < x_close_tol, abs.(train_trials_record[s]["x_tm1_trial"] .- x_tm1_trial))
            if !isempty(old_ite_ind_vec) # the involved part: recurrence of x_tm1_trial
                if !all( train_trials_record[s]["tha_trial"][old_ite_ind_vec] .<= tha_trial + value_improve_tol )
                    @error "see" ite s x_tm1_trial tha_trial train_trials_record[s]["tha_trial"]
                    error("fwd: recurrence of inferior trials, please check!")
                else # mark the old (all inferior) cuts at stage s
                    cut_ind_vec = Int[]
                    for old_ite in old_ite_ind_vec
                        tmp = findall(x -> x == old_ite, hatQ[s]["gen_in_ite"]) # tmp = Int[] or [i::Int]
                        cut_ind_vec = [cut_ind_vec; tmp]
                    end
                    push!(recur_xtrial_stages, s)
                    push!(corr_cut_ind_vec_vec, cut_ind_vec)
                end
            end
            push!(train_trials_record[s]["ite"], ite)
            push!(train_trials_record[s]["x_tm1_trial"], x_tm1_trial)
            push!(train_trials_record[s]["f_tm1_induced"], f_tm1_induced)
            push!(train_trials_record[s]["tha_trial"], tha_trial)
            push!(train_trials_record[s]["lb"], lb)
        end
        ub = sum(train_trials_record[s]["f_tm1_induced"][ite] for s in t:T+1) # update directly to save effort
        @info " ♠ Start bwd pass"
        for s in T:-1:t
            x_sm1_trial = train_trials_record[s]["x_tm1_trial"][ite]
            cut_is_gened = algorithm1(ite, s, x_sm1_trial, Q_hat(s, x_sm1_trial)) # use an ad hoc theta_trial
            if s in recur_xtrial_stages && cut_is_gened
                hatQ[s]["is_inferior"][corr_cut_ind_vec_vec[findall(x -> x == s, recur_xtrial_stages)[1]]] .= true # ensures that: from now on we can discard inferior cuts
            end
        end
    end
end

if true # Data_Main
    # global_logger(ConsoleLogger(Debug))
    global_logger(ConsoleLogger(Info))
    # global_logger(ConsoleLogger(Warn))
    GRB_ENV = Gurobi.Env() # Gurobi can deal with Quad_NL and MILP, thus Gurobi is competent for these 2 cases
    xi = Float64[1.878476717166122, 1.8552903302256576, -0.5258435328220221, -2.0016417222487712, -2.4571367609213413, -1.977854597034034, -0.1867543863247132, -0.4450970583895657]
    T, beta = 8, .9
    @assert length(xi) == T
    # --------------- try different variaties -(not realized yet!)--------------
    x0 = 2.0 # default 2.0
    t = 2 # default 2
    # --------------- try different variaties ---------------
    hatQ_range = 2:T+1 # we do not model Q_1 because its input is fixed at x0, thus Q_1 is a value v_prim. Q_{T+1} is dummy.
    boundDict = Dict(
        "xl" => OffsetVector([0. for _ in 0:T], 0:T),
        "xu" => OffsetVector([0. for _ in 0:T], 0:T),
        "fl" => [0. for _ in 1:T], # real zeros
        "fu" => [0. for _ in 1:T],
        "hatQl"=> OffsetVector([0. for _ in hatQ_range], hatQ_range), # real zeros; the end is fixed to 0.
        "hatQu"=> OffsetVector([0. for _ in hatQ_range], hatQ_range) # the end is fixed to 0.
    )
    fill_boundDict(T, x0)
    hatQ = OffsetVector([Dict(
        "cut_id" => Int[], # inside manager, linear currently
        "gen_in_ite" => Int[], # this cut is generated from the i'th ite
        "x_trial" => Float64[],
        "t_trial" => Float64[],
        "cx" => Float64[],
        "ct" => Float64[],
        "rhs" => Float64[],
        "is_inferior" => Bool[] # the cut is deemed inferior compared with a later-generated one
        ) for _ in hatQ_range], hatQ_range) # cutting plane model for Q[2:T], while the end is dummy and kept empty
    hatQast = OffsetVector([Dict("cp" => Float64[],"cp0" => Float64[]) for _ in t:T], t:T) # storing affine coeffs for Q*(π, π0), since it doesn't need dummy at end, it's end is T, compared with T+1 in hatQ
    train_trials_record = OffsetVector([Dict(
        "ite" => Int[], # linear currently
        "x_tm1_trial" => Float64[],
        "f_tm1_induced" => Float64[],
        "tha_trial" => Float64[],
        "lb" => Float64[], # "f_tm1_indecud" + "tha_trial"
        ) for _ in hatQ_range], hatQ_range)
end

x_tm1_opt, ub = train(t, x0) # to solve the MSP problem

if true # concluding session
    eval_this_solu_at_Q = Q(t-1, x0)
    diff_ub_between_ans___should_small = ub - eval_this_solu_at_Q
    theoretical_value = min_Q(t-1)
    diff_ub_between_ans___ie_the_abs_err = ub - theoretical_value
    @info "Some references from the brute-force solution" ub eval_this_solu_at_Q diff_ub_between_ans___should_small theoretical_value diff_ub_between_ans___ie_the_abs_err
end

if false #【Plots】
    function L_cut(lD::Dict, ind::Int)::Function  # distil the cut function 
        cx, ct, rhs = lD["cx"][ind], lD["ct"][ind], lD["rhs"][ind]
        @assert ct > 0.
        x -> (rhs - cx * x)/ct
    end
    f = Figure();
    ax = Axis(f[1, 1]) # ,limits = (-4.3, 7.5, 0, 10));
    num_points = 300
    for s in 4:5 # draw the Q functions, through large scale programmings 
        xtm1 = range(boundDict["xl"][s-1], boundDict["xu"][s-1]; length = num_points);
        val = Q.(s, xtm1);
        lines!(ax, xtm1, val)
        text!(ax, xtm1[num_points ÷ 2], val[num_points ÷ 2]; text = "Q$s")
        val = Q_hat.(s, xtm1);
        lines!(ax, xtm1, val)
        text!(ax, xtm1[1], val[1]; text = "hatQ$s")
    end
    s = 4
    xtm1 = range(boundDict["xl"][s-1], boundDict["xu"][s-1]; length = num_points);
    for c in [3, 5, 6] # draw some cuts
        fc = L_cut(hatQ[s], c)
        val = fc.(xtm1)
        lines!(ax, xtm1, val)
        text!(ax, xtm1[1], val[1]; text = "cut$c")
    end
end

if false # helper functions
    function cut_viewer(s::Int)
        for (cx, ct, rhs, old) in zip(hatQ[s]["cx"], hatQ[s]["ct"], hatQ[s]["rhs"], hatQ[s]["is_inferior"])
            old || println(cx," ",ct," ",rhs)
        end
    end
end

if false # generate a different random process
    function udrv() # uniformly distributed rv
        rand(Distributions.Uniform(-3,2))
    end
    xi = [udrv() for t in 1:T]
    ak47 = 3.
end





# column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
# norm_sense, norm_arg_num = Cdouble(1.0), Cint(1)

using CairoMakie
using OffsetArrays
using Logging
import JuMP
import Gurobi
import LinearAlgebra
import Distributions

# use Lag cut (13) to generate (very tight, sometimes precise) cutting plane model for MS-MILP value functions
# 19/12/23

function get_x_bound(t::Int, upper::Bool) # used for initialization, get the lb of state variables.
    @assert 1 <= t <= T
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
if true #【Functions】cutting plane model Q̂ related
    function Q_hat_kernel(t::Int, eval_mode::Bool, input_xm1::Float64)
        @assert 2 <= t <= T
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, boundDict["xl"][t-1] <= x <= boundDict["xu"][t-1]) # this x is the variable of Q_t(⋅), thus x means x[t-1]
        if eval_mode
            @assert boundDict["xl"][t-1] <= input_xm1 <= boundDict["xu"][t-1]
            JuMP.fix(x, input_xm1; force = true)
        end
        JuMP.@variable(m, boundDict["hatQl"][t] <= tha <= boundDict["hatQu"][t]) # aux objective. Notice: the lower bound here is as important as those L-cuts.
        lD = hatQ[t] # local Dict
        for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"]) # this is the cutting plane model of Q_t(x[t-1])
            JuMP.@constraint(m, cx * x + ct * tha >= rhs)
        end
        JuMP.@objective(m, Min, tha)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        JuMP.value(x), JuMP.value(tha)
    end
    function gen_trial_xtm1(t::Int)
        trial_xtm1, lb_Q_t = Q_hat_kernel(t, false, NaN)
    end
    function Q_hat(t::Int, input_xm1::Float64) # from gen_trial_xtm1
        val_Q_hat_at_input_xm1 = Q_hat_kernel(t, true, input_xm1)[2]
    end
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
        for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"])
            JuMP.@constraint(m, cx * x + ct * tha >= rhs)
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
    function algorithm1(t, ite, x_hat, tha_hat) # generate a (13) cut for Q_t(⋅) that cut off trial (x̂t-1, curr_hat_Q_t(x̂t-1))
        @assert 2 <= t <= T
        delta = 0.9
        icbt = zeros(4) # [1]=lb  [2]=pai  [3]=pai0 [4]=Q^ast_t(pai, pai0)  All incumbent 
        ub, icbt[1], icbt[2], icbt[3], icbt[4], c_pai, c_pai0 = algo1_ini(t, x_hat, tha_hat) # (3.8263752000000006, 0.0, 0.0, 1.0, 1.4450970583895657, 0.0)
        while true
            if ub <= 1e-6
                @warn "This trail point cannot be cut off with algorithm1." t ite x_hat tha_hat ub
                return nothing
            end
            if icbt[1] > (1.0 - delta) * ub + 1e-6 
                if icbt[3] > 0.
                    @assert icbt[2] * x_hat + icbt[3] * tha_hat < icbt[4] - 1e-6
                    if isempty(hatQ[t]["cut_id"])
                        push!(hatQ[t]["cut_id"], 1)
                    else
                        push!(hatQ[t]["cut_id"], 1 + maximum(hatQ[t]["cut_id"]))
                    end
                    push!(hatQ[t]["gen_in_ite"], ite)
                    push!(hatQ[t]["cx"], icbt[2])
                    push!(hatQ[t]["ct"], icbt[3])
                    push!(hatQ[t]["rhs"], icbt[4])
                    push!(hatQ[t]["x_trial"], x_hat)
                    push!(hatQ[t]["t_trial"], tha_hat)
                    @debug "algo1 SUCCESS: cut with distance" strengthed_Qt_ind=t icbt
                    return nothing
                elseif icbt[3] == 0.
                    error("Algo1: how to deal with coeff_tha == 0. ??? ")
                else
                    error("Algo1: very serious wrong!")
                end
            end
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

function fwd(t, x_tm1) # only the final stage Q is precise, because it involves no trailing term
    @assert 1 <= t <= T
    # example: t = 7, input = x_tm1 = x[6]
    # According to the input x[6], we `gen` an `ub` corr to the feasible policy according to f_{7} and CURRENT Q_hat_{8}  
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    JuMP.@variable(m, cb, Bin)
    JuMP.@variable(m, abs_x >= 0.) # can only used in Min-obj
    JuMP.@variable(m, boundDict["xl"][t] <= x <= boundDict["xu"][t]) # x[t]
    JuMP.@variable(m, boundDict["fl"][t] <= f_t <= boundDict["fu"][t])
    JuMP.@variable(m, boundDict["hatQl"][t+1] <= tha <= boundDict["hatQu"][t+1]) # cutting plane model of Q_{t+1}(x[t])
    JuMP.@constraint(m, abs_x >= x)
    JuMP.@constraint(m, abs_x >= -x)
    JuMP.@constraint(m, x == x_tm1 + (2. * cb - 1.) + xi[t])
    JuMP.@constraint(m, f_t == beta^(t-1) * abs_x)
    lD = hatQ[t+1]
    for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"])
        JuMP.@constraint(m, cx * x + ct * tha >= rhs)
    end
    JuMP.@objective(m, Min, f_t + tha)
    JuMP.optimize!(m)
    if JuMP.termination_status(m) != JuMP.OPTIMAL
        @error "in fwd()" JuMP.termination_status(m) t x_tm1
        error("fwd(): stop!->")
    end
    # [1] solution of this stage as trial for the next
    # [2] immediate cost of this stage,
    # [3] quasi upper bound, because the trailing term is a surrogate. It's precise when t == T though.
    JuMP.value(x), JuMP.value(f_t), JuMP.objective_value(m)
end
function train(t::Int) # train a cutting plane model for Q_t
    @assert 2 <= t <= T
    ite, ub = 1, boundDict["hatQu"][t] # initial bound of Q_t for a reference only
    if ite == 1 # initialization
        x_sm1_trial, lb_Q_t = gen_trial_xtm1(t) # x6_trial, lb_Q_7
        lb_Q_t >= ub - 1e-6 && error("Convergent Without Training! Please Check If the Model is Already Well Trained!")
        push!(bounds_vec, (ite, lb_Q_t, ub)) # record the Vital Info 
        for s in t:T # forward pass
            push!(train_trials_record[s]["ite"], ite)
            push!(train_trials_record[s]["x_trial"], x_sm1_trial)
            push!(train_trials_record[s]["t_trial"], lb_Q_t)
            x_sm1_trial, ft_vec[s], qub_vec[s] = fwd(s, x_sm1_trial) # notice we store the return at LHS
        end
        ub = sum(ft_vec)
        for s in T:-1:t
            x_sm1_trial = train_trials_record[s]["x_trial"][ite]
            algorithm1(s, ite, x_sm1_trial, Q_hat(s, x_sm1_trial)) # use an ad hoc theta_trial 
        end
    end
    for ite in 2:typemax(Int)
        x_sm1_trial, lb_Q_t = gen_trial_xtm1(t) # x6_trial, lb_Q_7
        if lb_Q_t >= ub - 1e-6 # This is almost IMPOSSIBLE because the value function of MILP is NON-CONVEX
            @info "😊 Gap for Q_t() is closed!" ite lb=lb_Q_t ub
            return argmin_Q_t = x_sm1_trial
        else # conclusion of this ite
            @info ">>> Vital Info in Main Iterate" ite lb=lb_Q_t ub
        end
        push!(bounds_vec, (ite, lb_Q_t, ub)) # record the Vital Info 
        @info " ⊗ Start fwd pass with current hat_Q's" x_input=x_sm1_trial  x_input_ind=t-1 thus_t=t lb_Q_t
        # ----- Above are the easy part -----
        if !any([ite-1 in hatQ[s]["gen_in_ite"] for s in t:T]) # the last ite doesn't generate a cut at any stage
            @warn "Quit training process due to saturation of cutting planes"
            return current_best_solution = x_sm1_trial
        end
        for s in t:T # forward pass
            mindist, minpla = findmin(abs.(train_trials_record[s]["x_trial"] .- x_sm1_trial))
            if mindist > 1e-5 || train_trials_record[s]["t_trial"][minpla] < lb_Q_t
                push!(train_trials_record[s]["ite"], ite)
                push!(train_trials_record[s]["x_trial"], x_sm1_trial)
                push!(train_trials_record[s]["t_trial"], lb_Q_t)
            end
            if mindist <= 1e-5 # abnormal state 
                if train_trials_record[s]["t_trial"][minpla] < lb_Q_t
                    @debug "fwd: recurrence of trials, but θ is updated, thus accept." oldval = train_trials_record[s]["t_trial"][minpla]  newval = lb_Q_t
                else
                    @error "fwd: recurrence of inferior trials" stage=s x_sm1_trial lb_Q_t ite CMP_ite_ind=minpla
                end
            end
            x_sm1_trial, ft_vec[s], qub_vec[s] = fwd(s, x_sm1_trial) # notice we store the return at LHS
            # @info "fwd: after 1 step," x_ind=s  x_solu=x_sm1_trial  generates_ft=ft_vec[s]  aug_quasi_cost=qub_vec[s]
        end
        ub = sum(ft_vec) # We miss the process of comparation with the best ub so far
        @info " ♠ Start bwd pass"
        for s in T:-1:t
            x_sm1_trial = train_trials_record[s]["x_trial"][ite]
            algorithm1(s, ite, x_sm1_trial, Q_hat(s, x_sm1_trial)) # use an ad hoc theta_trial 
            # @info "bwd: end of ite at" s current_lb=gen_trial_xtm1(s)[2]   compared_ub=qub_vec[s]
        end
    end
end

if true # Data_Main
    global_logger(ConsoleLogger(Info))
    # global_logger(ConsoleLogger(Debug))
    GRB_ENV = Gurobi.Env() # Gurobi can deal with Quad_NL and MILP, thus Gurobi is competent for these 2 cases
    T, beta = 8, .9
    xi = [1.878476717166122, 1.8552903302256576, -0.5258435328220221, -2.0016417222487712, -2.4571367609213413, -1.977854597034034, -0.1867543863247132, -0.4450970583895657]
    boundDict = Dict(
        "xl" => OffsetVector([0. for _ in 0:T], 0:T),
        "xu" => OffsetVector([0. for _ in 0:T], 0:T),
        "fl" => [0. for _ in 1:T], # real zeros
        "fu" => [0. for _ in 1:T],
        "hatQl"=> OffsetVector([0. for _ in 2:T+1], 2:T+1), # real zeros; the end is fixed to 0.
        "hatQu"=> OffsetVector([0. for _ in 2:T+1], 2:T+1) # the end is fixed to 0.
    )
    boundDict["xl"][0], boundDict["xu"][0] = -4., 4.
    x0 = 2. # the default input state of the MS_SP
    @assert boundDict["xl"][0] <= x0 <= boundDict["xu"][0] 
    for s in 1:T
        boundDict["xl"][s] = get_x_bound(s, false)
        boundDict["xu"][s] = get_x_bound(s, true)
    end
    boundDict["fu"] .= [beta^(s-1) * max(abs(boundDict["xl"][s]), abs(boundDict["xu"][s])) for s in 1:T]
    for s in 2:T
        boundDict["hatQu"][s] = sum(boundDict["fu"][i] for i in s:T)
    end
    hatQ = OffsetVector([Dict(
        "cut_id" => Int[], # inside manager
        "gen_in_ite" => Int[], # this cut is generated from the i'th iterate
        "x_trial" => Float64[],
        "t_trial" => Float64[],
        "cx" => Float64[],
        "ct" => Float64[],
        "rhs" => Float64[]
        ) for t in 2:T+1], 2:T+1) # cutting plane model for Q[2:T], while the end is dummy and kept empty
    hatQast = OffsetVector([Dict("cp" => Float64[],"cp0" => Float64[]) for _ in 2:T], 2:T) # storing affine coeffs for Q*(π, π0)
end

t = 2 # t in 2:T
train_trials_record = OffsetVector([Dict(
    "ite" => Int[],
    "x_trial" => Float64[],
    "t_trial" => Float64[]
    ) for _ in t:T], t:T)
ft_vec = OffsetVector([0. for _ in t:T], t:T) # immediate costs
qub_vec = OffsetVector([0. for _ in t:T], t:T) # augmented costs corr to the immediate costs
bounds_vec = Tuple{Int, Float64, Float64}[] # the crucial indication of the training process
x_tm1_opt = train(t) # the best solution given by our method

@info "Some references" trained_value=Q(t, x_tm1_opt) theoretical_value=min_Q(t)

if false #【Plots】
    function L_cut(lD::Dict, ind::Int)::Function  # distil the cut function 
        cx, ct, rhs = lD["cx"][ind], lD["ct"][ind], lD["rhs"][ind]
        @assert ct > 0.
        x -> (rhs - cx * x)/ct
    end
    f = Figure();
    ax = Axis(f[1, 1]) # ,limits = (-4.3, 7.5, 0, 10));
    num_points = 300
    for s in t:t+1 # draw the Q functions, through large scale programmings 
        xtm1 = range(boundDict["xl"][s-1], boundDict["xu"][s-1]; length = num_points);
        val = Q.(s, xtm1);
        lines!(ax, xtm1, val)
        text!(ax, xtm1[num_points ÷ 2], val[num_points ÷ 2]; text = "Q$s")
        val = Q_hat.(s, xtm1);
        lines!(ax, xtm1, val)
        text!(ax, xtm1[1], val[1]; text = "hatQ$s")
    end
    # s = t
    # xtm1 = range(boundDict["xl"][s-1], boundDict["xu"][s-1]; length = num_points);
    # f1 = L_cut(hatQ[s], 3)
    # val = f1.(xtm1)
    # lines!(ax, xtm1, val)
    # text!(ax, xtm1[1], val[1]; text = "cut3")
    # f2 = L_cut(hatQ[s], 4)
    # val = f2.(xtm1)
    # lines!(ax, xtm1, val)
    # text!(ax, xtm1[1], val[1]; text = "cut4")
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

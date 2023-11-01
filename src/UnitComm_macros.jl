import MathOptInterface as MOI
import Gurobi
# import SCIP
# 2023.10.29, consider sin fuel cost
const MODELING = 0 # keys for the dict: 0,1,2

if true # functions
    function allocate_dict()
        return Dict{Int64,Union{Float64, MOI.VariableIndex}}(0 => MOI.VariableIndex(0), 1 => Inf, 2 => NaN)
    end
    function allocate_x_like()
        return [allocate_dict() for g in 1:Gs, t in 1:Ts]
    end
    function allocate_dbar_like()
        return [allocate_dict() for t in 1:Ts]
    end
    function vio_ge(lhs::Float64, rhs::Float64)
        return max(rhs - lhs, 0.)
    end
    function vio_le(lhs::Float64, rhs::Float64)
        return max(lhs - rhs, 0.)
    end
    function vio_eq(lhs::Float64, rhs::Float64)
        return abs(lhs - rhs)
    end
    function vio_int(a::Float64)
        return abs(a - round(a))
    end
    function vio_01(a::Float64)
        return max(vio_ge(a,0.), vio_le(a,1.), vio_int(a))
    end
    vio_fun_dic_1 = Dict(
        "ge" => vio_ge,
        "le" => vio_le,
        "eq" => vio_eq
    )
    set_fun_dic_1 = Dict(
        "ge" => MOI.GreaterThan,
        "le" => MOI.LessThan,
        "eq" => MOI.EqualTo
    )
    vio_fun_dic_2 = Dict(
        "int" => vio_int,
        "01" => vio_01
    )
    set_fun_dic_2 = Dict(
        "int" => MOI.Integer,
        "01" => MOI.ZeroOne
    )
end
if true # macros
    macro add_free_mat(ki,x)
        return :(
            for g in 1:Gs, t in 1:Ts
                if $ki == MODELING
                    $x[g,t][$ki] = MOI.add_variable(o)
                else
                    $x[g,t][$ki] = MOI.get(o,MOI.VariablePrimal(),$x[g,t][MODELING])
                end
            end
        )
    end
    macro add_continuous_mat(ki,x,str,rhs)
        return :(
            for g in 1:Gs, t in 1:Ts
                if $ki == MODELING
                    $x[g,t][$ki] = MOI.add_variable(o)
                    MOI.add_constraint(o, $x[g,t][$ki], set_fun_dic_1[$str]($rhs))
                else
                    $x[g,t][$ki] = MOI.get(o,MOI.VariablePrimal(),$x[g,t][MODELING])
                    maxvio_cal[1] = max(maxvio_cal[1], vio_fun_dic_1[$str]($x[g,t][$ki], $rhs))
                end
            end
        )
    end
    macro add_continuous_vec(ki,x,str,rhs)
        return :(
            for t in 1:Ts
                if $ki == MODELING
                    $x[t][$ki] = MOI.add_variable(o)
                    MOI.add_constraint(o, $x[t][$ki], set_fun_dic_1[$str]($rhs))
                else
                    $x[t][$ki] = MOI.get(o,MOI.VariablePrimal(),$x[t][MODELING])
                    maxvio_cal[1] = max(maxvio_cal[1], vio_fun_dic_1[$str]($x[t][$ki], $rhs))
                end
            end
        )
    end
    macro add_integer_mat(ki,x,str)
        return :(
            for g in 1:Gs, t in 1:Ts
                if $ki == MODELING
                    $x[g,t][$ki] = MOI.add_variable(o)
                    MOI.add_constraint(o, $x[g,t][$ki], set_fun_dic_2[$str]())
                else
                    $x[g,t][$ki] = MOI.get(o,MOI.VariablePrimal(),$x[g,t][MODELING])
                    maxvio_cal[1] = max(maxvio_cal[1], vio_fun_dic_2[$str]($x[g,t][$ki]))
                end
            end
        )
    end
end
if true # Data Region
    G_keys = ["comm_ini","gen_ini","cgbar","cgund","om_cost","su_cost","sd_cost","rgbar","rgund","a","b","c","v_a","v_b","v_c","v_d","v_e"];
    G_raw = [[0.0, 0.0, 1.13, 0.48, 0.0, 171.6, 17.0, 0.28, 0.27, -0.24, 1.02, 0.0, 6.16, 49.01, 15.12, 1.13, 5.0], 
    [1.0, 2.2, 2.82, 0.85, 0.0, 486.81, 49.0, 0.9, 0.79, -0.3, 1.1, 0.0, 0.22, 61.19, 30.33, 2.82, 5.0], 
    [0.0, 0.0, 3.23, 0.84, 0.0, 503.34, 50.0, 1.01, 1.0, -0.24, 1.04, 0.0, 0.28, 54.35, 30.58, 3.23, 5.0]];
    G_vals = [[G_raw[j][i] for j in 1:3] for i in 1:17];
    G = Dict(k=>v for (k,v) in zip(G_keys,G_vals));
    #---------------------------- Optional: View the Data ----------------------------
    # import DataFrames
    # dfG = DataFrames.DataFrame(G)
    #---------------------------- Optional: View the Data ----------------------------
    demand = 0.85 * [3.06 2.91 2.71 2.7 2.73 2.91 3.38 4.01 4.6 4.78 4.81 4.84 4.89 4.44 4.57 4.6 4.58 4.47 4.32 4.36 4.5 4.27 3.93 3.61]
    emission_price = 25. # in the obj, 4th term
    demand_penalty = 500. # in the obj, last term
    const Gs, Ts = length(G["gen_ini"]), length(demand)
end
GRB_ENV = Gurobi.Env();
if false # these modifications have to be made before generating an optimizer `o`, if do not modify, the "FuncPieceError" is 1e-3 by default.
    ec = Gurobi.GRBsetintparam(GRB_ENV, "FuncPieces", Cint(-1))
    @assert ec === Cint(0)
    ec = Gurobi.GRBsetdblparam(GRB_ENV, "FuncPieceError", Cdouble(1e-6)) # cannot be finer than 1e-6
    @assert ec === Cint(0)
end
o = Gurobi.Optimizer(GRB_ENV); # create an optimizer after an ENV's settings are determined
a_first_feasibility_solve = true # only do it once at the beginning
if true # allocate space for variables
    x, y, ybar, yund = allocate_x_like(), allocate_x_like(), allocate_x_like(), allocate_x_like(); # ybar => start-up
    dbar,dund = allocate_dbar_like(),allocate_dbar_like(); # dbar => deficit, dund => surplus
    sinarg, absarg, absret = allocate_x_like(), allocate_x_like(), allocate_x_like(); # due to sinret == absarg, we skip introducint sinret
end


ki = MODELING


if true # add variables or get the decision values
    ki != MODELING && (maxvio_cal = [0.]) # set the violation_accumulator
    @add_continuous_mat(ki, x, "ge", 0.) # 37
    ki != MODELING && @info "x >= 0" maxvio_cal[1]
    @add_integer_mat(ki, y, "01") # 39.1
    ki != MODELING && @info "y in 0,1" maxvio_cal[1]
    @add_integer_mat(ki, ybar, "01") # 39.2
    ki != MODELING && @info "ybar in 0,1" maxvio_cal[1]
    @add_integer_mat(ki, yund, "01") # 39.3
    ki != MODELING && @info "yund in 0,1" maxvio_cal[1]
    @add_continuous_vec(ki, dbar, "ge", 0.) # 38.1
    ki != MODELING && @info "dbar >= 0" maxvio_cal[1]
    @add_continuous_vec(ki, dund, "ge", 0.) # 38.1
    ki != MODELING && @info "dund >= 0" maxvio_cal[1]
    @add_free_mat(ki, sinarg)
    @add_free_mat(ki, absarg)
    @add_free_mat(ki, absret)   
end

if true # Other constraints (excluding General constraints :sin and :abs)
    for g in 1:Gs, t in 1:Ts # Complete the rest of the fuel cost
        lhs = G["v_e"][g] * x[g,t][ki] + sinarg[g,t][ki]
        rhs = G["v_e"][g] * G["cgund"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.EqualTo(rhs)) # sinarg = ...
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_eq(lhs,rhs))
        end
    end
    ki != MODELING && @info "fuel cost completion" maxvio_cal[1]
    for t in 1:Ts # 30
        lhs = sum(1. * x[g,t][ki] for g in 1:Gs) + dbar[t][ki] - dund[t][ki]
        rhs = demand[t]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.EqualTo(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_eq(lhs,rhs))
        end
    end
    ki != MODELING && @info "demand balance" maxvio_cal[1]
    for g in 1:Gs, t in 1:Ts # 31
        lhs = x[g,t][ki] - G["cgbar"][g] * y[g,t][ki]
        rhs = 0.
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.LessThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_le(lhs,rhs))
        end
    end
    ki != MODELING && @info "(31)" maxvio_cal[1]
    for g in 1:Gs, t in 1:Ts # 32
        lhs = x[g,t][ki] - G["cgund"][g] * y[g,t][ki]
        rhs = 0.
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.GreaterThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_ge(lhs,rhs))
        end
    end
    ki != MODELING && @info "(32)" maxvio_cal[1]
    for t in 1:1, g in 1:Gs # 33_1
        lhs = 1. * x[g,t][ki]
        rhs = G["gen_ini"][g] + G["cgund"][g] + ( G["rgbar"][g] - G["cgund"][g] ) * G["comm_ini"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.LessThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_le(lhs,rhs))
        end
    end
    ki != MODELING && @info "(33_1)" maxvio_cal[1]
    for t in 2:Ts, g in 1:Gs # 33_2
        lhs = -1. * x[g,t-1][ki] + 1. * x[g,t][ki] + (G["cgund"][g] - G["rgbar"][g]) * y[g,t-1][ki]
        rhs = G["cgund"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.LessThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_le(lhs,rhs))
        end
    end
    ki != MODELING && @info "(33_2)" maxvio_cal[1]
    for t in 1:1, g in 1:Gs # 34_1
        lhs = -1. * x[g,t][ki] + (G["cgund"][g] - G["rgund"][g]) * y[g,t][ki]
        rhs = G["cgund"][g] - G["gen_ini"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.LessThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_le(lhs,rhs))
        end
    end
    ki != MODELING && @info "(34_1)" maxvio_cal[1]
    for t in 2:Ts, g in 1:Gs # 34_2
        lhs = 1. * x[g,t-1][ki] - 1. * x[g,t][ki] + (G["cgund"][g] - G["rgund"][g]) * y[g,t][ki]
        rhs = G["cgund"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.LessThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_le(lhs,rhs))
        end
    end
    ki != MODELING && @info "(34_2)" maxvio_cal[1]
    for t in 1:1, g in 1:Gs # 35_1
        lhs = 1. * ybar[g,t][ki] - 1. * y[g,t][ki]
        rhs = -1. * G["comm_ini"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.GreaterThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_ge(lhs,rhs))
        end
    end
    ki != MODELING && @info "(35_1)" maxvio_cal[1]
    for t in 2:Ts, g in 1:Gs # 35_2
        lhs = 1. * ybar[g,t][ki] - 1. * y[g,t][ki] + 1. * y[g,t-1][ki]
        rhs = 0.
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.GreaterThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_ge(lhs,rhs))
        end
    end
    ki != MODELING && @info "(35_2)" maxvio_cal[1]
    for t in 1:1, g in 1:Gs # 36_1
        lhs = 1. * yund[g,t][ki] + 1. * y[g,t][ki]
        rhs = G["comm_ini"][g]
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.GreaterThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_ge(lhs,rhs))
        end
    end
    ki != MODELING && @info "(36_1)" maxvio_cal[1]
    for t in 2:Ts, g in 1:Gs # 36_2
        lhs = 1. * yund[g,t][ki] + 1. * y[g,t][ki] - 1. * y[g,t-1][ki]
        rhs = 0.
        if ki == MODELING
            MOI.add_constraint(o, lhs, MOI.GreaterThan(rhs))
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_ge(lhs,rhs))
        end
    end
    ki != MODELING && @info "(36_2)" maxvio_cal[1]
end

if true # Nonlinear Portion with Gurobi's General constraint
    for g in 1:Gs, t in 1:Ts
        if ki == MODELING
            ec = Gurobi.GRBaddgenconstrSin(
                o, "sin_constr($g,$t)",
                Gurobi.c_column(o, sinarg[g,t][ki]), # arg
                Gurobi.c_column(o, absarg[g,t][ki]),
                ""
            )
            @assert ec === Cint(0)
            ec = Gurobi.GRBaddgenconstrAbs(
                o, "abs_constr($g,$t)",
                Gurobi.c_column(o, absret[g,t][ki]), 
                Gurobi.c_column(o, absarg[g,t][ki]) # arg
            )
            @assert ec === Cint(0)
        else
            maxvio_cal[1] = max(maxvio_cal[1], vio_eq(sin(sinarg[g,t][ki]), absarg[g,t][ki]))
            maxvio_cal[1] = max(maxvio_cal[1], vio_eq(abs(absarg[g,t][ki]), absret[g,t][ki]))
        end
    end
    ki != MODELING && @info "sin and abs" maxvio_cal[1]
end

if a_first_feasibility_solve && (ki == MODELING) # Solve the feasibility system (No obj established so far) to get a random solution first
    a_first_feasibility_solve = false
    @info "Do the feasibility_solve before any obj are added."
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    maxvio_grb::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio"))
    @info "feasible_solution with" maxvio_grb
end

if true # build the objective process; physical intermediate expressions are indented, others are component obj functions
    cgtf = [G["v_a"][g] * x[g,t][ki] * x[g,t][ki] + G["v_b"][g] * x[g,t][ki] + G["v_c"][g] * y[g,t][ki] + G["v_d"][g] * absret[g,t][ki] for g in 1:Gs, t in 1:Ts] # cgtf(...,absret[g,t])
    fuel_cost = sum(cgtf)
        demand_un_meet = [1. * dbar[t][ki] for t in 1:Ts]
        demand_un_meet_24h = sum(demand_un_meet)
    not_meet_demand_cost = demand_penalty * demand_un_meet_24h
        demand_surplus = [1. * dund[t][ki] for t in 1:Ts]
        demand_surplus_24h = sum(demand_surplus)
    over_meet_demand_cost = demand_penalty * demand_surplus_24h
        emissions = [G["a"][g] * x[g,t][ki] * x[g,t][ki] + G["b"][g] * x[g,t][ki] + G["c"][g] * y[g,t][ki] for g in 1:Gs, t in 1:Ts]
    emission_cost = emission_price * sum(emissions)
        start_ups = [1. * ybar[g,t][ki] for g in 1:Gs, t in 1:Ts]
        start_up_times_per_G = [sum(start_ups[g,t] for t in 1:Ts) for g in 1:Gs]
    start_up_cost = sum(G["su_cost"][g] * start_up_times_per_G[g] for g in 1:Gs)
        shut_downs = [1. * yund[g,t][ki] for g in 1:Gs, t in 1:Ts]
        shut_down_times_per_G = [sum(shut_downs[g,t] for t in 1:Ts) for g in 1:Gs]
    shut_down_cost = sum(G["sd_cost"][g] * shut_down_times_per_G[g] for g in 1:Gs)
    # obj function and SENSE
    obj_function = fuel_cost + not_meet_demand_cost + over_meet_demand_cost + emission_cost + start_up_cost + shut_down_cost
    @info "obj_function $obj_function = fuel_cost $fuel_cost  + not_meet_demand_cost $not_meet_demand_cost + over_meet_demand_cost $over_meet_demand_cost + emission_cost $emission_cost + start_up_cost $start_up_cost + shut_down_cost $shut_down_cost"
end

if ki == MODELING # The formal Obj Modeling part
    MOI.set(o,MOI.ObjectiveFunction{typeof(obj_function)}(),obj_function) # DON'T revise this line!
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(o, MOI.RawOptimizerAttribute("NonConvex"), 2) # for Gurobi only
end

MOI.optimize!(o) # ******** the formal solve! ********
status = MOI.get(o,MOI.TerminationStatus())
if status != MOI.OPTIMAL
    @error "In UC problem, TerminationStatus = $status"
else
    maxvio_grb::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio")) # 0.00031609245227626204, if do not adjust ENV_PWL_precision
end

if true # view results after the formal optimize! (but SHOULD update objterms first)
    @info "" start_ups
    @info "" shut_downs
    power_output_figure = [x[g,t][ki] for g in 1:Gs, t in 1:Ts]
    @info "" power_output_figure
    @info "" demand_un_meet
    @info "" demand_un_meet_24h
    @info "" demand_surplus
    @info "" demand_surplus_24h
    @info "" emissions
    @info "obj_function $obj_function = fuel_cost $fuel_cost  + not_meet_demand_cost $not_meet_demand_cost + over_meet_demand_cost $over_meet_demand_cost + emission_cost $emission_cost + start_up_cost $start_up_cost + shut_down_cost $shut_down_cost"
end


# ------------------------------------------
# const Nonlinear = MOI.Nonlinear
# nlpor = Nonlinear.Model()
# for g in 1:Gs, t in 1:Ts # Nonlinear Constraint (27), with equality, which is stable already (do not model as <=, like that in the objterms)
#     # (For the sake of reference) G["v_a"][g] * x_[g,t] ^ 2 + G["v_b"][g] * x_[g,t] + G["v_c"][g] * y_[g,t] + G["v_d"][g] * abs(sin( G["v_e"][g] * (G["cgund"][g] - x_[g,t]) ))
#     Nonlinear.add_constraint(nlpor,:( $(G["v_a"][g]) * $(x[g,t]) * $(x[g,t]) + $(G["v_b"][g]) * $(x[g,t]) + $(G["v_c"][g]) * $(y[g,t]) + $(G["v_d"][g]) * abs(sin( $(G["v_e"][g]) * ($(G["cgund"][g]) - $(x[g,t])) )) - $(cgtf[g,t]) ),MOI.EqualTo(0.)) # or MOI.LessThan(0.) ?
# end
# evaluator = Nonlinear.Evaluator(nlpor,Nonlinear.ExprGraphOnly(),MOI.VariableIndex[]) # Only the 1st arg matters. the 2nd parameter appears as a hint, the last parameter is a place-holder
# MOI.set(o,MOI.NLPBlock(),MOI.NLPBlockData(evaluator)) # register this Nonlinear portion to the main Optimizer named "o"
# obj surrogates
# ------------------------------------------

import MathOptInterface as MOI
import SCIP
# import Gurobi

# 2023.10.29

###   Data Region
G_keys = ["comm_ini","gen_ini","cgbar","cgund","pgf","om_cost","su_cost","sd_cost","rgbar","rgund","a","b","c"] # om_cost not used temporarily (and reset to zeros)
include(joinpath(pwd(),"data","uc_5_2.jl")) # G_preprocess_dir
#---------------------------- Optional: View the Data ----------------------------
# import DataFrames
# dfG = DataFrames.DataFrame(G)
#---------------------------- Optional: View the Data ----------------------------
demand = [4.27 4.01] # length(demand) = Ts
emission_price = 25. # in the obj, 4th term
demand_penalty = 500. # in the obj, last term

###   Modeling Region
const Gs,Ts = length(G["gen_ini"]),length(demand)
# o = Gurobi.Optimizer()
o = SCIP.Optimizer()
# add variables
x = similar(trues(Gs,Ts),MOI.VariableIndex)
y,ybar,yund = similar(x),similar(x),similar(x)
    for i in 1:Gs, j in 1:Ts
        x[i,j] = MOI.add_variable(o)
    end
    for i in 1:Gs, j in 1:Ts
        y[i,j] = MOI.add_variable(o)
    end
    for i in 1:Gs, j in 1:Ts
        ybar[i,j] = MOI.add_variable(o)
    end
    for i in 1:Gs, j in 1:Ts
        yund[i,j] = MOI.add_variable(o)
    end
dbar = similar(trues(Ts),MOI.VariableIndex) # deficit (the part that generation not meet demand)
dund = similar(dbar) # surplus
    for j in 1:Ts
        dbar[j] = MOI.add_variable(o)
    end
    for j in 1:Ts
        dund[j] = MOI.add_variable(o)
    end
# obj surrogates
fuel_cost = MOI.add_variable(o)
MOI.add_constraint(o,sum(G["pgf"][g] * x[g,t] for g in 1:Gs, t in 1:Ts) - fuel_cost, MOI.LessThan(0.)) # cf. (26), the linear fuel cost
not_meet_demand_cost = MOI.add_variable(o)
MOI.add_constraint(o,demand_penalty * ones(Ts)' * dbar - not_meet_demand_cost, MOI.LessThan(0.))
over_meet_demand_cost = MOI.add_variable(o)
MOI.add_constraint(o,demand_penalty * ones(Ts)' * dund - over_meet_demand_cost, MOI.LessThan(0.))
emission_cost = MOI.add_variable(o)
MOI.add_constraint(o,emission_price * sum(G["a"][g] * x[g,t] * x[g,t] + G["b"][g] * x[g,t] + G["c"][g] * y[g,t] for g in 1:Gs, t in 1:Ts) - emission_cost, MOI.LessThan(0.)) # this is a quadratic (non-convex)
start_up_cost = MOI.add_variable(o)
MOI.add_constraint(o,sum(G["su_cost"][g] * ybar[g,t] for g in 1:Gs, t in 1:Ts) - start_up_cost, MOI.LessThan(0.))
shut_down_cost = MOI.add_variable(o)
MOI.add_constraint(o,sum(G["sd_cost"][g] * yund[g,t] for g in 1:Gs, t in 1:Ts) - shut_down_cost, MOI.LessThan(0.))
# obj function and SENSE
obj_function = ones(6)' * [fuel_cost, not_meet_demand_cost, over_meet_demand_cost, emission_cost, start_up_cost, shut_down_cost]
MOI.set(o,MOI.ObjectiveFunction{typeof(obj_function)}(),obj_function) # DON'T revise this line!
MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
# constraints
for t in 1:Ts # 30
    MOI.add_constraint(o,ones(Gs)' * x[:,t] + dbar[t] - dund[t],MOI.EqualTo(demand[t]))
end
for g in 1:Gs, t in 1:Ts # 31
    MOI.add_constraint(o, x[g,t] - G["cgbar"][g] * y[g,t], MOI.LessThan(0.))
end
for g in 1:Gs, t in 1:Ts # 32
    MOI.add_constraint(o, x[g,t] - G["cgund"][g] * y[g,t], MOI.GreaterThan(0.))
end
for t in 1:1, g in 1:Gs # 33_1
    MOI.add_constraint(o, 1. * x[g,t], MOI.LessThan( G["gen_ini"][g] + G["cgund"][g] + ( G["rgbar"][g] - G["cgund"][g] ) * G["comm_ini"][g] ))
end
for t in 2:Ts, g in 1:Gs # 33_2
    f = -1. * x[g,t-1] + 1. * x[g,t] + (G["cgund"][g] - G["rgbar"][g]) * y[g,t-1]
    MOI.add_constraint(o, f, MOI.LessThan(G["cgund"][g]))
end
for t in 1:1, g in 1:Gs # 34_1
    f = -1. * x[g,t] + (G["cgund"][g] - G["rgund"][g]) * y[g,t]
    MOI.add_constraint(o, f, MOI.LessThan(G["cgund"][g] - G["gen_ini"][g]))
end
for t in 2:Ts, g in 1:Gs # 34_2
    f = 1. * x[g,t-1] - 1. * x[g,t] + (G["cgund"][g] - G["rgund"][g]) * y[g,t]
    MOI.add_constraint(o, f, MOI.LessThan(G["cgund"][g]))
end
for t in 1:1, g in 1:Gs # 35_1
    MOI.add_constraint(o, 1. * ybar[g,t] - 1. * y[g,t], MOI.GreaterThan(-1. * G["comm_ini"][g]))
end
for t in 2:Ts, g in 1:Gs # 35_2
    MOI.add_constraint(o, 1. * ybar[g,t] - 1. * y[g,t] + 1. * y[g,t-1], MOI.GreaterThan(0.))
end
for t in 1:1, g in 1:Gs # 36_1
    MOI.add_constraint(o, 1. * yund[g,t] + 1. * y[g,t], MOI.GreaterThan(G["comm_ini"][g]))
end
for t in 2:Ts, g in 1:Gs # 36_2
    MOI.add_constraint(o, 1. * yund[g,t] + 1. * y[g,t] - 1. * y[g,t-1], MOI.GreaterThan(0.))
end
MOI.add_constraint.(o,x,MOI.GreaterThan(0.)) # 37
(MOI.add_constraint.(o,dbar,MOI.GreaterThan(0.)); MOI.add_constraint.(o,dund,MOI.GreaterThan(0.))) # 38
(MOI.add_constraint.(o,y,MOI.ZeroOne()); MOI.add_constraint.(o,ybar,MOI.ZeroOne()); MOI.add_constraint.(o,yund,MOI.ZeroOne())) # 39

# MOI.set(o, MOI.RawOptimizerAttribute("NonConvex"), 2) # for Gurobi only
MOI.optimize!(o)
status = MOI.get(o,MOI.TerminationStatus())
if status != MOI.OPTIMAL
    @error "In UC problem, TerminationStatus = $status"
else
    # maxvio::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio"))
    # println("Gurobi TerminationStatus Optimal with MaxVio = $maxvio")
    objective_value = MOI.get(o,MOI.ObjectiveValue())
    @info "" objective_value
    objective_constitution = MOI.get.(o,MOI.VariablePrimal(),[fuel_cost, not_meet_demand_cost, over_meet_demand_cost, emission_cost, start_up_cost, shut_down_cost])
    @info "[fuel_cost, not_meet_demand_cost, over_meet_demand_cost, emission_cost, start_up_cost, shut_down_cost]" objective_constitution'
    
    x_ = MOI.get.(o,MOI.VariablePrimal(),x)
    y_ = MOI.get.(o,MOI.VariablePrimal(),y)
    ybar_ = MOI.get.(o,MOI.VariablePrimal(),ybar)
    yund_ = MOI.get.(o,MOI.VariablePrimal(),yund)
    dbar_ = MOI.get.(o,MOI.VariablePrimal(),dbar)
    dund_ = MOI.get.(o,MOI.VariablePrimal(),dund)

    fuel_cost_table = [G["pgf"][g] * x_[g,t] for g in 1:Gs, t in 1:Ts]
    @info "" fuel_cost_table

    not_meet_demand_cost_table = demand_penalty * ones(Ts) .* dbar_
    @info "" not_meet_demand_cost_table'

    over_meet_demand_cost_table = demand_penalty * ones(Ts) .* dund_
    @info "" over_meet_demand_cost_table'

    emission_cost_table = emission_price * [G["a"][g] * x_[g,t] * x_[g,t] + G["b"][g] * x_[g,t] + G["c"][g] * y_[g,t] for g in 1:Gs, t in 1:Ts]
    @info "" emission_cost_table

    start_up_cost_table = [G["su_cost"][g] * ybar_[g,t] for g in 1:Gs, t in 1:Ts]
    @info "" start_up_cost_table
    
    shut_down_cost_table = [G["sd_cost"][g] * yund_[g,t] for g in 1:Gs, t in 1:Ts]
    @info "" shut_down_cost_table
end




















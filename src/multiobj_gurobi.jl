import MathOptInterface as MOI
import Gurobi
o = Gurobi.Optimizer()
x = MOI.add_variable(o)
y = MOI.add_variable(o)
z = MOI.add_variable(o)
w = MOI.add_variable(o)
MOI.add_constraint(o, 1. * x + y + z + w, MOI.GreaterThan(1.))
MOI.add_constraint(o, 1. * x + y + z + w, MOI.LessThan(8.))
MOI.add_constraint(o, w, MOI.GreaterThan(0.))
MOI.add_constraint(o, z, MOI.GreaterThan(0.))
MOI.add_constraint(o, x, MOI.GreaterThan(0.))
MOI.add_constraint(o, y, MOI.GreaterThan(0.))

# terms = [
#     MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1., x)),
#     MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1., y)),
# ];
# f = MOI.VectorAffineFunction(terms, [0., 0.])


obj_fun_ind = 0

obj_fun_ind += 1
MOI.set(o, Gurobi.MultiObjectiveFunction(obj_fun_ind), 1. * x + y + z + w)
MOI.set(o, Gurobi.MultiObjectivePriority(obj_fun_ind), 4) # large number => higher priority

obj_fun_ind += 1
MOI.set(o, Gurobi.MultiObjectiveFunction(obj_fun_ind), -1. * y - z - w)
MOI.set(o, Gurobi.MultiObjectivePriority(obj_fun_ind), 3)

obj_fun_ind += 1
MOI.set(o, Gurobi.MultiObjectiveFunction(obj_fun_ind), -1. * y - z)
MOI.set(o, Gurobi.MultiObjectivePriority(obj_fun_ind), 2)

obj_fun_ind += 1
MOI.set(o, Gurobi.MultiObjectiveFunction(obj_fun_ind), -1. * y)
MOI.set(o, Gurobi.MultiObjectivePriority(obj_fun_ind), 1)


# MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.optimize!(o)

MOI.get(o, MOI.TerminationStatus())

MOI.get(o, MOI.VariablePrimal(), x)
MOI.get(o, MOI.VariablePrimal(), y)
MOI.get(o, MOI.VariablePrimal(), z)
MOI.get(o, MOI.VariablePrimal(), w)

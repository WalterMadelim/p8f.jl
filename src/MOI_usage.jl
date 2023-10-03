import MathOptInterface as MOI
import Gurobi
c = [1.0, 2.0, 3.0]
w = [0.3, 0.5, 1.0]
C = 3.2

optimizer = Gurobi.Optimizer() # a model = a constructor()
x = MOI.add_variables(optimizer, length(c))
# a_scalar_variable = MOI.add_variable(optimizer)

# if you want to set an affine function f as the objective function
terms = MOI.ScalarAffineTerm.(c, x) # a vector
constant = 0. # === zero(Float64)
f = MOI.ScalarAffineFunction(terms, constant)
type_matters = MOI.ObjectiveFunction{typeof(f)}()
MOI.set(
    optimizer,
    type_matters,
    f
)
# Don't forget to set the sense at the same time!
type_matters = MOI.ObjectiveSense()
sense = MOI.MAX_SENSE
MOI.set(optimizer, type_matters, sense)

# add an affine f <= C constraint
terms = MOI.ScalarAffineTerm.(w, x) # i.e., those of the 1st order
constant = 0.
f = MOI.ScalarAffineFunction(terms, constant)
s = MOI.LessThan(C) # feas. set
a_constraint_index = MOI.add_constraint(
    optimizer,
    f,
    s
)
# add an variable bound constraint
# c1 = MOI.add_constraint(o,x,MOI.LessThan(3.))


type_matters = MOI.ZeroOne() # Binary Constraint
# for x_i in x
#     MOI.add_constraint(optimizer, x_i, type_matters)
# end
MOI.add_constraint.(optimizer, x, type_matters)

# MOI.set(o, MOI.Silent(), true)
MOI.optimize!(optimizer)

attrs = [
    MOI.TerminationStatus(),
    MOI.PrimalStatus(),
    MOI.DualStatus(), # NO_SOLUTION, due to an MIP
    MOI.ResultCount(),
    MOI.ObjectiveValue()
]
MOI.get.(optimizer, attrs) # get a first insight

MOI.get.(optimizer, MOI.VariablePrimal(), x)

# check LP_duals
# MOI.optimize!(o)
# MOI.get(o, MOI.TerminationStatus())
# MOI.get(o, MOI.ResultCount())
# MOI.get(o, MOI.PrimalStatus())
# MOI.get(o, MOI.ObjectiveValue())
# MOI.get(o, MOI.DualStatus())
# MOI.get(o, MOI.DualObjectiveValue())
# MOI.get(o, MOI.VariablePrimal(1),x) # 1 can be omitted, in the result_count
# MOI.get(o, MOI.ConstraintDual(1),c1) # dual associated to c1, if c1 is (actively) >=, then sgn(dual) = +, if c2 is <=, then -.
# MOI.get(o, MOI.ConstraintDual(1),c2)


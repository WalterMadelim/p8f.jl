import MathOptInterface as MOI
import Gurobi
import Printf

function solve_subproblem(x)
    o = Gurobi.Optimizer()
    x1 = MOI.add_variable(o)
    x2 = MOI.add_variable(o)
    c3 = MOI.add_constraint(o,x1,MOI.GreaterThan(0.))
    c4 = MOI.add_constraint(o,x2,MOI.GreaterThan(0.))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1., x1), MOI.ScalarAffineTerm(-2., x2)], 0.) # here 4. must be Float64 !
    s = MOI.LessThan(-2. + 3. * x[2] - x[1])
    c1 = MOI.add_constraint(o,f,s)
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(-1., x1), MOI.ScalarAffineTerm(-1., x2)], 0.)
    s = MOI.LessThan(-3. + 3. * x[2] + x[1])
    c2 = MOI.add_constraint(o,f,s)
    # Obj
    terms = [MOI.ScalarAffineTerm(2., x1),MOI.ScalarAffineTerm(3., x2)] # a vector
    constant = 0.
    f = MOI.ScalarAffineFunction(terms, constant)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f) # and then ObjSense
    type_matters = MOI.ObjectiveSense()
    sense = MOI.MIN_SENSE
    MOI.set(o, type_matters, sense)
    # solve
    MOI.set(o, MOI.Silent(), true)
    MOI.optimize!(o)
    @assert MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    return (obj = MOI.get(o, MOI.ObjectiveValue()), y = MOI.get.(o,MOI.VariablePrimal(1),[x1,x2]), π = MOI.get.(o, MOI.ConstraintDual(1),[c1,c2]))
end

function print_iteration(k, args...)
    f(x) = Printf.@sprintf("%12.4e", x)
    println(lpad(k, 9), " ", join(f.(args), " "))
    return
end

MAXIMUM_ITERATIONS = 100
ABSOLUTE_OPTIMALITY_GAP = 1e-6
c_1 = [1., 4.]; A_1 = [1 -3; -1 -3];

o = Gurobi.Optimizer()
x1 = MOI.add_variable(o)
x2 = MOI.add_variable(o)
theta = MOI.add_variable(o)
c1 = MOI.add_constraint(o,x1,MOI.Integer())
c2 = MOI.add_constraint(o,x2,MOI.Integer())
c3 = MOI.add_constraint(o,x1,MOI.GreaterThan(0.))
c4 = MOI.add_constraint(o,x2,MOI.GreaterThan(0.))
c_lb = MOI.add_constraint(o,theta,MOI.GreaterThan(-1e3))
# Obj
terms = [MOI.ScalarAffineTerm(1., x1),MOI.ScalarAffineTerm(4., x2),MOI.ScalarAffineTerm(1., theta)] # a vector
constant = 0. # === zero(Float64)
f = MOI.ScalarAffineFunction(terms, constant)
type_matters = MOI.ObjectiveFunction{typeof(f)}()
MOI.set(o,type_matters,f) # and then ObjSense
type_matters = MOI.ObjectiveSense()
sense = MOI.MIN_SENSE
MOI.set(o, type_matters, sense)
# solve
MOI.set(o, MOI.Silent(), true)
println("Iteration  Lower Bound  Upper Bound          Gap")
for k in 1:MAXIMUM_ITERATIONS
    MOI.optimize!(o)
    @assert MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    lower_bound = MOI.get(o, MOI.ObjectiveValue())
    x_k = MOI.get.(o,MOI.VariablePrimal(1),[x1,x2])
    ret = solve_subproblem(x_k)
    upper_bound = c_1' * x_k + ret.obj
    gap = (upper_bound - lower_bound) / upper_bound
    print_iteration(k, lower_bound, upper_bound, gap)
    if gap < ABSOLUTE_OPTIMALITY_GAP
        println("Terminating with the optimal solution")
        break
    end
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(ret.π[1] - ret.π[2], x1), MOI.ScalarAffineTerm(-3. * ret.π[1] - 3. * ret.π[2], x2), MOI.ScalarAffineTerm(1.,theta)], 0.)
    s = MOI.GreaterThan(ret.obj + ret.π' * A_1 * x_k)
    cut = MOI.add_constraint(o,f,s) # really add to o a new cut, the name `cut` is re-bounded during each iteration
    @info "Adding cut $(cut.value)"
end
print(o)
MOI.optimize!(o)
x_star = MOI.get.(o,MOI.VariablePrimal(1),[x1,x2])
optimal_ret = solve_subproblem(x_star)
y_optimal = optimal_ret.y

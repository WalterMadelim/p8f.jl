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

function my_callback(cb_data)
    # debugging (means add @bp) here is not available, i.e., @bp will never be hit.
    global k += 1 # but k indeed updates 
    x_k = [MOI.get(o,MOI.CallbackVariablePrimal(cb_data),x1),MOI.get(o,MOI.CallbackVariablePrimal(cb_data),x2)] # 164
    θ_k = MOI.get(o,MOI.CallbackVariablePrimal(cb_data),theta) # 164
    lower_bound = c_1' * x_k + θ_k # calculation of Float64 
    ret = solve_subproblem(x_k) # solve the 2nd stage problem
    upper_bound = c_1' * x_k + c_2' * ret.y
    gap = (upper_bound - lower_bound) / upper_bound
    print_iteration(k, lower_bound, upper_bound, gap)
    if gap < ABSOLUTE_OPTIMALITY_GAP
        println("Terminating with the optimal solution, getting out of function my_callback(cb_data).")
        return nothing
    end
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(ret.π[1] - ret.π[2], x1), MOI.ScalarAffineTerm(-3. * ret.π[1] - 3. * ret.π[2], x2), MOI.ScalarAffineTerm(1.,theta)], 0.)
    s = MOI.GreaterThan(ret.obj + ret.π' * A_1 * x_k)
    MOI.submit(o,MOI.LazyConstraint(cb_data),f,s) # 192
    return nothing
end


# A_2 = [1 -2; -1 -1]
# M = -1000;
c_1 = [1, 4]
c_2 = [2, 3]
A_1 = [1 -3; -1 -3]
ABSOLUTE_OPTIMALITY_GAP = 1e-6

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

k = 0
MOI.set(o,MOI.LazyConstraintCallback(), my_callback) # 185
println("k is $k before optimize!")
MOI.optimize!(o)
println("k is $k after optimize!")
x_star = MOI.get.(o,MOI.VariablePrimal(1),[x1,x2])
optimal_ret = solve_subproblem(x_star);
y_star = optimal_ret.y

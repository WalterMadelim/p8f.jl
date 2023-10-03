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

function my_GRB_cb_function(cb_data, cb_where::Cint)
    push!(cb_registry, cb_where) # cb_where can be any possible Cint 0-9
    if cb_where == Gurobi.GRB_CB_MIPSOL # whenever a new MIP solution is found (cb_where = 4)
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        # - - - - - - - - - - - manipulations here - - - - - - - - - - - 
        x_k = [MOI.get(o,MOI.CallbackVariablePrimal(cb_data),x1),MOI.get(o,MOI.CallbackVariablePrimal(cb_data),x2)] # 164
        θ_k = MOI.get(o,MOI.CallbackVariablePrimal(cb_data),theta)
        lower_bound = c_1' * x_k + θ_k # calculation of Float64
        ret = solve_subproblem(x_k) # solve the 2nd stage problem
        upper_bound = c_1' * x_k + c_2' * ret.y
        gap = (upper_bound - lower_bound) / upper_bound
        # println("######### k _ lowerbound _ upper_bound , gap #############")
        # print_iteration(k, lower_bound, upper_bound, gap)
        coeff_x1 = ret.π[1] - ret.π[2]
        coeff_x2 = -3. * ret.π[1] - 3. * ret.π[2]
        coeff_theta = 1.
        rhs = ret.obj + ret.π' * A_1 * x_k
        # println("##########################################")
        # print_iteration(x_k[1],x_k[2],θ_k,coeff_x1 * x_k[1] + coeff_x2 * x_k[2] + coeff_theta * θ_k)
        # print_iteration(coeff_x1,coeff_x2,coeff_theta,rhs)
        violation = coeff_x1 * x_k[1] + coeff_x2 * x_k[2] + coeff_theta * θ_k < rhs - MIP_CONSTR_TOL
        if violation
            f = MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(coeff_x1, x1),
            MOI.ScalarAffineTerm(coeff_x2, x2),
            MOI.ScalarAffineTerm(coeff_theta,theta)
            ],0.)
            s = MOI.GreaterThan(rhs)
            MOI.submit(o,MOI.LazyConstraint(cb_data),f,s) # 192
        end
        # - - - - - - - - - - - manipulations here - - - - - - - - - - - 
    end
    return nothing
end

cb_registry = Cint[]
MIP_CONSTR_TOL = 1e-6
MAXIMUM_ITERATIONS = 100
ABSOLUTE_OPTIMALITY_GAP = 1e-6
c_1 = [1., 4.]; A_1 = [1 -3; -1 -3];
c_2 = [2, 3.];

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

# println("Iteration  Lower Bound  Upper Bound          Gap")


k = 0
println("k is $k before optimize!")
# MOI.set(o, MOI.Silent(), true)
MOI.set(o, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
MOI.set(o, Gurobi.CallbackFunction(), my_GRB_cb_function)

MOI.optimize!(o)
# cb_registry
MOI.get(o, MOI.TerminationStatus())
MOI.get(o, MOI.VariablePrimal(1),x1)
MOI.get(o, MOI.VariablePrimal(1),x2)

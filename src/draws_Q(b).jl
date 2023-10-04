import MathOptInterface as MOI
import Gurobi
import Plots
# draws a value (of 2nd stage MIP) function
# b is a 1-stage continuous variable
# Q(b) is non-convex, thus Benders-cut isn't tight
function Q(b)
    o = Gurobi.Optimizer()
    # variable vector
    x = MOI.add_variables(o,6)
    # constrs
    s = MOI.GreaterThan(0.)
    MOI.add_constraint.(o,x,s)
    s = MOI.Integer()
    MOI.add_constraint.(o,x[1:3],s)
    ctmp = [6.,5,-4,2,-7,1]
    terms = MOI.ScalarAffineTerm.(ctmp,x)
    f = MOI.ScalarAffineFunction(terms, 0.)
    cg = MOI.add_constraint(o,f,MOI.GreaterThan(b))
    cl = MOI.add_constraint(o,f,MOI.LessThan(b))
    # obj and sense
    ctmp .= [3.,7/2,3,6,7,5]
    terms = MOI.ScalarAffineTerm.(ctmp,x)
    f = MOI.ScalarAffineFunction(terms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    type_matters = MOI.ObjectiveSense()
    MOI.set(o, type_matters, MOI.MIN_SENSE)

    # solve
    MOI.set(o,MOI.Silent(),true)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end

l = -10.
r = 10.
d = .1
x = collect(l:d:r)
y = Q.(x)
phi_star, index = findmin(y)
b_star = x[index]
println("optval:", phi_star, "opt_b:", b_star)
Plots.plot(x,y)

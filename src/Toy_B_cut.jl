import MathOptInterface as MOI
import Gurobi

# Toy example at "The Benders Dual Decomposition Method"
# pure Continuous version (pure B cut)

# Ite 1 ----- -7.0 < 10.5 -----
# Ite 2 ----- -1.75 < 2.75 -----
# Ite 3 ----- 1.625 < 2.9166666666666665 -----
# Ite 4 ----- 2.4 < 2.4 -----

# Minimize ScalarAffineFunction{Float64}:
#  0.0 + 1.0 θ

# Subject to:

# VariableIndex-in-GreaterThan{Float64}
#  y >= 0.0

# ScalarAffineFunction{Float64}-in-GreaterThan{Float64}
#  0.0 + 15.0 y + 1.0 θ >= 8.0
#  0.0 - 35.0 y + 1.0 θ >= -24.5
#  0.0 - 5.0 y + 1.0 θ >= -0.5
#  0.0 + 3.333333333333333 y + 1.0 θ >= 4.333333333333333

# VariableIndex-in-LessThan{Float64}
#  y <= 1.0

# **********************************************************

# Plot code
# y1(x) = 8.0 - 15x
# y2(x) = -24.5 + 35x
# y3(x) = -0.5 + 5x
# y4(x) = 4.333333333333333 -  3.333333333333333 * x

# x = range(0., 1., length=100)
# Plots.plot(x, [y1.(x),y2.(x),y3.(x),y4.(x)])
# Plots.ylims!(0., 8.)


function Q(y_bar)
    o = Gurobi.Optimizer()
    MOI.set(o,MOI.Silent(),true)
    x = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),x,"x")
    z = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),z,"z")
    terms = [MOI.ScalarAffineTerm(1.,x),MOI.ScalarAffineTerm(15.,z)]
    f = MOI.ScalarAffineFunction(terms, 0.)
    s = MOI.GreaterThan(8.)
    MOI.add_constraint(o,f,s)
    terms = [MOI.ScalarAffineTerm(3.,x),MOI.ScalarAffineTerm(10.,z)]
    f = MOI.ScalarAffineFunction(terms, 0.)
    s = MOI.GreaterThan(13.)
    MOI.add_constraint(o,f,s)
    terms = [MOI.ScalarAffineTerm(1.,x),MOI.ScalarAffineTerm(10.,z)]
    f = MOI.ScalarAffineFunction(terms, 0.)
    s = MOI.GreaterThan(7.)
    MOI.add_constraint(o,f,s)
    terms = [MOI.ScalarAffineTerm(2.,x),MOI.ScalarAffineTerm(-10.,z)]
    f = MOI.ScalarAffineFunction(terms, 0.)
    s = MOI.GreaterThan(-1.)
    MOI.add_constraint(o,f,s)
    terms = [MOI.ScalarAffineTerm(2.,x),MOI.ScalarAffineTerm(-70.,z)]
    f = MOI.ScalarAffineFunction(terms, 0.)
    s = MOI.GreaterThan(-49.)
    MOI.add_constraint(o,f,s)

    # copy constr
    s = MOI.EqualTo(y_bar)
    cpc = MOI.add_constraint(o,z,s)

    # obj function and SENSE
    objterms = [MOI.ScalarAffineTerm(1.,x)]
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    type_matters = MOI.ObjectiveSense()
    MOI.set(o, type_matters, MOI.MIN_SENSE)
    
    MOI.optimize!(o)
    attrs = [
        MOI.TerminationStatus(),
        MOI.PrimalStatus(),
        MOI.DualStatus(), # NO_SOLUTION, due to an MIP
        MOI.ResultCount(),
        MOI.ObjectiveValue()
    ]
    attrs = MOI.get.(o, attrs)
    @assert attrs[1] == MOI.OPTIMAL
    obj = MOI.get(o, MOI.ObjectiveValue()) # actually the objval!
    lambda = MOI.get(o, MOI.ConstraintDual(), cpc)
    return (s=lambda,c=obj-lambda*y_bar,o=obj) # slope, const; 2nd_stage_obj
end

function model_init()
    # Master problem
    o = Gurobi.Optimizer()
    # Silent
    MOI.set(o,MOI.Silent(),true)
    # variables, auxiliary variables
    y = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),y,"y")
    theta = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),theta,"θ")
    # pure 1st-stage <= constrs
    s = MOI.GreaterThan(0.)
    MOI.add_constraint(o,y,s)
    s = MOI.LessThan(1.)
    MOI.add_constraint(o,y,s)
    # obj function and SENSE
    objterms = [MOI.ScalarAffineTerm(1.,theta)]
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    type_matters = MOI.ObjectiveSense()
    MOI.set(o, type_matters, MOI.MIN_SENSE)
    return o,y,theta
end

IteCnt = 0
o,y,theta = model_init()

# 2nd-stage cuts to avoid initial unboundedness
initial_trial = 0.
ret = Q(initial_trial)
terms = [MOI.ScalarAffineTerm(1.,theta),MOI.ScalarAffineTerm(-ret.s,y)]
f = MOI.ScalarAffineFunction(terms, 0.)
s = MOI.GreaterThan(ret.c)
MOI.add_constraint(o,f,s) # the initial B cut: theta + 15y >= 8

# optimize! and OPT check
MOI.optimize!(o)
attrs = [
    MOI.TerminationStatus(),
    MOI.PrimalStatus(),
    MOI.DualStatus(), # NO_SOLUTION, due to an MIP
    MOI.ResultCount(),
    MOI.ObjectiveValue()
]
attrs = MOI.get.(o, attrs)
@assert attrs[1] == MOI.OPTIMAL

# Get solution
lb = MOI.get(o, MOI.ObjectiveValue())
y_k = MOI.get(o, MOI.VariablePrimal(), y)
ret = Q(y_k)
ub = 0. + ret.o # 1st-cost (without theta) and 2nd-cost
IteCnt += 1
println("Ite $IteCnt ----- $lb < $ub -----")

for mainloopcount in 1:100
    if lb < ub - 1e-6
        # add 2nd stage cut
        terms = [MOI.ScalarAffineTerm(1.,theta),MOI.ScalarAffineTerm(-ret.s,y)]
        f = MOI.ScalarAffineFunction(terms, 0.)
        s = MOI.GreaterThan(ret.c)
        MOI.add_constraint(o,f,s)
        # optimize! and OPT check
        MOI.optimize!(o)
        attrs = [
            MOI.TerminationStatus(),
            MOI.PrimalStatus(),
            MOI.DualStatus(), # NO_SOLUTION, due to an MIP
            MOI.ResultCount(),
            MOI.ObjectiveValue()
        ]
        attrs = MOI.get.(o, attrs)
        @assert attrs[1] == MOI.OPTIMAL
        # Get solution
        lb = MOI.get(o, MOI.ObjectiveValue())
        y_k = MOI.get(o, MOI.VariablePrimal(), y)
        ret = Q(y_k)
        ub = 0. + ret.o # 1st-cost (without theta) and 2nd-cost
        IteCnt += 1
        println("Ite $IteCnt ----- $lb < $ub -----")
    elseif abs(ub-lb) <= 1e-6
        println("----------- CONVERGE :  abs(ub-lb) <= 1e-6 -----------")
        break
    else
        error(" In Main LOOP! ")
    end
end



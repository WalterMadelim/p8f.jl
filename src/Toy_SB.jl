import MathOptInterface as MOI
import Gurobi

# Toy example at "The Benders Dual Decomposition Method"
#  (1st stage y_k::Int ? B cut : SB cut)

# Ite 1 ----- -7.0 < 10.5 -----
# Ite 2 ----- -1.75 < 2.75 -----
# Ite 3 ----- 6.125 < 6.125 -----

# Minimize ScalarAffineFunction{Float64}:
#  0.0 + 1.0 v[2]

# Subject to:

# VariableIndex-in-GreaterThan{Float64}
#  v[1] >= 0.0

# ScalarAffineFunction{Float64}-in-GreaterThan{Float64}   
#  0.0 + 15.0 v[1] + 1.0 v[2] >= 8.0
#  0.0 - 35.0 v[1] + 1.0 v[2] >= -24.5
#  0.0 - 5.0 v[1] + 1.0 v[2] >= 5.5

# VariableIndex-in-LessThan{Float64}
#  v[1] <= 1.0

function is_int(x, tolerance=1e-6)
    return abs(x - round(x)) <= tolerance
end

function Q(y_bar)
    o = Gurobi.Optimizer()
    x = MOI.add_variable(o)
    z = MOI.add_variable(o)

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

    # s = MOI.LessThan(1.)
    # MOI.add_constraint(o,z,s)
    # s = MOI.GreaterThan(0.)
    # MOI.add_constraint(o,z,s)

    s = MOI.EqualTo(y_bar)
    cpc = MOI.add_constraint(o,z,s)

    # obj function and SENSE
    objterms = [MOI.ScalarAffineTerm(1.,x)]
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    type_matters = MOI.ObjectiveSense()
    MOI.set(o, type_matters, MOI.MIN_SENSE)
    MOI.set(o,MOI.Silent(),true)
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
    x_k = MOI.get(o, MOI.VariablePrimal(), x) # actually the objval!
    z_k = MOI.get(o, MOI.VariablePrimal(), z)
    lambda = MOI.get(o, MOI.ConstraintDual(), cpc)
    return (s=lambda,c=x_k-lambda*z_k,o=x_k) # slope, const; 2nd_stage_obj
end

function Q_SB(y_bar)
    ret = Q(y_bar)
    lambda = ret.s # slope of SB cut is the same as B cut
    # Model and variables
    o = Gurobi.Optimizer()
    x = MOI.add_variable(o)
    z = MOI.add_variable(o)
    # constrs
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
    # --------------- Relaxing This --------------- 
    # s = MOI.EqualTo(y_bar)
    # cpc = MOI.add_constraint(o,z,s)
    # --------------- Relaxing This --------------- 
    # --------------- Adding This --------------- 
        # here is the pure 1_stage constrs
    s = MOI.LessThan(1.)
    MOI.add_constraint(o,z,s)
    s = MOI.GreaterThan(0.)
    MOI.add_constraint(o,z,s)
        # And Int_constr of 1_stage
    MOI.add_constraint(o,z,MOI.Integer())
    # --------------- Adding This --------------- 
    # --------------- Modifying This --------------- 
    # objterms = [MOI.ScalarAffineTerm(1.,x)]
    # f = MOI.ScalarAffineFunction(objterms, 0.)
    objterms = [MOI.ScalarAffineTerm(1.,x), MOI.ScalarAffineTerm(-lambda,z)]
    f = MOI.ScalarAffineFunction(objterms, lambda * y_bar)
    # --------------- Modifying This --------------- 
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    type_matters = MOI.ObjectiveSense()
    MOI.set(o, type_matters, MOI.MIN_SENSE)
    MOI.set(o,MOI.Silent(),true)
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
    x_k = MOI.get(o, MOI.VariablePrimal(), x) # unpenalized obj
    z_k = MOI.get(o, MOI.VariablePrimal(), z)
    return (s=lambda, c=x_k-lambda*z_k, o=ret.o) # const
end


IteCnt = 0
# Master problem
o = Gurobi.Optimizer()
MOI.set(o,MOI.Silent(),true)
y = MOI.add_variable(o)
theta = MOI.add_variable(o)
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
# 2nd-stage cuts 
ret = Q(0.) 
terms = [MOI.ScalarAffineTerm(1.,theta),MOI.ScalarAffineTerm(-ret.s,y)]
f = MOI.ScalarAffineFunction(terms, 0.)
s = MOI.GreaterThan(ret.c)
MOI.add_constraint(o,f,s) # the cheat B cut: theta >= 8 - 15y
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
lb = MOI.get(o, MOI.VariablePrimal(), theta)
y_k = MOI.get(o, MOI.VariablePrimal(), y)
int_flag = is_int(y_k)
println("Get an Integer? == $int_flag")
ret = ifelse(int_flag,Q(y_k),Q_SB(y_k))
ub = ret.o
IteCnt += 1
println("Ite $IteCnt ----- $lb < $ub -----")
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
lb = MOI.get(o, MOI.VariablePrimal(), theta)
y_k = MOI.get(o, MOI.VariablePrimal(), y)
int_flag = is_int(y_k)
println("Get an Integer? == $int_flag")
ret = ifelse(int_flag,Q(y_k),Q_SB(y_k))
ub = ret.o
IteCnt += 1
println("Ite $IteCnt ----- $lb < $ub -----")
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
lb = MOI.get(o, MOI.VariablePrimal(), theta)

y_k = MOI.get(o, MOI.VariablePrimal(), y)
int_flag = is_int(y_k)
println("Get an Integer? == $int_flag")
ret = ifelse(int_flag,Q(y_k),Q_SB(y_k))
ub = ret.o
IteCnt += 1
println("Ite $IteCnt ----- $lb < $ub -----") # close the gap


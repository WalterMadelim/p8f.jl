import MathOptInterface as MOI
import Gurobi

function is_int(x, tolerance=1e-6)
    return abs(x - round(x)) <= tolerance
end

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
    
    MOI.add_constraint(o,x,MOI.GreaterThan(0.))

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

function Q_SB(y_bar)
    ret = Q(y_bar)
    lambda = ret.s # slope of SB cut is the same as B cut
    # Model and variables
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
    MOI.add_constraint(o,x,MOI.GreaterThan(0.))
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
    x_k = MOI.get(o, MOI.VariablePrimal(), x)
    z_k = MOI.get(o, MOI.VariablePrimal(), z)
    return (s=lambda, c=x_k-lambda*z_k, o=ret.o) # const see (10), slope/obj is directly from Q
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

function is_trial_repeat(y=y_k,t=trials,c=TrialCnt,d=TrialDistLimit)
    for r in c:-1:1
        if abs(y - t[r]) <= d
            return true
        end
    end
    return false
end


TrialDistLimit = 1e-3
MaxIteNum = 8
trials = similar(fill(missing,MaxIteNum+2),Union{Missing,Float64})
TrialCnt = 0

o,y,theta = model_init()

# 2nd-stage cuts to avoid initial unboundedness
TrialCnt += 1
trials[TrialCnt] = 0. # <<<<---- Initial Trail Point that you specify 
ret = Q(trials[TrialCnt])
ub = 0. + ret.o # Initial Upper Bound
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
@assert abs(y_k-trials[TrialCnt]) > TrialDistLimit
TrialCnt += 1
trials[TrialCnt] = y_k
# check Integrality of 1st stage solution
int_flag = is_int(y_k)
println("Get an Integer? == $int_flag")
ret = ifelse(int_flag,Q(y_k),Q_SB(y_k))
new_ub = 0. + ret.o
ub = ifelse(new_ub<ub,new_ub,ub)

IteCnt = 0
IteCnt += 1
println("Ite $IteCnt ----- $lb < $ub -----")


for mainloopcount in 1:MaxIteNum
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
        # check if this trial point is new
        if is_trial_repeat(y_k,trials,TrialCnt,TrialDistLimit)
            println("----------- TrialPoint Repeat ($y_k)-----------")
            break
        end
        # add this new trial point to list
        TrialCnt += 1
        trials[TrialCnt] = y_k
        # check Integrality of 1st stage solution
        int_flag = is_int(y_k)
        println("Get an Integer? == $int_flag")
        ret = ifelse(int_flag,Q(y_k),Q_SB(y_k))
        if int_flag
            new_ub = 0. + ret.o
            ub = ifelse(new_ub<ub,new_ub,ub)
        end
        IteCnt += 1
        println("Ite $IteCnt ----- $lb < $ub -----")
    elseif abs(ub-lb) <= 1e-6
        println("----------- CONVERGE :  abs(ub-lb) <= 1e-6 -----------")
        break
    else
        error(" In Main LOOP! ")
    end
end




const sites, clients, scenes = 3,4,5
const h = [1. 1 0 1 0; 0 1 0 0 1; 1 0 0 1 1; 1 1 0 1 0] # scene-wise demand per client # random variable
const c = [54., 40, 53] # cost of build a server at 3 sites
const d = [9.0 15.0 10.0; 3.0 2.0 17.0; 11.0 11.0 16.0; 10.0 1.0 4.0]
const q = [16.0 20.0 18.0; 13.0 6.0 7.0; 8.0 4.0 21.0; 21.0 10.0 7.0]
const q0 = fill(12.,sites) # penalty coefficient
const u = 15. # capacity of a established server
const p = fill(0.2,scenes)

# 2023/10/19
# 59.6

import LinearAlgebra
import MathOptInterface as MOI
import Gurobi


function silent_new_optimizer()
    o = Gurobi.Optimizer(GRB_ENV); # master
    MOI.set(o,MOI.RawOptimizerAttribute("OutputFlag"),0)
    return o
end

function terms_init(l)
    return [MOI.ScalarAffineTerm(0.,MOI.VariableIndex(0)) for _ in 1:l]
end

function Q(s,xt)::Float64 # s only affects rand variable at RHS of constr 2 in this function
    @assert 0. <= minimum(xt) && maximum(xt) <= 1. # validity of trial point
    o = silent_new_optimizer() # the s subproblem with trial point xt
    objterms = terms_init(sites + clients * sites) # if no penalty
    x = similar(xt,MOI.VariableIndex) # copy vector of xt
    for j in 1:sites
        x[j] = MOI.add_variable(o)
    end
    cpc = MOI.add_constraint.(o,x,MOI.EqualTo.(xt)) # copy constraint index vector

    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:sites
        y0[j] = MOI.add_variable(o)
    end
    objterms[1:sites] .= MOI.ScalarAffineTerm.(q0,y0)
    MOI.add_constraint.(o,y0,MOI.GreaterThan(0.)) # constr 5 (the last)

    y = similar(q,MOI.VariableIndex)
    for i in 1:clients
        terms = terms_init(sites) # constr 2
        for j in 1:sites
            y[i,j] = MOI.add_variable(o)
            objterms[sites + (i-1)sites + j] = MOI.ScalarAffineTerm(-q[i,j],y[i,j]) # NEGATIVE revenue
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.EqualTo(h[i,s])) # constr 2
    end
    MOI.add_constraint.(o,y,MOI.GreaterThan(0.)) # constr 4
    MOI.add_constraint.(o,y,MOI.LessThan(1.)) # constr 4
    # y ∈ Y
    MOI.add_constraint.(o,y,MOI.Integer())

    for j in 1:sites
        terms = terms_init(clients+2)
        for i in 1:clients
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[clients+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[clients+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.LessThan(0.)) # constr 1
    end

    # obj function and SENSE of the s subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end

const GRB_ENV = Gurobi.Env()

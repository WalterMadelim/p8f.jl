const sites, clients, scenes = 3,4,5
const h = [1. 1 0 1 0; 0 1 0 0 1; 1 0 0 1 1; 1 1 0 1 0] # scene-wise demand per client # random variable
const c = [54., 40, 53] # cost of build a server at 3 sites
const d = [9.0 15.0 10.0; 3.0 2.0 17.0; 11.0 11.0 16.0; 10.0 1.0 4.0]
const q = [16.0 20.0 18.0; 13.0 6.0 7.0; 8.0 4.0 21.0; 21.0 10.0 7.0]
const q0 = fill(12.,sites) # penalty coefficient
const u = 15. # capacity of a established server
const p = fill(0.2,scenes)

# a suitably designed sslp problem
# 2023/10/19

# Nodes    |    Current Node    |     Objective Bounds      |     Work
# Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

#     0     0   39.35763    0    8  214.20000   39.35763  81.6%     -    0s
# H    0     0                      69.4000000   39.35763  43.3%     -    0s
# H    0     0                      67.0000000   39.35763  41.3%     -    0s
#     0     0   52.36299    0   23   67.00000   52.36299  21.8%     -    0s
# H    0     0                      59.6000000   52.98337  11.1%     -    0s
#     0     0   59.60000    0   18   59.60000   59.60000  0.00%     -    0s

import LinearAlgebra
import MathOptInterface as MOI
import Gurobi

function terms_init(l)
    return [MOI.ScalarAffineTerm(0.,MOI.VariableIndex(0)) for _ in 1:l]
end

o = Gurobi.Optimizer()

objterms = terms_init( sites + scenes * (sites + clients * sites) );

# build first stage variable x
x = similar(c,MOI.VariableIndex);
for j in 1:sites
    x[j] = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),x[j],"x[$j]")
end
# 1st stage cost
objterms[1:sites] .= MOI.ScalarAffineTerm.(c,x)
# Ax ≥ b
MOI.add_constraint.(o,x,MOI.GreaterThan(0.))
MOI.add_constraint.(o,x,MOI.LessThan(1.))
# x ∈ X
MOI.add_constraint.(o,x,MOI.Integer())

# every second stage variable is a vector. vector[s] is the variable per scene
y0 = [similar(q0,MOI.VariableIndex) for _ in 1:scenes];
for s in 1:scenes
    for j in 1:sites
        y0[s][j] = MOI.add_variable(o)
        MOI.set(o,MOI.VariableName(),y0[s][j],"y0[$s][$j]")
        objterms[sites + (s-1)sites + j] = MOI.ScalarAffineTerm(p[s] * q0[j], y0[s][j]) # penalty cost
    end
    MOI.add_constraint.(o,y0[s],MOI.GreaterThan(0.)) # constr 5 (the last)
end

y = [similar(q,MOI.VariableIndex) for _ in 1:scenes]
baseInd = sites + scenes * sites
for s in 1:scenes
    for i in 1:clients
        terms = terms_init(sites) # constr 2
        for j in 1:sites
            y[s][i,j] = MOI.add_variable(o)
            MOI.set(o,MOI.VariableName(),y[s][i,j],"y[$s][$i,$j]")
            objterms[baseInd + (s-1)clients * sites + (i-1)sites + j] = MOI.ScalarAffineTerm(-p[s] * q[i,j],y[s][i,j]) # NEGATIVE revenue
            terms[j] = MOI.ScalarAffineTerm(1.,y[s][i,j]) # constr 2
        end
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.EqualTo(h[i,s])) # constr 2
    end
    MOI.add_constraint.(o,y[s],MOI.GreaterThan(0.)) # constr 4
    MOI.add_constraint.(o,y[s],MOI.LessThan(1.)) # constr 4
    # y ∈ Y
    MOI.add_constraint.(o,y[s],MOI.Integer())
end

# ******************************** Linking Constraint Here ********************************
for s in 1:scenes
    for j in 1:sites
        tmp = terms_init(clients+2)
        for i in 1:clients
            tmp[i] = MOI.ScalarAffineTerm(d[i,j],y[s][i,j])
        end
        tmp[clients+1] = MOI.ScalarAffineTerm(-1.,y0[s][j])
        tmp[clients+2] = MOI.ScalarAffineTerm(-u,x[j]) # 1st stage variable here
        MOI.add_constraint(o,MOI.ScalarAffineFunction(tmp, 0.),MOI.LessThan(0.))
    end
end

# obj function and SENSE
f = MOI.ScalarAffineFunction(objterms, 0.);
type_matters = MOI.ObjectiveFunction{typeof(f)}()
MOI.set(o,type_matters,f)
type_matters = MOI.ObjectiveSense()
MOI.set(o, type_matters, MOI.MIN_SENSE)
MOI.optimize!(o)
@assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL

xt = MOI.get.(o,MOI.VariablePrimal(),x)
allocate_cost = c' * xt

@info "Obj: $(MOI.get(o,MOI.ObjectiveValue())) = allocate_cost $(allocate_cost) + sum(p[s] * 2ndcost[s])"

y0t = [similar(c) for _ in 1:scenes]
purchase_cost = similar(p)
for s in 1:scenes
    y0t[s] .= MOI.get.(o,MOI.VariablePrimal(),y0[s])
    purchase_cost[s] = q0' * y0t[s]
end

revenue = similar(p) # The NEGATIVE revenue then add to cost
yt = [similar(q) for _ in 1:scenes]
for s in 1:scenes
    yt[s] .= MOI.get.(o,MOI.VariablePrimal(),y[s]) # Matrix
    revenue[s] = sum(q .* yt[s])
end

actual_load = [similar(c) for _ in 1:scenes]
for s in 1:scenes
    for j in 1:sites
        actual_load[s][j] = d[:,j]' * yt[s][:,j]
    end
end

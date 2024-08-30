using JuMP
import Ipopt
# Ipopt can only find a LOCAL optimal point
m = JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(m, -1. <= x <= 4., start = 0.9) # it should be different when start = 1.1
JuMP.@objective(m, Max, abs(x-1)) # abs, although not in C^2 function class from the strict math standpoint, can be handled by Ipopt
JuMP.optimize!(m)
JuMP.termination_status(m)
JuMP.value(x)
JuMP.objective_value(m)

m = JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(m, x)
JuMP.@objective(m, Min, abs(x-1)) # minimize
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.NUMERICAL_ERROR # we don't have derivative at opt point
JuMP.value(x)
JuMP.objective_value(m)


m = JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(m, x)
JuMP.@objective(m, Min, abs2(x-1)) # change to abs2 function can fix this
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED 
JuMP.value(x)
JuMP.objective_value(m)






using JuMP, AmplNLWriter, Couenne_jll
model = Model()
@variable(model,x≥0)
@variable(model,0≤y≤3)
@constraint(model,c1,6x+8y≥100)
@constraint(model,c2,7x+12y≥120)
@objective(model,Min,12x+20y+5)
set_optimizer(model, ()->AmplNLWriter.Optimizer(Couenne_jll.amplexe))
# set_silent(model)
optimize!(model)
solution_summary(model)
termination_status(model)
objective_value(model)
primal_status(model)
dual_status(model)
value(x) # value.(x) if x is a Vector

shadow_price(c1) # Only for MILP solvers.
set_lower_bound(x,1.)

delete_lower_bound(x)

set_upper_bound(y,3.)
set_integer.(x)
unset_integer.(y)
set_binary.(x)
unset_binary.(y)

# construct variables
sources = ["A", "B", "C"]
sinks = ["D", "E"]
S = [(source, sink) for source in sources, sink in sinks]
model = Model()
@variable(model, x[S]) # it's wrong, if x[i for i in 1:3] here. Just write x[1:3]

# eq ways of generate variables
the_iterable = [y for y in 1:10 if y % 3 == 0]
@variable(model,x[the_iterable])
# eq_2 the following
@variable(model,x[y in 1:10; y%3 == 0])
# variant
@variable(model,y[i in 1:17;i % 2 != 0 && i % 3 != 0])
# 2-dim
@variable(model,x[i in 1:4,j in 1:4;distances[i,j]>0])
# JuMP.Containers.SparseAxisArray{VariableRef, 2, Tuple{Int64, Int64}} with 7 entries:
#   [1, 2]  =  x[1,2]
#   [1, 3]  =  x[1,3]
#   [1, 4]  =  x[1,4]
#   [2, 3]  =  x[2,3]
#   [2, 4]  =  x[2,4]
#   [3, 4]  =  x[3,4]
#   [4, 1]  =  x[4,1]

# use eachindex; where distances isa 4-by-4 matrix
@variable(model, x[i = 1:N, j = 1:N; distances[i, j] > 0])
@objective(model, Min, sum(distances[i...] * x[i] for i in eachindex(x)))

# @constraint(model, 2x-1 ⟂ x)
# A special way to deal with this complementary-constraint
model = Model();x=y=z=w=nothing;
@variable(model,x≥0)
@constraint(model,2x-1≥0)
@constraint(model,(2x-1)x==0)
@objective(model,Max,x) # superfluous, since feasible set is a singleton
set_optimizer(model,Gurobi.Optimizer)
get_attribute(model,"NonConvex") # -1 now
optimize!(model) # error!
set_attribute(model,"NonConvex",2)
optimize!(model) # Continuous model is non-convex -- solving as a MIP; solved to Opt.

# Usage of undo = fix_discrete_variables(model), see the end at the LinearPrograms section of JuMP.

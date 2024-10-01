import LinearAlgebra
import Distributions
import Random
import Gurobi
import JuMP
using Logging

# A note on How to convert an inf-program to a feasibility system by introducing dual side variables
# 1. write primal-side feasibility
# 2. write dual-side feasibility
# 3. add strong duality by objective cut
# NB 
# 1. CS condition is an SOS1 constraint which might entail variable bounds, which is loathsome
# 2. CS condition might only be derived under programs in the ≤ constr style, which is restrictive and not user-friendly
# 1/10/2024

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function JumpModel(i)
    if i == 0 # the most frequently used
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    end
    # JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end

d = Distributions.Uniform(-5., 5.)
s = abs(rand(Int))
s = 2522323970478014153
@info "seed = $s"
Random.seed!(s)
c = rand(d, 5)
A = rand(d, 3, 5)
b = rand(d, 3)


υ = JumpModel(0) # primal problem
JuMP.@variable(υ, x[1:5])
JuMP.@constraint(υ, du[r=1:3], A[r, :]' * x == b[r])
JuMP.@objective(υ, Min, c' * x)
JuMP.@constraint(υ, x in JuMP.SecondOrderCone())
JuMP.set_attribute(υ, "QCPDual", 1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)
JuMP.objective_value(υ) 
x = JuMP.value.(x)
pai = JuMP.dual.(du)


υ = JumpModel(0) # dual problem
JuMP.@variable(υ, pai[1:3])
JuMP.@variable(υ, aux[1:5])
JuMP.@objective(υ, Max, b' * pai)
JuMP.@expression(υ, auxconstr[i = 1:5], c[i] - A[:, i]' * pai - aux[i])
JuMP.@constraint(υ, x[i = 1:5], auxconstr[i] == 0.) # CS condition should be satisfied
JuMP.@constraint(υ, aux in JuMP.SecondOrderCone())
JuMP.set_attribute(υ, "QCPDual", 1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)
JuMP.objective_value(υ) 
x = JuMP.dual.(x)
auxconstr = JuMP.value.(auxconstr)
pai = JuMP.value.(pai)

# julia> JuMP.objective_value(υ)
# -7.01213655475072

# julia> x = JuMP.dual.(x)
# 5-element Vector{Float64}:
#   50.93277379323141
#  -50.11102936101879
#   -5.749805630237112
#   -7.0465484640906455
#    0.5553904373709846

# julia> auxconstr = JuMP.value.(auxconstr)
# 5-element Vector{Float64}:
#  -1.7260859408452234e-11
#  -1.6984635919925495e-11
#  -1.9430013153964865e-12
#  -2.387312569851474e-12
#   1.8379742172669467e-13

# julia> pai = JuMP.value.(pai)
# 3-element Vector{Float64}:
#  -1.2963468540582057
#  -1.5432614798749202
#   0.8384699968319406

υ = JumpModel(0) # feas. system by objective cut
JuMP.@variable(υ, x[1:5])
JuMP.@constraint(υ, [r=1:3], A[r, :]' * x == b[r])
JuMP.@constraint(υ, x in JuMP.SecondOrderCone())
JuMP.@variable(υ, pai[1:3])
JuMP.@variable(υ, aux[1:5])
JuMP.@expression(υ, auxconstr[i = 1:5], c[i] - A[:, i]' * pai - aux[i])
JuMP.@constraint(υ, [i = 1:5], auxconstr[i] == 0.)
JuMP.@constraint(υ, aux in JuMP.SecondOrderCone())
JuMP.@constraint(υ, b' * pai >= c' * x) # objective cut to assure strong duality
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)
x = JuMP.value.(x)
pai = JuMP.value.(pai)



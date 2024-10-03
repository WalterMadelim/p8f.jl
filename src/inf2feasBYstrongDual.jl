import LinearAlgebra
import Distributions
import Random
import Gurobi
import JuMP
using Logging

# A note on the subproblem of CCG2012
# There are 3 methods
# 1. write the strong dual program of the 2nd stage problem, and then the 2 'max' merges
# In method 1, the primal variables in 2nd stage are absent
# 2. primal feas + dual feas + obj cut, leading to bilinear constraints which are hard to contend with
# 3. primal feas + dual feas + CS condition (SOS1)
# From the standpoint of programming, method 2 is easier than 3
# If the 2nd stage problem is LP, then 3 might lead to MILP, which might be easier to solve.
# If the 2nd stage problem is CQP. The CS condition is more subtle (The CS condition contain bilinear constraints)
# therefore, When the 2nd stage problem is CQP, the subproblem of CCG2012 is hard intrinsically

function ip(x, y) return LinearAlgebra.dot(x, y) end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()

function JumpModel(i)
    if i == 0 # the most frequently used
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    elseif i == 1 # generic convex conic program
        m = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 # if you need Gurobi callback
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    # JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end


d = Distributions.Uniform(-5., 5.)
s = abs(rand(Int))
s = 5747327597527341603
@info "seed = $s"
Random.seed!(s)
c = rand(d, 3)
A = rand(d, 5, 3)
b = rand(d, 5)


υ = JumpModel(0) # primal problem
JuMP.@variable(υ, x[1:3])
JuMP.@constraint(υ, du[r=1:5], ip(A[r, :], x) <= b[r])
JuMP.@objective(υ, Min, ip(c, x))
JuMP.@constraint(υ, x in JuMP.SecondOrderCone())
JuMP.set_attribute(υ, "QCPDual", 1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)

JuMP.objective_value(υ)
x = JuMP.value.(x)
pai = JuMP.dual.(du)


υ = JumpModel(0) # dual problem
JuMP.@variable(υ, pai[1:5] >= 0.)
JuMP.@variable(υ, aux[1:3])
JuMP.@objective(υ, Max, -ip(b, pai))
JuMP.@constraint(υ, [r = 1:3], aux[r] == ip(pai, A[:, r]) + c[r])
JuMP.@constraint(υ, aux in JuMP.SecondOrderCone())
JuMP.set_attribute(υ, "QCPDual", 1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)
JuMP.objective_value(υ)


υ = JumpModel(0) # feas. sys
JuMP.@variable(υ, x[1:3])
JuMP.@variable(υ, pri_aux[1:5] >= 0.) # b - Ax
JuMP.@variable(υ, pai[1:5] >= 0.)
JuMP.@variable(υ, dual_aux[1:3]) # A' * pai + c
JuMP.@constraint(υ, x in JuMP.SecondOrderCone())
JuMP.@constraint(υ, dual_aux in JuMP.SecondOrderCone())
# definition
JuMP.@constraint(υ, [r = 1:5], pri_aux[r] == b[r] - ip(A[r, :], x))
JuMP.@constraint(υ, [i = 1:3], dual_aux[i] == ip(pai, A[:, i]) + c[i])
# CS condition
JuMP.@constraint(υ, [r=1:5], [pai[r], pri_aux[r]] in JuMP.SOS1())
JuMP.@constraint(υ, ip(x, dual_aux) <= 0) # ⚠️ CS condition for CQP is not componentwise !!!!
# 2 objectives
JuMP.@expression(υ, pri_obj, ip(c, x))
JuMP.@expression(υ, dual_obj, -ip(b, pai))
# JuMP.@constraint(υ, dual_obj >= pri_obj)
# JuMP.set_attribute(υ, "FeasibilityTol", 1e-2)
# JuMP.set_attribute(υ, "IntFeasTol", 1e-1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)

JuMP.value(pri_obj)
JuMP.value(dual_obj)


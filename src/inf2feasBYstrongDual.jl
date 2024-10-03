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
# From the standpoint of programming, method 2 is easier than I
# If the 2nd stage problem is LP, then I might lead to MILP, which might be easier to solve.
# If the 2nd stage problem is CQP. The CS condition is more subtle (The CS condition contain bilinear constraints)
# therefore, When the 2nd stage problem is CQP, the subproblem of CCG2012 is hard intrinsically
# KKT system based on stong dual's feas cut might be more efficient than CS condition, when the 2nd stage problem is CQP

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

I = 7 # number of primal variables
R = 14 # number of primal constraints
d = Distributions.Uniform(-5., 5.)
s = abs(rand(Int)) # generate seed randomly to generate test cases
# s = 4897215767195788552 # this is a workable case for I = 30, R = 50
# s = 8825688010375607466 # this is a workable case for I = 30, R = 50
# s = 1872257587845040550 # this is a workable case for I = 4, R = 7
s = 6854818527793916636 # this is a workable case for I = 7, R = 14
@info "seed = $s"
Random.seed!(s)
c = rand(d, I);
A = rand(d, R, I);
b = rand(d, R);

υ = JumpModel(0) # 📚 primal problem
JuMP.@variable(υ, x[1:I])
JuMP.@constraint(υ, du[r=1:R], ip(A[r, :], x) <= b[r])
JuMP.@objective(υ, Min, ip(c, x))
JuMP.@constraint(υ, x in JuMP.SecondOrderCone())
JuMP.set_attribute(υ, "QCPDual", 1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)

JuMP.objective_value(υ)


υ = JumpModel(0) # 📚 dual problem
JuMP.@variable(υ, pai[1:R] >= 0.)
JuMP.@variable(υ, dual_aux[1:I])
JuMP.@objective(υ, Max, -ip(b, pai))
JuMP.@constraint(υ, [i = 1:I], dual_aux[i] == ip(pai, A[:, i]) + c[i])
JuMP.@constraint(υ, dual_aux in JuMP.SecondOrderCone())
JuMP.set_attribute(υ, "QCPDual", 1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)
JuMP.objective_value(υ)


υ = JumpModel(0) # feas. sys
JuMP.@variable(υ, x[1:I])
JuMP.@variable(υ, pri_aux[1:R] >= 0.) # b - Ax
JuMP.@variable(υ, pai[1:R] >= 0.)
JuMP.@variable(υ, dual_aux[1:I]) # A' * pai + c
JuMP.@constraint(υ, x in JuMP.SecondOrderCone())
JuMP.@constraint(υ, dual_aux in JuMP.SecondOrderCone())
# definition
JuMP.@constraint(υ, [r = 1:R], pri_aux[r] == b[r] - ip(A[r, :], x))
JuMP.@constraint(υ, [i = 1:I], dual_aux[i] == ip(pai, A[:, i]) + c[i])
# CS condition
JuMP.@constraint(υ, [r=1:R], [pai[r], pri_aux[r]] in JuMP.SOS1())
JuMP.@constraint(υ, ip(x, dual_aux) <= 0) # ⚠️ CS condition for CQP is not componentwise !!!!
# 2 objectives
JuMP.@expression(υ, pri_obj, ip(c, x))
JuMP.@expression(υ, dual_obj, -ip(b, pai))
# strong dual's obj cut
# JuMP.@constraint(υ, dual_obj >= pri_obj)
# JuMP.set_attribute(υ, "FeasibilityTol", 1e-2)
# JuMP.set_attribute(υ, "IntFeasTol", 1e-1)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)

JuMP.value(pri_obj)
JuMP.value(dual_obj)


import JuMP
import Gurobi
using CairoMakie
import LinearAlgebra
# use ALR cuts (non-conv but Lipschitz) to under-relax a value function of MILP
# ALR augmenting function is ||⋅||_1
# thus the Maximum of ALR cuts is MILP representable
# inspired by SLDP algorithm1
# 14/12/23
function Q(b)
    m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    a = [6.,5,-4,2,-7,1]
    JuMP.@variable(m, x[1:6] >= 0)
    JuMP.set_integer.(x[1:3])
    JuMP.@constraint(m, sum(a[i] * x[i] for i in 1:6) == b)
    c = [3.,7/2,3,6,7,5]
    JuMP.@objective(m, Min, sum(c[i] * x[i] for i in 1:6))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m)
end
function Q_ALR(lambda,rho,b0)
    m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    a = [6.,5,-4,2,-7,1]
    JuMP.@variable(m, x[1:6] >= 0)
    JuMP.set_integer.(x[1:3])
    e = sum(a[i] * x[i] for i in 1:6) - b0
    JuMP.@variable(m, t)
    JuMP.@constraint(m, [t; e] in JuMP.MOI.NormOneCone(2))
    c = [3.,7/2,3,6,7,5]
    JuMP.@objective( m, Min, sum(c[i] * x[i] for i in 1:6) - lambda * e + rho * t )
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m)
end
function cut(lambda, rho, b0, b)
    Q_ALR(lambda,rho,b0) - lambda * (b0 - b) - rho * LinearAlgebra.norm(b0 - b, 1)
end
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))

f = Figure();
ax = Axis(f[1, 1],limits = (-13, 23, -25, 25));
xt = range(-10, 20; length=300);
yt = Q.(xt);
lines!(ax, xt, yt;color = :black, linewidth = 1)
GRB_ENV = Gurobi.Env()
norm_sense = Cdouble(1.0)
norm_arg_num = Cint(1)

lambda = -2.
rho = 5.
iteNum = 13

m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
JuMP.set_silent(m)
JuMP.@variable(m, -10. <= x <= 20.)
JuMP.@variable(m, tha)
JuMP.@objective(m, Min, tha)
JuMP.@variable(m, d1[1:iteNum])
JuMP.@variable(m, n1[1:iteNum]) 

trials = [10.14]
lbs = [-101.]
for ite in 1:iteNum
    b0 = trials[end] # current trial point
    JuMP.@constraint(m, d1[ite] == b0 - x)
    Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1[ite]), norm_arg_num, [column(d1[ite])], norm_sense) # n1[ite] = || d1[ite] ||_1
    y0 = Q_ALR(lambda, rho, b0)
    tmpy = Q(b0)
    text!(ax, b0, tmpy + 1.; text = "$ite")
    @assert abs(y0 - tmpy) < 1e-6 # ensure tightness of the LR+ cut
    lines!(ax, xt, cut.(lambda, rho, b0, xt)) # view the cut
    JuMP.@constraint(m, tha >= y0 - lambda * d1[ite] - rho * n1[ite]) # add cut
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    push!(trials, JuMP.value(x))
    push!(lbs, JuMP.objective_value(m))
    @info "ite = $ite, lb = $(lbs[end])"
end

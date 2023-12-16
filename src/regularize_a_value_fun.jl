import JuMP
import Gurobi
using CairoMakie
import LinearAlgebra
# 2 examples illustrating how to regularize a value function that is not Lipschitzian over certain domain
# 16/12/23
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))

example_case = false # true: case 1 (Quad_NL); false: case 2 (MILP)
if example_case
    function Q(xt)::Float64 # value function of a MILP
        # Min     w
        # s.t.    z = xt
        #         (w-1)^2 + z^2 ≤ 1 (quad nonlinear)
        #         w, z
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, z)
        JuMP.@variable(m, w)
        JuMP.@constraint(m, z == xt)
        JuMP.@constraint(m, (w - 1.)^2 + z^2 ≤ 1.)
        JuMP.@objective(m, Min, w)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        JuMP.objective_value(m)
    end
else
    function Q(xt)::Float64 # value function of a MILP
        # Min   w
        # s.t.  z = xt
        #       w >= z
        #       w Bin
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, z)
        JuMP.@variable(m, w, Bin)
        JuMP.@constraint(m, z == xt)
        JuMP.@constraint(m, w >= z)
        JuMP.@objective(m, Min, w)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        JuMP.objective_value(m)
    end
end

if example_case
    function Q_R(xt, sigma) # regularized using ℓ1-norm
        # Min     w + sigma * n
        # s.t.    b = xt - z
        #         n == ||b||₁ 
        #         (w-1)^2 + z^2 ≤ 1 (quad nonlinear)
        #         w, z, b, n
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, w)
        JuMP.@variable(m, z)
        JuMP.@variable(m, b)
        JuMP.@constraint(m, b == xt - z)
        JuMP.@variable(m, n)
        Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n), norm_arg_num, [column(b)], norm_sense) # n1[ite] = || d1[ite] ||_1
        JuMP.@constraint(m, (w - 1.)^2 + z^2 ≤ 1.)
        JuMP.@objective(m, Min, w + sigma * n) # regularized obj
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        if JuMP.value(b) > 1e-6
            @warn "The solution of Regularized is NOT feasible for its naive counterpart, please tune up the ρ!" "bias = $(JuMP.value(b))"
        end
        JuMP.objective_value(m)
    end
else
    function Q_R(xt, sigma) # regularized using ℓ1-norm
        # Min     w + sigma * n
        # s.t.    b = xt - z
        #         n == ||b||₁ 
        #         w >= z
        #         w Bin
        #         w, z, b, n
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(m)
        JuMP.@variable(m, w, Bin)
        JuMP.@variable(m, z)
        JuMP.@variable(m, b)
        JuMP.@constraint(m, b == xt - z)
        JuMP.@variable(m, n)
        Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n), norm_arg_num, [column(b)], norm_sense) # n1[ite] = || d1[ite] ||_1
        JuMP.@constraint(m, w >= z)
        JuMP.@objective(m, Min, w + sigma * n) # regularized obj
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        if JuMP.value(b) > 1e-6
            @warn "The solution of Regularized is NOT feasible for its naive counterpart, please tune up the ρ!" "bias = $(JuMP.value(b))"
        end
        JuMP.objective_value(m)
    end
end

GRB_ENV = Gurobi.Env() # Gurobi can deal with Quad_NL and MILP, thus Gurobi is competent for these 2 cases
norm_sense = Cdouble(1.0)
norm_arg_num = Cint(1)
sigma = 5. # coefficient of regularization

f = Figure();
ax = Axis(f[1, 1]) #,limits = (-13, 23, -25, 25));
xt = range(0., 1.; length=40);
yt = Q.(xt);
stem!(ax, xt, yt; color = :black) # the primal value function
xt = range(0., 1.; length=100);
yt_R = Q_R.(xt, sigma)
lines!(ax, xt, yt_R; color = :red) # the regularized function <= primal value function

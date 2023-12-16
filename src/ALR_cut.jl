import JuMP
import Gurobi
using CairoMakie
import LinearAlgebra
# use ALR cuts (non-conv but Lipschitz) to under-relax a value function of MILP
# ALR augmenting function is ||⋅||_1
# thus the Maximum of ALR cuts is MILP representable
# inspired by SLDP algorithm1
# 16/12/23
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
function Q(xt)::Float64 # value function of a MILP
    # Min     <c, x>
    # s.t.    xt - z == 0 [λ, ρ]
    #         <a, x> == z 
    #         x[1:6] >= 0
    #         x[1:3] Int
    #         z
    a = [6.,5,-4,2,-7,1]
    c = [3.,7/2,3,6,7,5]
    m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    JuMP.@variable(m, x[1:6] >= 0)
    JuMP.set_integer.(x[1:3])
    JuMP.@variable(m, z) # copy variable
    JuMP.@constraint(m, z == xt) # copy constraint
    JuMP.@constraint(m, a' * x == z)
    JuMP.@objective(m, Min, c' * x)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m)
end
function Q_ALR(xt, lambda, rho)::Float64 # for a given param xt, do the ALR opt corresponding to the naive Q(xt)
    # Min     <c, x> + <λ, b> + ρ * n
    # s.t.    b == xt - z
    #         n == ||b||₁ 
    #         <a, x> == z 
    #         x[1:6] >= 0
    #         x[1:3] Int
    #         z (copy)
    #         b (bias)
    #         n (norm)
    a = [6.,5,-4,2,-7,1]
    c = [3.,7/2,3,6,7,5]
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(m)
    JuMP.@variable(m, x[1:6] >= 0)
    JuMP.set_integer.(x[1:3])
    JuMP.@variable(m, z) # copy variable, with copy constraint relaxed in ALR
    JuMP.@variable(m, b) # bias
    JuMP.@constraint(m, b == xt - z)
    JuMP.@variable(m, n) # norm
    Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n), norm_arg_num, [column(b)], norm_sense) # n1[ite] = || d1[ite] ||_1
    JuMP.@constraint(m, a' * x == z)
    JuMP.@objective(m, Min, c' * x + lambda' * b + rho * n) # ALR's obj
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    if JuMP.value(b) > 1e-6
        @error "The solution of ALR is NOT feasible for its naive counterpart, please tune up the ρ!" "bias = $(JuMP.value(b))"
    end
    JuMP.objective_value(m)
end
function cut_map(xt0, lambda, rho)::Function
    # Q(xt) ≥ Q_ALR(xt, lambda, rho) := Min <c, x> + <λ, xt - A * x> + ρ * ||xt - A * x||₁
    # This is true: ||xt - A * x|| ≥ ||xt0 - A * x|| - ||xt - xt0||. 
    # Thus we have the relation: Q(⋅) ≥ Q_ALR(⋅, lambda, rho) ≥ cut_map(xt0, lambda, rho)(⋅). (Draw pictures to check it)
    x -> Q_ALR(xt0, lambda, rho) + lambda * (x - xt0) - rho * LinearAlgebra.norm(x - xt0, 1)
end

GRB_ENV = Gurobi.Env()
norm_sense = Cdouble(1.0)
norm_arg_num = Cint(1)
lambda = -0.7
rho = 5.

f = Figure();
ax = Axis(f[1, 1],limits = (-13, 23, -25, 25));
xt = range(-10, 20; length=300);
yt = Q.(xt);
yt_alr = Q_ALR.(xt, lambda, rho);
lines!(ax, xt, yt_alr; color = :red, linewidth = 3)
lines!(ax, xt, yt; color = :black, linewidth = 1)
for cut_gen_point in range(-10, 20; length = 11)
    lines!(ax, xt, cut_map(cut_gen_point, lambda, rho).(xt))
end

# iteNum = 13

# m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) # use direct_model to leverage API provided by Gurobi
# JuMP.set_silent(m)
# JuMP.@variable(m, -10. <= x <= 20.) # compact state space
# JuMP.@variable(m, tha) # the N+1 dimension for the surrogate
# JuMP.@objective(m, Min, tha) # we have no f_n() term. We only have a bare value function
# JuMP.@variable(m, d1[1:iteNum])
# JuMP.@variable(m, n1[1:iteNum]) 



# trials = [10.14]
# lbs = [-101.]
# for ite in 1:iteNum
#     b0 = trials[end] # current trial point
#     JuMP.@constraint(m, d1[ite] == b0 - x)
#     Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1[ite]), norm_arg_num, [column(d1[ite])], norm_sense) # n1[ite] = || d1[ite] ||_1
#     y0 = Q_ALR(lambda, rho, b0)
#     tmpy = Q(b0)
#     text!(ax, b0, tmpy + 1.; text = "$ite")
#     @assert abs(y0 - tmpy) < 1e-6 # ensure tightness of the LR+ cut
#     lines!(ax, xt, cut.(lambda, rho, b0, xt)) # view the cut
#     JuMP.@constraint(m, tha >= y0 - lambda * d1[ite] - rho * n1[ite]) # add cut
#     JuMP.optimize!(m)
#     @assert JuMP.termination_status(m) == JuMP.OPTIMAL
#     push!(trials, JuMP.value(x))
#     push!(lbs, JuMP.objective_value(m))
#     @info "ite = $ite, lb = $(lbs[end])"
# end


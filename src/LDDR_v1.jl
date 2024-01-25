import JuMP
import Gurobi
import Random
import Distributions
import Statistics
using Logging

# version 2.0: presolving an SP, calculate its statistical_lb and statistical_ub by basic methods 
# 2024/1/25

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function get_logNormal_dist(E, s) # args is of the target distribution
    m, v = E, s^2
    Distributions.LogNormal(log((m^2)/sqrt(v+m^2)), sqrt(log(v/(m^2)+1)))
end

Random.seed!(1)
global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()

if true # data for the program
    alpha = .05
    z_alpha = Distributions.quantile(Distributions.Normal(), 1. - alpha) # used in one-side confidence bound
    statistical_ub = 67924.7279074679 # due to policy 4p1, at N = 10000
    statistical_lb = 55838.76768156331 # due to EVPI, at N = 10000
    T, J, S = 4, 3, 5 # S is the samples taken in SAA
    MU = Float64[ # serves as the Expectation of D
        74  140 88
        64  99  105
        77  128 95
        42  120 78
    ]
    rho, rhoY = .6, .2
    EDj_bar = Statistics.mean.(eachcol(MU)) # indexed by j
    c_H, c_O = 15., 100.
    c_B = 30. * ones(T) # indexed by t
    c_B[T] *= 5
    c_Y = 1.2 * 4 * 15 * EDj_bar # indexed by j
    T_Y = .25 * EDj_bar # indexed by j, in (27c)
    production_UB = 6. * EDj_bar # indexed by j, in (27d)
    inventory_UB = 10. * EDj_bar # indexed by j, RHS of (27e, 27f)
    time_UB = 1.5 * sum.(eachrow(MU)) # indexed by t, RHS of (27c)
    o_UB = .25 * time_UB # indexed by t, RHS of (27g)
    obj3a_UB = statistical_ub
    d_epsilon = get_logNormal_dist(1., .5)
    d_delta = [get_logNormal_dist(MU[t, j], .2 * t * MU[t, j]) for t in eachindex(eachrow(MU)), j in eachindex(eachcol(MU))]
end

function one_side_bound_raw(sampler::Function, N)
    v = [sampler() for _ in 1:N]
    x̄ = Statistics.mean(v)
    s = Statistics.std(v; mean = x̄)
    x̄, z_alpha * s / sqrt(N) # (67702.00986320838, 222.71804425951458) at N = 10000
end
function a_sample_value_by_policy_4p1()
    costs = zeros(T)
    x_m1 = [0. for j in 1:J]
    x_B_m1 = [0. for j in 1:J]
    x_H_m1 = [0. for j in 1:J]
    D_s = deepcopy(MU)
    (Ups_s = deepcopy(D_s); Ups_s .= 1.)
    for t in 1:T
        if t >= 2
            for j in 1:J # sampling to generate ξₜ
                epsilon, delta = rand(d_epsilon), rand(d_delta[t,j])
                Ups_s[t, j] = rho * Ups_s[t-1, j] + (1-rho) * epsilon
                D_s[t, j] = rhoY * Ups_s[t, j] * MU[t, j] + (1-rhoY) * delta
            end
            for u in t+1:T, j in 1:J # calculate 𝔼_{ξᵗ}{ r.v. > t }
                D_s[u, j] = MU[u, j] * (1 + rhoY * rho^(u-t) * (Ups_s[t, j] - 1.))
            end
        end
        m = JumpModel()
        JuMP.@variable(m, y[t:T, 1:J], Bin)
        JuMP.@variable(m, 0. <= o[u = t:T] <= o_UB[u])
        JuMP.@variable(m,   x[t:T, 1:J] >= 0.)
        JuMP.@variable(m, x_B[t:T, 1:J] >= 0.)
        JuMP.@variable(m, x_H[t:T, 1:J] >= 0.)
        JuMP.@constraint(m, [u=t:T], sum(T_Y[j] * y[u,j] + x[u,j] for j in 1:J) - o[u] <= time_UB[u])
        JuMP.@constraint(m, [u=t:T, j=1:J], x[u,j] <= production_UB[j] * y[u,j])
        JuMP.@constraint(m, [u=t:T, j=1:J], x_H[u,j] + x[u,j] <= inventory_UB[j])
        JuMP.@constraint(m, [j=1:J], x_B[t,j] - x_H[t,j] + x_H_m1[j] - x_B_m1[j] + x_m1[j] == D_s[t,j]) # this RHS is by sampling
        t != T && JuMP.@constraint(m, [u=t+1:T, j=1:J], x_B[u,j] - x_H[u,j] + x_H[u-1,j] - x_B[u-1,j] + x[u-1,j] == D_s[u,j]) # these RHS's are by calculating cond_𝔼
        costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
        JuMP.@objective(m, Min, sum(c_O * o[u] + costs_oto(u) for u in t:T))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        costs[t] = c_O * JuMP.value(o[t]) + sum(c_B[t] * JuMP.value(x_B[t,j]) + c_H * JuMP.value(x_H[t,j]) + c_Y[j] * JuMP.value(y[t,j]) for j in 1:J)
        x_m1 = [JuMP.value(x[t,j]) for j in 1:J]
        x_B_m1 = [JuMP.value(x_B[t,j]) for j in 1:J]
        x_H_m1 = [JuMP.value(x_H[t,j]) for j in 1:J]
    end
    sum(costs)
end
function a_perfect_info_sample_value()
    D_s = deepcopy(MU)
    (Ups_s = deepcopy(D_s); Ups_s .= 1.)
    for t in 2:T, j in 1:J
        epsilon, delta = rand(d_epsilon), rand(d_delta[t,j])
        Ups_s[t, j] = rho * Ups_s[t-1, j] + (1-rho) * epsilon
        D_s[t, j] = rhoY * Ups_s[t, j] * MU[t, j] + (1-rhoY) * delta
    end
    t = 1
    m = JumpModel()
    JuMP.@variable(m, y[t:T, 1:J], Bin)
    JuMP.@variable(m, 0. <= o[u = t:T] <= o_UB[u])
    JuMP.@variable(m,   x[t:T, 1:J] >= 0.)
    JuMP.@variable(m, x_B[t:T, 1:J] >= 0.)
    JuMP.@variable(m, x_H[t:T, 1:J] >= 0.)
    JuMP.@constraint(m, [u=t:T], sum(T_Y[j] * y[u,j] + x[u,j] for j in 1:J) - o[u] <= time_UB[u])
    JuMP.@constraint(m, [u=t:T, j=1:J], x[u,j] <= production_UB[j] * y[u,j])
    JuMP.@constraint(m, [u=t:T, j=1:J], x_H[u,j] + x[u,j] <= inventory_UB[j])
    JuMP.@constraint(m, [j=1:J], x_B[t,j] - x_H[t,j] + 0. - 0. + 0. == D_s[t,j])
    JuMP.@constraint(m, [u=t+1:T, j=1:J], x_B[u,j] - x_H[u,j] + x_H[u-1,j] - x_B[u-1,j] + x[u-1,j] == D_s[u,j])
    costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
    JuMP.@objective(m, Min, sum(c_O * o[u] + costs_oto(u) for u in t:T))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    JuMP.objective_value(m)
end

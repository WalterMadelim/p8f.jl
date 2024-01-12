using CairoMakie
import Gurobi
import JuMP
using Logging
import Distributions
import Random
import Statistics
import QuadGK

# a sampling based gap analysis of a candidate solution
# see ShaBook2021, Chapter 5 (5.6.1) p194
# example is the Newsvendor problem
# we also draw plots for this problem
# 12/01/24

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()

function get_lognormal_d(mean, std) # mean, std is the statistics of the target distribution
    m, v = mean, std^2
    d = Distributions.LogNormal(log((m^2)/sqrt(v+m^2)), sqrt(log(v/(m^2)+1)))
end
function sample_LH_lognormal(d::Distributions.Distribution, N::Int) 
    Random.shuffle([Distributions.quantile(d, rand( Distributions.Uniform((i-1)/N, i/N) )) for i in 1:N])
end
function int_of_H(d::Distributions.Distribution, x::Float64)::Float64
    # ∫₀ˣ H(t)dt
    # where, H is the cdf of r.v. D
    # D ∼ distribution `d`
    QuadGK.quadgk(t -> Distributions.cdf(d,t), 0., x; rtol=1e-8)[1]
end
function objfun(d::Distributions.Distribution, c, b, h, x)
    b * Distributions.mean(d) + (c - b) * x + (b + h) * int_of_H(d, x)
end

alpha = .05 # significance level

c, b, h = .2, .7, .3 # parameters of the newsvendor problem
d = get_lognormal_d(1., .5) # args: mean, std of the target `d`

x_prim = Distributions.quantile(d, (b-c)/(b+h)) # theoretical opt_solu
v_prim = objfun(d, c, b, h, x_prim) # theoretical opt_val

x_can = 1.
N, M, N1 = 2000, 100, 10000

function ϑ_hat_N(D::Vector{Float64}, c, b, h)
    N = length(D)
    m = JumpModel()
    JuMP.@variable(m, x >= 0.)
    JuMP.@variable(m, wp[1:N] >= 0.)
    JuMP.@variable(m, wn[1:N] >= 0.)
    JuMP.@constraint(m, [s = 1:N], wp[s] - wn[s] == D[s] - x)
    JuMP.@objective(m, Min, c * x + fill(1/N, N)' * [b * wp[s] + h * wn[s] for s in 1:N])
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    JuMP.objective_value(m)
end

function eq168_169(d::Distributions.Distribution, M, N, c, b, h)
    # vec = [ϑ_hat_N(rand(d, N), c, b, h) for _ in 1:M]
    vec = [ϑ_hat_N(sample_LH_lognormal(d, N), c, b, h) for _ in 1:M] # we can change MC  to LH
    e168 = Statistics.mean(vec)
    e169 = Statistics.var(vec) / M
    e168, e169
end

function F_x_xi(x::Float64, D::Float64, c, b, h)
    wp, wn = max(D - x, 0.), max(x - D, 0.)
    c * x + b * wp + h * wn
end

function eq1701_171(d::Distributions.Distribution, N1, x_can, c, b, h)
    Ds = rand(d, N1)
    vec = F_x_xi.(x_can::Float64, Ds, c::Float64, b::Float64, h::Float64)
    eq1701 = Statistics.mean(vec)
    eq171 = Statistics.var(vec) / N1
    eq1701, eq171
end

function ub_for_f_x_can(d::Distributions.Distribution, alpha, N1, x_can, c, b, h)
    z_alpha = Distributions.quantile(Distributions.Normal(), 1. - alpha)
    eq1701, eq171 = eq1701_171(d, N1, x_can, c, b, h)
    eq1701 + z_alpha * sqrt(eq171)
end

function L_NM(d::Distributions.Distribution, M, N, alpha, c, b, h)
    t_alpha_Mm1 = Distributions.quantile(Distributions.TDist(M-1), 1. - alpha)
    v, s = eq168_169(d, M, N, c, b, h)
    v - t_alpha_Mm1 * sqrt(s)
end

function conser_ub_for_gap_x_can(d::Distributions.Distribution, alpha, M, N, N1, x_can, c, b, h) # 💡 the central result in Chapter 5, ShaBook2021
    z_alpha = Distributions.quantile(Distributions.Normal(), 1. - alpha)
    eq168, eq169 = eq168_169(d, M, N, c, b, h)
    eq1701, eq171 = eq1701_171(d, N1, x_can, c, b, h)
    eq1701 - eq168 + z_alpha * sqrt(eq171 + eq169)
end

function rel_gap_for_x_can(d::Distributions.Distribution, alpha, M, N, N1, x_can, c, b, h)
    gap_abs = conser_ub_for_gap_x_can(d::Distributions.Distribution, alpha, M, N, N1, x_can, c, b, h)
    ub = ub_for_f_x_can(d::Distributions.Distribution, alpha, N1, x_can, c, b, h)
    gap_rel = gap_abs / ub
    gap_rel = round(gap_rel; digits = 4)
    @info "the rel_gap for x = $(x_can) is $(100. * gap_rel)% (For a ref only)"
end

rel_gap_for_x_can(d::Distributions.Distribution, alpha, M, N, N1, x_can, c, b, h)

f = Figure();
axs = Axis.([f[i...] for i in Iterators.product([1,2],[1,2])]); # 2 * 2 subplots
xt = range(0, 2; length = 1000);
yt = objfun.(d,c,b,h,xt);

lines!(axs[1], fill(x_prim,2), [.0, .6]; color = :tan, linestyle = :dot)
lines!(axs[1], [.0, 2.], fill(v_prim,2); color = :tan, linestyle = :dot)
text!(axs[1], x_prim, 0.; text = "$x_prim", fontsize = 8)
text!(axs[1], 0., v_prim; text = "$v_prim", fontsize = 8)
lines!(axs[1], xt, yt; color = :brown)
lines!(axs[1], xt, yt .- c * xt ; color = :tomato) # Q(x)
lines!(axs[1], xt, c * xt; color = :pink)
lines!(axs[1], xt, .377 .- c * xt; color = :yellow) # auxiliary

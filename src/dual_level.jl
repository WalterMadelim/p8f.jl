import JuMP
import Gurobi
import Statistics
import Random
using Logging

# solving a dual problem with level method
# instance MKAP
# cf. [Rui Chen 2024]
# 22/01/24

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
function get_params(N, M, R)
    w = rand(Int, N) .% R
    w[w .< -0.5] .= w[w .< -0.5] .+ R
    w *= 1. # coefficient in programs must be of Float64 type
    p = .6 * w .+ rand(N) * .4 * R
    vec = rand(M)
    vec = vec ./ sum(vec)
    C = .5 * sum(w) * vec
    p/100, w/100, C/100
end
function j2k(j)
    div(j-1, xsblen) + 1
end
function j2ii(j)
    (j-1) % xsblen + 1
end
function sbrg(k) # for subvector of x only
    1+xsblen*(k-1) : xsblen*k
end
function g(i, k)
    [w[sbrg(k)]; -C[i]]
end
function nm2(m)
    sum(e ^ 2 for e in m)
end
function dist2(m, mb)
    nm2(m .- mb)
end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
N, M, K = 20, 10, 5
Random.seed!(1)
p, w, C = get_params(N, M, 999+1);
@assert length(p) == length(w) == N
@assert length(C) == M
xsblen = div(N, K) # length of x_I for each block

function primal_with_copy()
    m = JumpModel()
    JuMP.@variable(m, 0. <= xI[i = 1:M, k = 1:K, ii = 1:xsblen+1] <= 1.) # decision (xsblen+1)-length-vector in a block indexed by [i,k]
    JuMP.@constraint(m, [i = 1:M, k = 1:K], g(i, k)' * xI[i, k, :] <= 0.) # in-block
    JuMP.@variable(m, y[i = 1:M, k = 1:K]) # complicating / natural
    JuMP.@variable(m, x[i = 1:M, j = 1:N]) # complicating / natural
    JuMP.@constraint(m, beta_y[i = 1:M], -sum(y[i, :]) >= -1.) # complicating
    JuMP.@constraint(m, beta_x[j = 1:N], -sum(x[:, j]) >= -1.) # complicating 
    JuMP.@constraint(m, pi_y[i = 1:M, k = 1:K], y[i, k] == xI[i, k, xsblen+1]) # copy
    JuMP.@constraint(m, pi_x[i = 1:M, j = 1:N], x[i, j] == xI[i, j2k(j), j2ii(j)]) # copy
    JuMP.@objective(m, Min, -sum(p' * x[i, :] for i in 1:M)) # obj being fun of complicating variables
    JuMP.set_integer.(xI) # `X`
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    JuMP.objective_value(m) # -54.747354489285534
end

Vhat = [Vector{Float64}[] for i = 1:M, k = 1:K]; # Vhat[i, k] store the ext points of conv(Q[i, k]) 
Dmat = zeros(M, K); # a tmp matrix storing pricing problem values
lb = [-Inf, -Inf]; # ObjValue
ub = [Inf, Inf]; # ObjBound
beta_y_t = NaN * ones(M, 2);
beta_x_t = NaN * ones(N, 2);
pi_y_t   = NaN * ones(M, K, 2);
pi_x_t   = NaN * ones(M, N, 2);
for lv_ite in 1:typemax(Int)
    m = JumpModel()
    if true # to be copied to the stability QP
        JuMP.@variable(m, beta_y[i = 1:M] >= 0.)
        JuMP.@variable(m, beta_x[j = 1:N] >= 0.)
        JuMP.@variable(m, pi_y[i = 1:M, k = 1:K])
        JuMP.@variable(m, pi_x[i = 1:M, j = 1:N])
        JuMP.@constraint(m, [i = 1:M, k = 1:K], beta_y[i] - pi_y[i, k] == 0.) # coeff of y == 0.
        JuMP.@constraint(m, [i = 1:M, j = 1:N], beta_x[j] - pi_x[i, j] == p[j]) # coeff of x == 0.
        JuMP.@variable(m, th[i = 1:M, k = 1:K]) # one θ per block
        for i in 1:M, k in 1:K
            for coeff_vec in Vhat[i, k]
                JuMP.@constraint(m, th[i, k] <= coeff_vec' * [pi_x[i, sbrg(k)]; pi_y[i, k]]) # (EC.3b)
            end
        end
        JuMP.@variable(m, obj3a <= -54.) # (EC.3d)
        JuMP.@constraint(m, obj3a == sum(th) - sum(beta_y) - sum(beta_x)) # purple underline
    end
    JuMP.@objective(m, Max, obj3a)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    ub[end] = JuMP.objective_value(m) # ObjBound
    if ub[end] < ub[begin]
        ub[begin] = ub[end]
    end
    if lb[begin] == -Inf
        beta_y_t[:, end]  .= JuMP.value.(beta_y)
        beta_x_t[:, end]  .= JuMP.value.(beta_x)
        pi_y_t[:, :, end] .= JuMP.value.(pi_y)
        pi_x_t[:, :, end] .= JuMP.value.(pi_x)
    else
        m = JumpModel()
        JuMP.@variable(m, beta_y[i = 1:M] >= 0.)
        JuMP.@variable(m, beta_x[j = 1:N] >= 0.)
        JuMP.@variable(m, pi_y[i = 1:M, k = 1:K])
        JuMP.@variable(m, pi_x[i = 1:M, j = 1:N])
        JuMP.@constraint(m, [i = 1:M, k = 1:K], beta_y[i] - pi_y[i, k] == 0.) # coeff of y == 0.
        JuMP.@constraint(m, [i = 1:M, j = 1:N], beta_x[j] - pi_x[i, j] == p[j]) # coeff of x == 0.
        JuMP.@variable(m, th[i = 1:M, k = 1:K]) # one θ per block
        for i in 1:M, k in 1:K
            for coeff_vec in Vhat[i, k]
                JuMP.@constraint(m, th[i, k] <= coeff_vec' * [pi_x[i, sbrg(k)]; pi_y[i, k]]) # (EC.3b)
            end
        end
        JuMP.@variable(m, obj3a <= -54.) # (EC.3d)
        JuMP.@constraint(m, obj3a == sum(th) - sum(beta_y) - sum(beta_x)) # purple underline
        # above part copied
        JuMP.@constraint(m, obj3a >= .7 * ub[begin] + .3 * lb[begin]) # (EC.4)
        JuMP.@objective(m, Min, dist2(beta_x, beta_x_t[:, end]) + dist2(beta_y, beta_y_t[:, end]) + dist2(pi_x, pi_x_t[:, :, end]) + dist2(pi_y, pi_y_t[:, :, end]))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        beta_y_t[:, end]  .= JuMP.value.(beta_y)
        beta_x_t[:, end]  .= JuMP.value.(beta_x)
        pi_y_t[:, :, end] .= JuMP.value.(pi_y)
        pi_x_t[:, :, end] .= JuMP.value.(pi_x)
    end
    # JuMP.@constraint(m, pi_y[i = 1:M, k = 1:K], y[i, k] == xI[i, k, xsblen+1]) # copy
    # JuMP.@constraint(m, pi_x[i = 1:M, j = 1:N], x[i, j] == xI[i, j2k(j), j2ii(j)]) # copy
    # i = 1, j = 1:4, k = 1
    # pi_x[i, j] * xI[i, j2k(j), j2ii(j)]; pi_y[i, 1] * xI[i, k, end]
    for i in 1:M, k in 1:K # for each block, line 9
        m = JumpModel()
        JuMP.@variable(m, 0. <= xI[1:xsblen+1] <= 1., Int)
        JuMP.@constraint(m, g(i, k)' * xI <= 0.) # in-block constraint
        JuMP.@objective(m, Min, [pi_x_t[i, sbrg(k), end]; pi_y_t[i, k, end]]' * xI)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        push!(Vhat[i, k], JuMP.value.(xI))
        Dmat[i, k] = JuMP.objective_value(m)
    end
    lb[end] = sum(Dmat) - sum(beta_y_t[:, end]) - sum(beta_x_t[:, end]) # a feasible ObjValue
    if lb[end] > lb[begin]
        lb[begin] = lb[end] # update incumbent ObjValue
        ObjValue, ObjBound = lb[begin], ub[begin]
        rel_gap = abs((ObjBound - ObjValue) / ObjValue)
        @info "▶ level_ite = $(lv_ite)" ObjValue ObjBound rel_gap
        beta_y_t[:, begin]  .= beta_y_t[:, end] 
        beta_x_t[:, begin]  .= beta_x_t[:, end] 
        pi_y_t[:, :, begin] .= pi_y_t[:, :, end]
        pi_x_t[:, :, begin] .= pi_x_t[:, :, end] # update incumbent dual variable
    end
    if rel_gap < 0.01 / 100
        @info "😊 dual problem convergent"
        return lb[begin]
    end
end
beta_y = beta_y_t[:, begin]
beta_x = beta_x_t[:, begin]
pi_y   = pi_y_t[:, :, begin]  
pi_x   = pi_x_t[:, :, begin]  

if false
    function primal_formulation()
        m = JumpModel()
        JuMP.@variable(m, y[i = 1:M, k = 1:K], Bin)
        JuMP.@variable(m, x[i = 1:M, j = 1:N], Bin)
        JuMP.@constraint(m, cply[i = 1:M], sum(y[i, :]) <= 1.)
        JuMP.@constraint(m, cplx[j = 1:N], sum(x[:, j]) <= 1.)
        JuMP.@constraint(m, [i = 1:M, k = 1:K], w[sbrg(k)]' * x[i, sbrg(k)] <= C[i] * y[i, k])
        JuMP.@objective(m, Min, -sum(p' * x[i, :] for i = 1:M))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        JuMP.objective_value(m) # -54.747354489285534, and z_LP = -64.65610760976934
    end
end

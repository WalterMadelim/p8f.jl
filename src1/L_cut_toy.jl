using CairoMakie
import JuMP
import Gurobi
import LinearAlgebra
using Logging

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
function cut_fun(hatQ, x, id)
    # cx * x + ct * t >= rhs
    cx, ct, rhs = hatQ["cx"][id], hatQ["ct"][id], hatQ["rhs"][id]
    @assert ct > 0.
    1/ct * (rhs - cx * x)
end
function Q(x0)::Float64 # the primal value function is used in algorithm1
    m = JumpModel()
    JuMP.@variable(m, 0. <= b1 <= 1.) # relax Int firstly
    JuMP.@variable(m, c1)
    JuMP.@variable(m, x1)
    JuMP.@variable(m, a1)
    JuMP.@variable(m, f1)
    JuMP.@constraint(m, c1 == 2. * b1 - 1.)
    JuMP.@constraint(m, x1 == x0 + xi1 + c1) # linking
    JuMP.@constraint(m, a1 >=  x1)
    JuMP.@constraint(m, a1 >= -x1)
    JuMP.@constraint(m, f1 == beta^(1-1) * a1)
    JuMP.@objective(m, Min, f1) # objective
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m) # this is objective value
end
function Q_ast(pai, pai0)
    m = JumpModel()
    JuMP.@variable(m, -4. <= x0 <= 5.) # Int means the cut is under approx of Int valid values only, it doesn't need to be under the LP relaxed value function
    #--------------------------------
    JuMP.@variable(m, 0. <= b1 <= 1.) # relax Int firstly
    JuMP.@variable(m, c1)
    JuMP.@variable(m, x1)
    JuMP.@variable(m, a1)
    JuMP.@variable(m, f1)
    JuMP.@constraint(m, c1 == 2. * b1 - 1.)
    JuMP.@constraint(m, x1 == x0 + xi1 + c1) # linking
    JuMP.@constraint(m, a1 >=  x1)
    JuMP.@constraint(m, a1 >= -x1)
    JuMP.@constraint(m, f1 == beta^(1-1) * a1)
    #--------------------------------
    JuMP.@objective(m, Min, pai * x0 + pai0 * f1) # this obj doesn't have any relation to the obj of 1st stage
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    cp = JuMP.value(x0)
    cp0 = JuMP.value(f1)
    JuMP.objective_value(m), cp, cp0
end
function initialize_Q_ast_hat(Q_ast, Q_ast_hat, pai, pai0)
    _, cp, cp0 = Q_ast(pai, pai0) 
    push!(Q_ast_hat["by_pai0"    ], pai0)
    push!(Q_ast_hat["by_pai"     ], pai)
    push!(Q_ast_hat["cp0"        ], cp0)
    push!(Q_ast_hat["cp"         ], cp)
    push!(Q_ast_hat["id"         ], length(Q_ast_hat["id"]) + 1)
end
function algorithm1(Q, Q_ast, Q_ast_hat, x, th, delta = .98)
    # TODO: do not add pai_stalling check yet
    incumbent = Dict(
        "lb" => -Inf,
        "pai" => NaN,
        "pai0" => NaN,
        "rhs" => NaN,
        "cut_gened" => false
    )
    while true
        m = JumpModel()
        JuMP.@variable(m, 0. <= pai0)
        JuMP.@variable(m, pai)
        JuMP.@constraint(m, 1. - pai0 >=  pai)
        JuMP.@constraint(m, 1. - pai0 >= -pai)
        JuMP.@variable(m, phi)
        JuMP.@objective(m, Max, phi - x * pai - th * pai0) # phi is an overestimate of rhs
        for (cp, cp0) in zip(Q_ast_hat["cp"], Q_ast_hat["cp0"])
            JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0)
        end
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        ub = JuMP.objective_value(m) # so called objBound
        ub < 1e-6 && return incumbent # fail to generate a cut
        pai, pai0 = JuMP.value(pai), JuMP.value(pai0)
        rhs, cp, cp0 = Q_ast(pai, pai0)
        pai0 < 1e-4 && (cp0 = Q(cp)) # the 2nd stage in Q_ast is suppressed
        if true # push Q_ast_hat
            push!(Q_ast_hat["by_pai0"    ], pai0)
            push!(Q_ast_hat["by_pai"     ], pai)
            push!(Q_ast_hat["cp0"        ], cp0)
            push!(Q_ast_hat["cp"         ], cp)
            push!(Q_ast_hat["id"         ], length(Q_ast_hat["id"]) + 1)
        end
        lb = rhs - x * pai - th * pai0
        if incumbent["lb"] < lb
            incumbent["lb"] = lb
            incumbent["pai"] = pai
            incumbent["pai0"] = pai0
            incumbent["rhs"] = rhs
        end
        if incumbent["lb"] > (1.0 - delta) * ub + 1e-6 # it's pretty well to keep delta large, because it'll benefit sufficient exploring, there's no reason to stick to only one point
            incumbent["cut_gened"] = true
            return incumbent
        end
    end
end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()

xi1 = -2.
beta = .9
Q_ast_hat = Dict( # used (only) in algorithm1, need a global initialization
    "by_pai0"     => Float64[],
    "by_pai"      => Float64[],
    "cp0"         => Float64[], # theta_1 the aux
    "cp"          => Float64[], # first stage decision vector
    "id"          => Int[]
)
hatQ = Dict( # filled by algorithm1
    "is_inferior" => Bool[],
    "cx"          => Float64[],
    "ct"          => Float64[],
    "rhs"         => Float64[],
    "id"          => Int[]
)

f = Figure();
axs = Axis.([f[i...] for i in Iterators.product([1,2,3],[1,2,3])]);

initialize_Q_ast_hat(Q_ast, Q_ast_hat, -2., 1.) # ⚠️ the z_PI need to modify

x0 = range(-4., 5.; length = 500);
lines!(axs[1], x0, Q.(x0)) # relax int, draw this and its cut, you'll get what you want to see.
lines!(axs[2], x0, -2. * x0 .+ Q.(x0)) # relax int, draw this and its cut, you'll get what you want to see.
ite = 0




ite += 1
@info "beginning" ite
m = JumpModel() # the master problem of the 1st stage
JuMP.@variable(m, -4. <= x0 <= 5.) # the 1st stage decision (vector later)
JuMP.@variable(m, 0. <= th) # the 1st stage aux variable representing aftereffect
for (is_inferior, cx, ct, rhs) in zip(hatQ["is_inferior"], hatQ["cx"], hatQ["ct"], hatQ["rhs"])
    is_inferior || JuMP.@constraint(m, cx * x0 + ct * th >= rhs)
end
JuMP.@objective(m, Min, -2. * x0 + th) # the goal
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL
x0, th = JuMP.value(x0), JuMP.value(th) # trial
@info "aftere generating trial" ub=-2. * x0 + Q(x0) lb = -2. * x0 + th
scatter!(axs[1], [x0], [th])
scatter!(axs[2], [x0], [-2. * x0 + th])

algo1dict = algorithm1(Q, Q_ast, Q_ast_hat, x0, th)

if algo1dict["cut_gened"]
    push!(hatQ["is_inferior"], false)
    push!(hatQ["cx"         ], algo1dict["pai"])
    push!(hatQ["ct"         ], algo1dict["pai0"])
    push!(hatQ["rhs"        ], algo1dict["rhs"])
    push!(hatQ["id"         ], length(hatQ["id"]) + 1)
    @info "check this is one" ite hatQ["ct"][ite] + abs( hatQ["cx"][ite] )
end

x0 = range(3., 5.; length = 2);
lines!(axs[1], x0, [cut_fun(hatQ, yp, ite) for yp in x0]) # ⚠️ notice that cut_fun is related to the Q_hat dict
lines!(axs[2], x0, [-2. * p + cut_fun(hatQ, p, ite) for p in x0]) # ⚠️ notice that cut_fun is related to the Q_hat dict

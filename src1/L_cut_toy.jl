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
function cut_fun(x, id)
    # cx * x + ct * t >= rhs
    cx, ct, rhs = hatQ["cx"][id], hatQ["ct"][id], hatQ["rhs"][id]
    1/ct * (rhs - cx * x)
end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
Q_ast_hat = Dict( # used (only) in algorithm1, need a global initialization
    "by_pai0"     => Float64[],
    "by_pai"      => Float64[],
    "cp0"         => Float64[],
    "cp"          => Float64[],
    "id"          => Int[]
)
hatQ = Dict( # filled by algorithm1
    "is_inferior" => Bool[],
    "cx"          => Float64[],
    "ct"          => Float64[],
    "rhs"         => Float64[],
    "id"          => Int[]
)


function Q(y)::Float64
    m = JumpModel()
    JuMP.@variable(m, 0. <= x)
    JuMP.@constraint(m, x + 15. * y >= 8.)
    JuMP.@constraint(m, 3. * x + 10. * y >= 13.)
    JuMP.@constraint(m, x + 10. * y >= 7.)
    JuMP.@constraint(m, 2. * x - 10. * y >= -1.)
    JuMP.@constraint(m, 2. * x - 70. * y >= -49.)
    JuMP.@objective(m, Min, x)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(x)
end
function Q_ast(pai, pai0)
    m = JumpModel()
    JuMP.@variable(m, 0. <= y <= 1., Int) # Int means the cut is under approx of Int valid values only, it doesn't need to be under the LP relaxed value function
    JuMP.@variable(m, 0. <= x)
    JuMP.@constraint(m, x + 15. * y >= 8.)
    JuMP.@constraint(m, 3. * x + 10. * y >= 13.)
    JuMP.@constraint(m, x + 10. * y >= 7.)
    JuMP.@constraint(m, 2. * x - 10. * y >= -1.)
    JuMP.@constraint(m, 2. * x - 70. * y >= -49.)
    JuMP.@objective(m, Min, pai * y + pai0 * x)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    cp = JuMP.value(y)
    cp0 = JuMP.value(x)
    JuMP.objective_value(m), cp, cp0
end
if true # globally initialize Q_ast_hat
    cp, cp0 = Q_ast(1., 1.) # do z_PI to initialize Q_ast_hat
    push!(Q_ast_hat["by_pai0"    ], 1.)
    push!(Q_ast_hat["by_pai"     ], 1.)
    push!(Q_ast_hat["cp0"        ], cp0)
    push!(Q_ast_hat["cp"         ], cp)
    push!(Q_ast_hat["id"         ], length(Q_ast_hat["id"]) + 1)
end

function algorithm1(x, th, delta = .98)
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
        if incumbent["lb"] > (1.0 - delta) * ub + 1e-6
            incumbent["cut_gened"] = true
            return incumbent
        end
    end
end

f = Figure();
axs = Axis.([f[i...] for i in Iterators.product([1,2,3],[1,2,3])]);

y = range(0., 1.; length = 500);
# lines!(axs[1], y, Q.(y))
lines!(axs[1], y, y .+ Q.(y))

m = JumpModel() # the master problem of the 1st stage
JuMP.@variable(m, 0. <= y <= 1.) # the 1st stage decision (vector later)
JuMP.@variable(m, 0. <= th) # the 1st stage aux variable representing aftereffect
for (is_inferior, cx, ct, rhs) in zip(hatQ["is_inferior"], hatQ["cx"], hatQ["ct"], hatQ["rhs"])
    is_inferior || JuMP.@constraint(m, cx * y + ct * th >= rhs)
end
JuMP.@objective(m, Min, y + th)
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL
y, th = JuMP.value(y), JuMP.value(th) # trial
scatter!(axs[1], [y], [th])

algo1dict = algorithm1(y, th)
if algo1dict["cut_gened"]
    push!(hatQ["is_inferior"], false)
    push!(hatQ["cx"         ], algo1dict["pai"])
    push!(hatQ["ct"         ], algo1dict["pai0"])
    push!(hatQ["rhs"        ], algo1dict["rhs"])
    push!(hatQ["id"         ], length(hatQ["id"]) + 1)
end
y = range(0., 1.; length = 2);
lines!(axs[1], y, cut_fun.(y, 1))


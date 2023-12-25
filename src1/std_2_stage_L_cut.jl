import JuMP
import Gurobi
import LinearAlgebra
using Logging
import Distributions

# absolutely right this time
# 2-stage L-cut generation
# 25/12/23

if false
    function cut_fun(hatQ2, x, id)
        # cx * x + ct * t >= rhs
        cx, ct, rhs = hatQ2["cx"][id], hatQ2["ct"][id], hatQ2["rhs"][id]
        @assert ct > 0. 
        1/ct * (rhs - cx * x)
    end
end

column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
function Q2(stage_c, t2, xi2, x1sv) # the primal value function is used in algorithm1
    x1 = x1sv[3] # only need one component as link 
    m = JumpModel()
    JuMP.@variable(m, 0. <= b2 <= 1.) # relax Int firstly
    JuMP.@variable(m, c2)
    JuMP.@variable(m, x2)
    JuMP.@variable(m, a2)
    JuMP.@variable(m, f2)
    JuMP.@constraint(m, c2 == 2. * b2 - 1.)
    JuMP.@constraint(m, x2 == x1 + xi2 + c2) # linking
    JuMP.@constraint(m, a2 >=  x2)
    JuMP.@constraint(m, a2 >= -x2)
    JuMP.@constraint(m, f2 == stage_c(t2) * a2) # ⚠️modify the stage num
    JuMP.@objective(m, Min, f2) # objective
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m) # this is objective value
end
function Q2_ast(stage_c, t1, x0, xi1, xi2, pai, pai0)
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
    JuMP.@constraint(m, f1 == stage_c(t1) * a1) # ⚠️modify the stage num
    #--------------------------------
    JuMP.@variable(m, 0. <= b2 <= 1.) # relax Int firstly
    JuMP.@variable(m, c2)
    JuMP.@variable(m, x2)
    JuMP.@variable(m, a2)
    JuMP.@variable(m, f2)
    JuMP.@constraint(m, c2 == 2. * b2 - 1.)
    JuMP.@constraint(m, x2 == x1 + xi2 + c2) # linking
    JuMP.@constraint(m, a2 >=  x2)
    JuMP.@constraint(m, a2 >= -x2)
    JuMP.@constraint(m, f2 == stage_c(t1+1) * a2) # ⚠️modify the stage num
    #--------------------------------
    x1sv = [b1, c1, x1, a1, f1] # ⚠️
    JuMP.@objective(m, Min, pai' * x1sv + pai0 * f2) # ⚠️ # this obj doesn't have any relation to the obj of 1st stage
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    cp = JuMP.value.(x1sv)
    cp0 = JuMP.value(f2)
    JuMP.objective_value(m), cp, cp0 # the precise value of (12), one extreme point of projed Ks, second stage cost
end
function initialize_Q2_ast_hat(Q2_ast, Q2_ast_hat, stage_c, t1, x0, xi1, xi2, pai, pai0)
    _, cp, cp0 = Q2_ast(stage_c, t1, x0, xi1, xi2, pai, pai0) 
    push!(Q2_ast_hat["by_pai0"    ], pai0)
    push!(Q2_ast_hat["by_pai"     ], pai)
    push!(Q2_ast_hat["cp0"        ], cp0)
    push!(Q2_ast_hat["cp"         ], cp)
    push!(Q2_ast_hat["id"         ], length(Q2_ast_hat["id"]) + 1)
end
function algorithm1(Q2_ast_hat, Q2_ast, stage_c, t1, x0, xi1, xi2, Q2, x1sv, th1)
    # TODO: do not add pai_stalling check yet
    incumbent = Dict(
        "lb" => -Inf,
        "pai" => Float64[NaN for _ in eachindex(x1sv)],
        "pai0" => NaN,
        "rhs" => NaN,
        "cut_gened" => false
    )
    while true
        if true
            m = JumpModel()
            JuMP.@variable(m, 0. <= pai0)
            JuMP.@variable(m, pai[eachindex(x1sv)])
            JuMP.@variable(m, n1_pai)
            JuMP.@constraint(m, pai0 + n1_pai <= 1.)    
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(length(pai)), column.(pai), norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@variable(m, phi)
        end
        JuMP.@objective(m, Max, phi - x1sv' * pai - th1 * pai0) # phi is an overestimate of rhs
        @assert !isempty(Q2_ast_hat["cp"]) "you must initialize Q2_ast_hat, maybe use `initialize_Q2_ast_hat`"
        for (cp, cp0) in zip(Q2_ast_hat["cp"], Q2_ast_hat["cp0"])
            JuMP.@constraint(m, phi <= cp' * pai + cp0 * pai0)
        end
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        ub = JuMP.objective_value(m) # so called objBound
        ub < 1e-6 && return incumbent # fail to generate a cut
        pai, pai0 = JuMP.value.(pai), JuMP.value(pai0)
        @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
        rhs, cp, cp0 = Q2_ast(stage_c, t1, x0, xi1, xi2, pai, pai0)
        pai0 < 1e-4 && (cp0 = Q2(stage_c, t1 + 1, xi2, cp)) # the 2nd stage in Q2_ast is suppressed
        if true # push Q2_ast_hat
            push!(Q2_ast_hat["by_pai0"    ], pai0)
            push!(Q2_ast_hat["by_pai"     ], pai)
            push!(Q2_ast_hat["cp0"        ], cp0)
            push!(Q2_ast_hat["cp"         ], cp)
            push!(Q2_ast_hat["id"         ], length(Q2_ast_hat["id"]) + 1)
        end
        lb = rhs - x1sv' * pai - th1 * pai0
        if incumbent["lb"] < lb
            incumbent["lb"] = lb
            incumbent["pai"] .= pai
            incumbent["pai0"] = pai0
            incumbent["rhs"] = rhs
        end
        if incumbent["lb"] > (1. - .98) * ub + 1e-6 # it's pretty well to keep delta large, because it'll benefit sufficient exploring, there's no reason to stick to only one point
            incumbent["cut_gened"] = true
            return incumbent
        end
    end
end
function train_hatQ2_with_1stage_cost(c1sv, hatQ2, Q2_ast_hat, Q2_ast, stage_c, t1, x0, xi1, xi2, Q2)
    for ite in 1:typemax(Int)
        m = JumpModel() # the master problem of the 1st stage
        JuMP.@variable(m, 0. <= b1 <= 1.) # relax Int firstly
        JuMP.@variable(m, c1)
        JuMP.@variable(m, x1)
        JuMP.@variable(m, a1)
        JuMP.@variable(m, f1)
        JuMP.@constraint(m, c1 == 2. * b1 - 1.)
        JuMP.@constraint(m, x1 == x0 + xi1 + c1) # linking
        JuMP.@constraint(m, a1 >=  x1)
        JuMP.@constraint(m, a1 >= -x1)
        JuMP.@constraint(m, f1 == stage_c(t1) * a1)
        x1sv = [b1, c1, x1, a1, f1] # ⚠️ aux variable `th1` belongs to 1st stage, but is excluded here
        JuMP.@variable(m, 0. <= th1) # ⚠️ the 1st stage aux variable representing aftereffect ⚠️ mind the lower bound for th1
        for (is_inferior, cx, ct, rhs) in zip(hatQ2["is_inferior"], hatQ2["cx"], hatQ2["ct"], hatQ2["rhs"])
            is_inferior || JuMP.@constraint(m, cx' * x1sv + ct * th1 >= rhs) # ⚠️ x1sv ~ th1 relation is derived from hatQ2
        end
        JuMP.@objective(m, Min, c1sv' * x1sv + th1) # the goal
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        x1sv, th1 = JuMP.value.(x1sv), JuMP.value(th1)
        @info "▶ ite = $ite" lb_immediate_cost_1_plus_cost_ge2 = c1sv' * x1sv + th1
        algo1dict = algorithm1(Q2_ast_hat, Q2_ast, stage_c, t1, x0, xi1, xi2, Q2, x1sv, th1)
        if algo1dict["cut_gened"]
            push!(hatQ2["is_inferior"], false)
            push!(hatQ2["cx"         ], algo1dict["pai"])
            push!(hatQ2["ct"         ], algo1dict["pai0"])
            push!(hatQ2["rhs"        ], algo1dict["rhs"])
            push!(hatQ2["id"         ], length(hatQ2["id"]) + 1)
            if 1. - 1e-5 <= hatQ2["ct"][ite] + LinearAlgebra.norm(hatQ2["cx"][ite], norm_sense) <= 1. + 1e-5
                nothing
            else
                @error "cut coeff not sum to 1.0" x0 xi1 xi2 hatQ2["ct"][ite] LinearAlgebra.norm(hatQ2["cx"][ite], norm_sense)
                error()
            end
        else
            retDict = Dict(
                "ite" => ite,
                "x1sv" => x1sv,
                "th1" => th1,
                "lb" => c1sv' * x1sv + th1
            )
            @info " 🧐 no more L-cuts " ite
            @info "current feasible value: c1sv' * x1sv + Q2(x1sv) PROVIDED Q2 is accurate (i.e., without aftereffect)"
            return retDict
        end
    end
end


global_logger(ConsoleLogger(Warn))
GRB_ENV = Gurobi.Env()
norm_sense = Cdouble(1.0) # absolute (don't pass as arg) global
function stage_c(t::Int)
        beta = .9
        beta^(t-1)
end
c1sv = [0, 0, 0, 0, 1.] # related to 1st stage cost

while true
    x0 = rand(Distributions.Uniform(-10., 10.))                # related to 1st stage feas. region
    xi1 = rand(Distributions.Uniform(-10., 10.))                # related to 1st stage feas. region    
    xi2 = rand(Distributions.Uniform(-10., 10.))              # related to 2st stage feas. region
    Q2_ast_hat = Dict( # used (only) in algorithm1, need a global initialization
        "by_pai0"     => Float64[],
        "by_pai"      => Vector{Float64}[],
        "cp0"         => Float64[], # theta_1 the aux
        "cp"          => Vector{Float64}[], # first stage decision vector
        "id"          => Int[]
    )
    hatQ2 = Dict( # filled by algorithm1
        "is_inferior" => Bool[],
        "cx"          => Vector{Float64}[],
        "ct"          => Float64[],
        "rhs"         => Float64[],
        "id"          => Int[]
    )

    initialize_Q2_ast_hat(Q2_ast, Q2_ast_hat, stage_c, 1, x0, xi1, xi2, c1sv, 1.) # the last 2 param means zPI

    ret = train_hatQ2_with_1stage_cost(c1sv, hatQ2, Q2_ast_hat, Q2_ast, stage_c, 1, x0, xi1, xi2, Q2)

    gap_for_Q2_hat = Q2(stage_c, 2, xi2, ret["x1sv"]) - ret["th1"]

    @assert abs(gap_for_Q2_hat) < 1e-5
    if ret["ite"] >= 3
        @warn "ite" ret["ite"]
        @warn "" hatQ2
    end
end

using Logging
using OffsetArrays
import LinearAlgebra
import Gurobi
import JuMP
import Distributions

# [test passed] multistage linear programming: cutting plane under relaxation trainning with pure L-cut
# cost_discount_factor in stage_c = 0.9
# very fast up to num_decisions = 96
# 26/12/23

column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
function Q2(stage_c::Function, hatQ3::Dict, xi2; t2, x1sv) # ⚠️ this is not a precise function, used in bwd phase only
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
    if t2 == num_decisions
        JuMP.@objective(m, Min, f2)
    else
        x2sv = [b2, c2, x2, a2, f2]
        JuMP.@variable(m, 0. <= th2)
        for (is_inferior, cx, ct, rhs) in zip(hatQ3["is_inferior"], hatQ3["cx"], hatQ3["ct"], hatQ3["rhs"])
            is_inferior || JuMP.@constraint(m, cx' * x2sv + ct * th2 >= rhs) # ⚠️ x1sv ~ th1 relation is derived from hatQ2
        end
        JuMP.@objective(m, Min, f2 + th2) # objective
    end
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m) # this is objective value
end
function Q2_ast(stage_c::Function, hatQ3::Dict, x0, xi1, xi2; t1, pai, pai0)
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
    JuMP.@constraint(m, f1 == stage_c(t1) * a1)
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
    x1sv = [b1, c1, x1, a1, f1] # ⚠️
    if t1 + 1 == num_decisions
        JuMP.@objective(m, Min, pai' * x1sv + pai0 * f2)
    else
        x2sv = [b2, c2, x2, a2, f2]
        JuMP.@variable(m, 0. <= th2)
        for (is_inferior, cx, ct, rhs) in zip(hatQ3["is_inferior"], hatQ3["cx"], hatQ3["ct"], hatQ3["rhs"])
            is_inferior || JuMP.@constraint(m, cx' * x2sv + ct * th2 >= rhs) # ⚠️ x1sv ~ th1 relation is derived from hatQ2
        end
        JuMP.@objective(m, Min, pai' * x1sv + pai0 * (f2 + th2)) # ⚠️ # this obj doesn't have any relation to the obj of 1st stage
    end
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    cp = JuMP.value.(x1sv)
    if t1 + 1 == num_decisions
        cp0 = JuMP.value(f2)
    else
        cp0 = JuMP.value(f2 + th2) # ⚠️
    end
    JuMP.objective_value(m), cp, cp0
end
function initialize_Q2_ast_hat(stage_c::Function, Q2_ast::Function, hatQ3::Dict, Q2_ast_hat::Dict, x0, xi1, xi2; t1, pai, pai0)
    if isempty(Q2_ast_hat["cp0"])
        _, cp, cp0 = Q2_ast(stage_c, hatQ3, x0, xi1, xi2; t1 = t1, pai = pai, pai0 = pai0)
        push!(Q2_ast_hat["by_pai0"    ], pai0)
        push!(Q2_ast_hat["by_pai"     ], pai)
        push!(Q2_ast_hat["cp0"        ], cp0)
        push!(Q2_ast_hat["cp"         ], cp)
        push!(Q2_ast_hat["id"         ], length(Q2_ast_hat["id"]) + 1)
    end
end
function algorithm1(stage_c::Function, Q2::Function, Q2_ast::Function, Q2_ast_hat::Dict, hatQ3::Dict, x0, xi1, xi2; t1, x1sv, th1) # this algorithm is for non-final stage
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
        pai, pai0 = JuMP.value.(pai), JuMP.value(pai0) # generate a feas. solution
        @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
        rhs, cp, cp0 = Q2_ast(stage_c, hatQ3, x0, xi1, xi2; t1 = t1, pai = pai, pai0 = pai0) # eval, accurately
        pai0 < 1e-4 && (cp0 = Q2(stage_c, hatQ3, xi2; t2 = t1 + 1, x1sv = cp))
        if true
            push!(Q2_ast_hat["by_pai0"    ], pai0)
            push!(Q2_ast_hat["by_pai"     ], pai)
            push!(Q2_ast_hat["cp0"        ], cp0)
            push!(Q2_ast_hat["cp"         ], cp)
            push!(Q2_ast_hat["id"         ], length(Q2_ast_hat["id"]) + 1)
        end
        lb = rhs - x1sv' * pai - th1 * pai0
        if incumbent["lb"] < lb
            incumbent["lb"], incumbent["pai0"], incumbent["rhs"] = lb, pai0, rhs
            incumbent["pai"] .= pai
        end
        if incumbent["lb"] > (1. - .98) * ub + 1e-6 # it's pretty well to keep delta large, because it'll benefit sufficient exploring, there's no reason to stick to only one point
            incumbent["cut_gened"] = true
            return incumbent
        end
    end
end


global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
norm_sense = Cdouble(1.0) # absolute (don't pass as arg) global
function stage_c(t::Int)
    beta = .9
    beta^(t-1)
end
function Q_ast_ini()::Dict
    Dict(
        "by_pai0"     => Float64[],
        "by_pai"      => Vector{Float64}[],
        "cp0"         => Float64[],
        "cp"          => Vector{Float64}[],
        "id"          => Int[]
    )
end
function hatQ_ini()::Dict
    Dict(
        "is_inferior" => Bool[],
        "cx"          => Vector{Float64}[],
        "ct"          => Float64[],
        "rhs"         => Float64[],
        "id"          => Int[]
    )
end

c_sv = [0, 0, 0, 0, 1.]
num_decisions = 12 # number of formal decisions
x0 = rand(Distributions.Uniform(-10., 10.))
xi = [rand(Distributions.Uniform(-10., 10.)) for _ in 1:num_decisions]
@info "beginning" x0 "xi"
@info xi

Q_ast_hat = OffsetVector([Q_ast_ini() for _ in 2:num_decisions], 2:num_decisions)
hatQ =      OffsetVector([hatQ_ini()  for _ in 2:num_decisions], 2:num_decisions)

tdi = Dict(
    "ite" => -1,
    "xdsv" => Vector{Float64}[],
    "thd" => Float64[]
)

for ite in 1:typemax(Int)
    x0v = Float64[x0]
    xdsv = Vector{Float64}[zeros(5) for _ in 1:num_decisions-1]
    thd = zeros(num_decisions-1)
    for d in 1:num_decisions-1 # fwd pass
        m = JumpModel() # the master problem of the 1st stage
        JuMP.@variable(m, 0. <= b1 <= 1.)
        JuMP.@variable(m, c1)
        JuMP.@variable(m, x1)
        JuMP.@variable(m, a1)
        JuMP.@variable(m, f1)
        JuMP.@constraint(m, c1 == 2. * b1 - 1.)
        JuMP.@constraint(m, x1 == x0v[end] + xi[d] + c1) # linking
        JuMP.@constraint(m, a1 >=  x1)
        JuMP.@constraint(m, a1 >= -x1)
        JuMP.@constraint(m, f1 == stage_c(d) * a1) # time here
        x1sv = [b1, c1, x1, a1, f1]
        JuMP.@variable(m, 0. <= th1) # ⚠️ mind the lower bound for th1
        lD = hatQ[d+1]
        for (is_inferior, cx, ct, rhs) in zip(lD["is_inferior"], lD["cx"], lD["ct"], lD["rhs"])
            is_inferior || JuMP.@constraint(m, cx' * x1sv + ct * th1 >= rhs) # ⚠️ x1sv ~ th1 relation is derived from hatQ2
        end
        JuMP.@objective(m, Min, c_sv' * x1sv + th1) # the goal
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        xdsv[d], thd[d], x0v[end] = JuMP.value.(x1sv), JuMP.value(th1), JuMP.value(x1)
    end
    # fwd pass above, bwd pass below
    bv = falses(num_decisions-1)
    for d in num_decisions-1:-1:1 # bwd pass, range is the reverse of fwd
        d != num_decisions-1 && bv[d+1] && (Q_ast_hat[d+1] = Q_ast_ini())
        initialize_Q2_ast_hat(
            stage_c, 
            Q2_ast,
            (d == num_decisions - 1 ? Dict() : hatQ[d+2]), 
            Q_ast_hat[d+1],
            (d == 1 ? x0 : xdsv[d-1][3]),
            xi[d],
            xi[d+1]; 
            t1 = d, 
            pai = c_sv, 
            pai0 = 1.
        )
        algo1dict = algorithm1(
            stage_c,
            Q2,
            Q2_ast,
            Q_ast_hat[d+1]::Dict,
            (d == num_decisions - 1 ? Dict() : hatQ[d+2]),
            (d == 1 ? x0 : xdsv[d-1][3]),
            xi[d],
            xi[d+1];
            t1 = d,
            x1sv = xdsv[d],
            th1 = thd[d]
        )
        if algo1dict["cut_gened"]
            push!(hatQ[d+1]["is_inferior"], false)
            push!(hatQ[d+1]["cx"         ], algo1dict["pai"])
            push!(hatQ[d+1]["ct"         ], algo1dict["pai0"])
            push!(hatQ[d+1]["rhs"        ], algo1dict["rhs"])
            push!(hatQ[d+1]["id"         ], length(hatQ[d+1]["id"]) + 1)
            @assert isapprox(hatQ[d+1]["ct"][ite] + LinearAlgebra.norm(hatQ[d+1]["cx"][ite], norm_sense), 1.; atol = 1e-6) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
            bv[d] = true
        end
    end
    if !any(bv)
        tdi["xdsv"], tdi["thd"], tdi["ite"] = xdsv, thd, ite
        break
    end
end


# test if all stages's under relaxation theta is tight 
for d in 1:num_decisions-1
    @assert isapprox((
        d == num_decisions-1 ?
        Q2(stage_c, Dict(), xi[d+1]; t2 = d+1, x1sv = tdi["xdsv"][d]) :
        c_sv' * tdi["xdsv"][d+1] + tdi["thd"][d+1]
    ),  tdi["thd"][d]; atol = 1e-6) " $(((
        d == num_decisions-1 ?
        Q2(stage_c, Dict(), xi[d+1]; t2 = d+1, x1sv = tdi["xdsv"][d]) :
        c_sv' * tdi["xdsv"][d+1] + tdi["thd"][d+1]
    )) - (tdi["thd"][d])) "
end

# log info to give more evidence
for d in 1:num_decisions-1
    @info "cmp $d" val1 = tdi["thd"][d] val2 = (
        d == num_decisions-1 ?
        Q2(stage_c, Dict(), xi[d+1]; t2 = d+1, x1sv = tdi["xdsv"][d]) :
        c_sv' * tdi["xdsv"][d+1] + tdi["thd"][d+1]
    )
end
@info "end of file" ite=tdi["ite"]


# To test this file, run the following code in REPL, if it doesn't terminate with error, then the test is passed.
# while true
#     include("src/dev_g3.jl")
# end

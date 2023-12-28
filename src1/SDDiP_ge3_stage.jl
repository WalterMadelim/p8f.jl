using Logging
using OffsetArrays
import LinearAlgebra
import Gurobi
import JuMP
import Distributions

# (>=3)-stage SDDiP with iterated Lag-cutting planes
# ⚠️ specify cbv according to data firstly
# 28/12/23

function get_anchors(cbv, bias)
    step = minimum(cbv)
    interval_length = sum(cbv)
    anchors = bias:step:bias+interval_length
end
function bv2f(bitVector, cbv=cbv, bias=bias)
    cbv' * bitVector + bias
end
function f2bv(anchors, x, ind, n_bits) # inner function
    for (i,e) in enumerate(Iterators.product([[0,1] for _ in 1:n_bits]...))        
        i == ind && return e
    end
end
function f2bv(x, anchors=anchors)
    @assert minimum(anchors) <= x <= maximum(anchors) "float input out of range"
    err, ind = findmin(abs.(x .- anchors))
    n_bits = Int(log2(length(anchors)))
    e = f2bv(anchors, x, ind, n_bits)
    bitVector = zeros(n_bits)
    bitVector .= e
    return bitVector
end
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
norm_sense = Cdouble(1.0)

# ⚠️ Before decide anchors, we need to know the bounds of x's
# x1 in [-1.1322946198963155, 0.8677053801036845]
# x2 in [3.953541798331047, 7.953541798331047]
# x3 in [12.594430370291727, 18.594430370291725]
cbv = 2. .^ (-1:4) # isa Vector{Float64}, coefficients of bitVector
TOL = minimum(cbv)/2
const bias = -10.
anchors = collect(get_anchors(cbv, bias));

num_decisions = 5 # number of formal decisions
# x0 = rand(Distributions.Uniform(-4., 4.))
xi = [rand(Distributions.Uniform(-4., 4.)) for _ in 1:num_decisions]
x0 = 3.563022746374866
x0b = f2bv(x0)
# xi = [-7.695317366271182, 6.0858364182273625, 9.64088857196068]


function Q2_kernel(t2, hatQ3, x1b_) # 📄the precise eval only when t2 == num_decisions
    m = JumpModel()
    # 1, `x1b` is a stage-2 local variable, meant to be the copy variable
    # 2, `x1b` is relaxed to in [0,1], because we allow fractional states
    JuMP.@variable(m, 0. <= x1b[i = eachindex(x1b_)] <= 1.)
    JuMP.@constraint(m, [i = eachindex(x1b)], x1b[i] == x1b_[i]) # 💡 copy constr
    JuMP.@variable(m, x2b[eachindex(x1b)], Bin) # ✏️ stage-2 state
    JuMP.@variable(m, cb2, Bin) # action, absolute Binary
    JuMP.@variable(m, a2)
    x2 = bv2f(x2b) # the float number state
    x1_but_local_in_2 = bv2f(x1b)
    chain_err_2 = x1_but_local_in_2 + (2. * cb2 - 1.) + xi[t2] - x2
    JuMP.@constraint(m, chain_err_2 <= TOL)
    JuMP.@constraint(m, -TOL <= chain_err_2)
    JuMP.@constraint(m, a2 >=  x2)
    JuMP.@constraint(m, a2 >= -x2)
    f2 = stage_c(t2) * a2
    if t2 == num_decisions
        JuMP.@objective(m, Min, f2)
    else # consider aftereffects
        JuMP.@variable(m, 0. <= th2)
        lD = hatQ3
        for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"])
            JuMP.@constraint(m, cx' * x2b + ct * th2 >= rhs)
        end
        JuMP.@objective(m, Min, f2 + th2)
    end
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    JuMP.objective_value(m), JuMP.value.(x2b)
end
function Q2(t2, hatQ3, x1b_)
    Q2_kernel(t2, hatQ3, x1b_)[1]
end
function Q2_ast(t2, hatQ3, pai, pai0) # 📄convention: this function is to generate x1-θ1 relation, thus subscript with 2
    m = JumpModel()
    # 1, `x1b` is a stage-2 local variable, meant to be the copy variable
    # 2, `x1b` is relaxed to in [0,1], because we allow fractional states
    # 3, `x1b` is ranged over the feasible region of the last period decision, thus is paired with `pai` in obj
    JuMP.@variable(m, 0. <= x1b[i = eachindex(cbv)] <= 1.)
    JuMP.@variable(m, x2b[eachindex(x1b)], Bin)
    JuMP.@variable(m, cb2, Bin)
    JuMP.@variable(m, a2)
    x2 = bv2f(x2b)
    x1_but_local_in_2 = bv2f(x1b)
    chain_err_2 = x1_but_local_in_2 + (2. * cb2 - 1.) + xi[t2] - x2
    JuMP.@constraint(m, chain_err_2 <= TOL)
    JuMP.@constraint(m, -TOL <= chain_err_2)
    JuMP.@constraint(m, a2 >=  x2)
    JuMP.@constraint(m, a2 >= -x2)
    f2 = stage_c(t2) * a2
    JuMP.@variable(m, th1) # ✏️ aux variable
    if t2 == num_decisions
        JuMP.@constraint(m, th1 >= f2) # ✏️ for any extended trail that is over the epi( Q2(⋅) )
    else # consider aftereffects
        JuMP.@variable(m, 0. <= th2)
        lD = hatQ3
        for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"])
            JuMP.@constraint(m, cx' * x2b + ct * th2 >= rhs)
        end
        JuMP.@constraint(m, th1 >= f2 + th2)
    end
    JuMP.@objective(m, Min, pai' * x1b + pai0 * th1) # ⚠️ here, x1b is deemed any decision from the last period
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
    precise_Q_ast_val, cp, cp0 = JuMP.objective_value(m), JuMP.value.(x1b), JuMP.value(th1)
end
function algorithm1(t1, hatQ3, x1b, th1) #📄convention: we pass time t1 to generate `x1 - θ1` which is stored in hatQ2
    t2 = t1 + 1
    m = JumpModel()
    JuMP.@variable(m, 0. <= pai0)
    JuMP.@variable(m, pai[eachindex(x1b)])
    JuMP.@variable(m, n1_pai)
    JuMP.@constraint(m, pai0 + n1_pai == 1.) # take == initially  
    errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(length(pai)), column.(pai), norm_sense)
    @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
    JuMP.@objective(m, Max, - x1b' * pai - th1 * pai0)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    pai, pai0 = JuMP.value.(pai), JuMP.value(pai0) # generate a feas. solution
    @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
    tmp = [pai, pai0]
    rhs, cp, cp0 = Q2_ast(t2, hatQ3, pai, pai0) # ✏️ genereate initial cut coeff for phi
    m = JumpModel()
    JuMP.@variable(m, 0. <= pai0)
    JuMP.@variable(m, pai[eachindex(x1b)])
    JuMP.@variable(m, n1_pai)
    JuMP.@constraint(m, pai0 + n1_pai <= 1.)  
    errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(length(pai)), column.(pai), norm_sense)
    @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
    JuMP.@variable(m, phi)
    JuMP.@constraint(m, phi <= cp' * pai + cp0 * pai0)
    JuMP.@objective(m, Max, phi - x1b' * pai - th1 * pai0) # phi is an overestimate of rhs
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))" # ✏️ if this test passed, we finish initialization
    Q2_ast_hat = Dict( # inner data struct
        "by_pai0"     => Float64[tmp[2]],
        "by_pai"      => Vector{Float64}[tmp[1]],
        "cp0"         => Float64[cp0],
        "cp"          => Vector{Float64}[cp],
        "id"          => Int[1]
    )
    incumbent = Dict(
        "lb" => -Inf,
        "pai" => Float64[NaN for _ in eachindex(x1b)],
        "pai0" => NaN,
        "rhs" => NaN,
        "cut_gened" => false
    )
    while true
        if true
            m = JumpModel()
            JuMP.@variable(m, 0. <= pai0)
            JuMP.@variable(m, pai[eachindex(x1b)])
            JuMP.@variable(m, n1_pai)
            JuMP.@constraint(m, pai0 + n1_pai <= 1.)    
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(length(pai)), column.(pai), norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@variable(m, phi)
        end
        JuMP.@objective(m, Max, phi - x1b' * pai - th1 * pai0) # phi is an overestimate of rhs
        for (cp, cp0) in zip(Q2_ast_hat["cp"], Q2_ast_hat["cp0"]) # >= 1 cut generated at the entrance
            JuMP.@constraint(m, phi <= cp' * pai + cp0 * pai0)
        end
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        ub = JuMP.objective_value(m) # so called objBound
        ub < 1e-6 && return incumbent # fail to generate a cut
        pai, pai0 = JuMP.value.(pai), JuMP.value(pai0) # generate a feas. solution
        @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
        rhs, cp, cp0 = Q2_ast(t2, hatQ3, pai, pai0) # eval, accurately
        pai0 < 1e-4 && (cp0 = Q2(t2, hatQ3, cp))
        if true
            push!(Q2_ast_hat["by_pai0"    ], pai0)
            push!(Q2_ast_hat["by_pai"     ], pai)
            push!(Q2_ast_hat["cp0"        ], cp0)
            push!(Q2_ast_hat["cp"         ], cp)
            push!(Q2_ast_hat["id"         ], length(Q2_ast_hat["id"]) + 1)
        end
        lb = rhs - x1b' * pai - th1 * pai0
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
function stage_c(t::Int)
    beta = .9
    beta^(t-1)
end
function hatQ_ini()::Dict
    Dict(
        "cx"  => Vector{Float64}[],
        "ct"  => Float64[],
        "rhs" => Float64[],
        "id"  => Int[]
    )
end

# 💡💡💡 first: try direct train with Int restrictions
# if the results is not pleasant, consider relaxed trials
imcost =             [NaN          for _ in 1:num_decisions]              
xbv =                [similar(x0b) for _ in 1:num_decisions-1]
th =                 [NaN          for _ in 1:num_decisions-1]
hatQ = OffsetVector( [hatQ_ini()   for _ in 1:num_decisions-1], 2:num_decisions )
x0btmp = similar(x0b)
for ite in 1:1000000
    x0btmp .= x0b
    for d in 1:num_decisions-1
        m = JumpModel()
        JuMP.@variable(m, 0. <= x1b[i = eachindex(x0b)] <= 1.)
        JuMP.@constraint(m, [i = eachindex(x0b)], x1b[i] == x0btmp[i]) # 💡 copy constr
        JuMP.@variable(m, 0. <= x2b[eachindex(x0b)] <= 1., Int) # ⚠️ although it should be Int, we relax when generating trail points
        JuMP.@variable(m, cb2, Bin)
        JuMP.@variable(m, a2)
        x2 = bv2f(x2b)
        x1_but_local_in_2 = bv2f(x1b)
        chain_err_2 = x1_but_local_in_2 + (2. * cb2 - 1.) + xi[d] - x2 # 💡 xi[t]
        JuMP.@constraint(m, chain_err_2 <= TOL)
        JuMP.@constraint(m, -TOL <= chain_err_2)
        JuMP.@constraint(m, a2 >=  x2)
        JuMP.@constraint(m, a2 >= -x2)
        f2 = stage_c(d) * a2 # 💡 stage cost 
        JuMP.@variable(m, th2 >= 0.)
        JuMP.@objective(m, Min, f2 + th2)
        lD = hatQ[d+1] # 💡 use the correct cut dict
        for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"])
            JuMP.@constraint(m, cx' * x2b + ct * th2 >= rhs)
        end
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        imcost[d] = JuMP.value(f2)
        xbv[d] .= JuMP.value.(x2b)
        th[d] = JuMP.value(th2)
        x0btmp .= xbv[d] # 💡 update
    end
    d = num_decisions
    imcost[d] = Q2(d, Dict(), x0btmp) # precise eval at last stage
    @info "action in float" x=[ bv2f(xbv[d]) for d in 1:num_decisions-1 ]
    upper=sum(imcost[2:end])
    lower = th[1]
    @assert lower <= upper + 1e-6
    @info "th1 gap" upper lower cf_TOL=TOL
    @info "imcumbent cost" global_ub=sum(imcost) imcost
    bv = falses(num_decisions-1)
    for d in num_decisions-1:-1:1
        algo1dict = algorithm1(d, (d+1 == num_decisions ? Dict() : hatQ[d+2]), xbv[d], th[d]) # ⚠️ notice this relation
        if algo1dict["cut_gened"]
            @assert isapprox(algo1dict["pai0"] + LinearAlgebra.norm(algo1dict["pai"], norm_sense), 1.; atol = 1e-6) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
            push!(hatQ[d+1]["cx"], algo1dict["pai"])
            push!(hatQ[d+1]["ct"], algo1dict["pai0"])
            push!(hatQ[d+1]["rhs"], algo1dict["rhs"])
            push!(hatQ[d+1]["id"], length(hatQ[d+1]["id"]) + 1) # after push in this ite, we can immediately utilize it in the next ite 
            bv[d] = true
        end
    end
    if !any(bv)
        @info "saturation of cuts" cut_num = [length(di["id"]) for di in hatQ]
        @info "Terminating with gap" ub=sum(imcost) lb=imcost[1]+th[1] gap=sum(imcost)-(imcost[1]+th[1]) cf_TOL=TOL
        @info "best decision at 1-stage" xbv[1]
        println("The full decision chain, with the last action omitted")
        @info xbv
        break # leave training
    end
end
imcost[num_decisions], x0btmp = Q2_kernel(num_decisions, Dict(), x0btmp) # precise eval at last stage
@info "action in float" x=[[ bv2f(xbv[d]) for d in 1:num_decisions-1 ]; bv2f(x0btmp)]
println("full action chain")
@info [xbv; [x0btmp]]
@info "Integer imcumbent cost" global_ub=sum(imcost) imcost





# # 💡💡 Integer Application
# x0btmp .= x0b
# for d in 1:num_decisions-1
#     m = JumpModel()
#     JuMP.@variable(m, 0. <= x1b[i = eachindex(x0b)] <= 1.)
#     JuMP.@constraint(m, [i = eachindex(x0b)], x1b[i] == x0btmp[i]) # 💡 copy constr
#     JuMP.@variable(m, 0. <= x2b[eachindex(x0b)] <= 1., Int) # ⚠️⚠️ 
#     JuMP.@variable(m, cb2, Bin)
#     JuMP.@variable(m, a2)
#     x2 = bv2f(x2b)
#     x1_but_local_in_2 = bv2f(x1b)
#     chain_err_2 = x1_but_local_in_2 + (2. * cb2 - 1.) + xi[d] - x2 # 💡 xi[t]
#     JuMP.@constraint(m, chain_err_2 <= TOL)
#     JuMP.@constraint(m, -TOL <= chain_err_2)
#     JuMP.@constraint(m, a2 >=  x2)
#     JuMP.@constraint(m, a2 >= -x2)
#     f2 = stage_c(d) * a2 # 💡 stage cost 
#     JuMP.@variable(m, th2 >= 0.)
#     JuMP.@objective(m, Min, f2 + th2)
#     lD = hatQ[d+1] # 💡 use the correct cut dict
#     for (cx, ct, rhs) in zip(lD["cx"], lD["ct"], lD["rhs"])
#         JuMP.@constraint(m, cx' * x2b + ct * th2 >= rhs)
#     end
#     JuMP.optimize!(m)
#     @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
#     imcost[d] = JuMP.value(f2)
#     xbv[d] .= JuMP.value.(x2b)
#     th[d] = JuMP.value(th2)
#     x0btmp .= xbv[d] # 💡 update
# end



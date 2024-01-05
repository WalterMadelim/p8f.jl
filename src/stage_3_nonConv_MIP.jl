using CairoMakie
using Logging
using OffsetArrays
import LinearAlgebra
import Gurobi
import JuMP
import Distributions

# test results
# ┌ Info: cnt ▶ 2
# │   ub = 7.847689827427952
# │   lb = 5.633410155824562
# └   gap = 0.2821568793231869
# [ Info: 📘 break due to saturation of Lag-cuts
# ┌ Info: cnt ▶ 115
# │   ub = 6.925228378650781
# │   lb = 6.924698971429888
# └   gap = 7.644617504959864e-5
# ┌ Info: 😊 gap <= 0.01%, check solutions
# │   x1_candidate = 2.754952949908123
# └   val_candidate = 6.925228378650781
# julia> (val_candidate_by_our_algorithm)6.925228378650781 - (Gurobi_large_scale_program_ref)6.9252259523428705
# 2.426307910141645e-6

column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end
function cut_init()::Dict # 1 content per stage
    Dict(
        "lip" => Dict(
            "xcenter"  => Float64[],
            "pai"  => Float64[],
            "pai0"  => Float64[],
            "rhs" => Float64[],
            "id"  => Int[]
        ),
        "lag" => Dict(
            "pai" => Float64[],
            "pai0" => Float64[],
            "rhs" => Float64[],
            "id" => Int[],
        )
    )
end
function stage_c(t::Int)
    beta = .9
    beta^(t-1)
end
function v1(x1::Float64) # the immediate cost of 1st-stage
    1.3 * abs(x1 - 3.4)
end
function v_opt(xi = xi[2:end]) # the real optimal (val = 1.48338964314651, x1 = 1.2851074717115383)
    # use Gurobi-large-scale-MILP-brute-force
    t2 = 2
    len = length(xi)
    m = JumpModel()
    JuMP.@variable(m, a1)
    JuMP.@variable(m, x[t2-1:t2+len-1])
    JuMP.@constraint(m, a1 >= x[begin] - 3.4)
    JuMP.@constraint(m, a1 >= 3.4 - x[begin])
    JuMP.@variable(m, b[t2:t2+len-1], Bin)
    JuMP.@variable(m, c[t2:t2+len-1])
    JuMP.@variable(m, a[t2:t2+len-1])
    JuMP.@variable(m, f[t2:t2+len-1])
    JuMP.@constraint(m, [d=t2:t2+len-1], x[d] == x[d-1] + xi[1+d-t2] + c[d])
    JuMP.@constraint(m, [d=t2:t2+len-1], c[d] == 2. * b[d] - 1.)
    JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >=  x[d])
    JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >= -x[d])
    JuMP.@constraint(m, [d=t2:t2+len-1], f[d] == stage_c(d) * a[d])
    JuMP.@objective(m, Min, 1.3 * a1 + sum(f))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(x[begin]), JuMP.objective_value(m) # this is objective value
end
function Q2(t2::Int, xi::Vector{Float64}, x1::Float64) # this is a cheat multi-stage value function
    len = length(xi)
    m = JumpModel()
    JuMP.@variable(m, x[t2-1:t2+len-1])
    JuMP.fix(x[t2-1], x1)
    JuMP.@variable(m, b[t2:t2+len-1], Bin)
    JuMP.@variable(m, c[t2:t2+len-1])
    JuMP.@variable(m, a[t2:t2+len-1])
    JuMP.@variable(m, f[t2:t2+len-1])
    JuMP.@constraint(m, [d=t2:t2+len-1], x[d] == x[d-1] + xi[1+d-t2] + c[d])
    JuMP.@constraint(m, [d=t2:t2+len-1], c[d] == 2. * b[d] - 1.)
    JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >=  x[d])
    JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >= -x[d])
    JuMP.@constraint(m, [d=t2:t2+len-1], f[d] == stage_c(d) * a[d])
    JuMP.@objective(m, Min, sum(f))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.objective_value(m) # this is objective value
end
if true
    function Q3(x2) # this is the ending precise value function
        Q2(3, xi[3:end], x2)
    end
    function Q3_ast(pai::Float64, pai0::Float64)
        t2 = 3
        xi_local = xi[3:end]
        m = JumpModel()
        # first stage variable and constraints
        JuMP.@variable(m, -20. <= x2 <= 20.)
        # second stage
        len = length(xi_local)
        JuMP.@variable(m, x[t2-1:t2+len-1])
        JuMP.@constraint(m, x[t2-1] == x2)
        JuMP.@variable(m, b[t2:t2+len-1], Bin)
        JuMP.@variable(m, c[t2:t2+len-1])
        JuMP.@variable(m, a[t2:t2+len-1])
        JuMP.@variable(m, f[t2:t2+len-1])
        JuMP.@constraint(m, [d=t2:t2+len-1], x[d] == x[d-1] + xi_local[1+d-t2] + c[d])
        JuMP.@constraint(m, [d=t2:t2+len-1], c[d] == 2. * b[d] - 1.)
        JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >=  x[d])
        JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >= -x[d])
        JuMP.@constraint(m, [d=t2:t2+len-1], f[d] == stage_c(d) * a[d])
        o_3 = sum(f)
        JuMP.@objective(m, Min, pai * x2 + pai0 * o_3)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        JuMP.objective_value(m)
        precise_Q_ast_val, cp, cp0 = JuMP.objective_value(m), JuMP.value(x2), JuMP.value(o_3)
    end
    function Q3_ast(rho::Float64, xcenter::Float64, pai::Float64, pai0::Float64)
        t2 = 3
        xi_local = xi[3:end]
        m = JumpModel()
        # first stage variable and constraints
        JuMP.@variable(m, -20. <= x2 <= 20.)
        # second stage
        len = length(xi_local)
        JuMP.@variable(m, x[t2-1:t2+len-1])
        JuMP.@constraint(m, x[t2-1] == x2)
        JuMP.@variable(m, b[t2:t2+len-1], Bin)
        JuMP.@variable(m, c[t2:t2+len-1])
        JuMP.@variable(m, a[t2:t2+len-1])
        JuMP.@variable(m, f[t2:t2+len-1])
        JuMP.@constraint(m, [d=t2:t2+len-1], x[d] == x[d-1] + xi_local[1+d-t2] + c[d])
        JuMP.@constraint(m, [d=t2:t2+len-1], c[d] == 2. * b[d] - 1.)
        JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >=  x[d])
        JuMP.@constraint(m, [d=t2:t2+len-1], a[d] >= -x[d])
        JuMP.@constraint(m, [d=t2:t2+len-1], f[d] == stage_c(d) * a[d])
        o_3 = sum(f)
        JuMP.@variable(m, a2)
        JuMP.@constraint(m, a2 >= x2 - xcenter)
        JuMP.@constraint(m, a2 >= xcenter - x2)
        JuMP.@objective(m, Min, pai * x2 + pai0 * (o_3 + rho * a2))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        JuMP.objective_value(m)
        precise_Q_ast_val, cp, cp0 = JuMP.objective_value(m), JuMP.value(x2), JuMP.value(o_3)
    end
    function Q2(x1) # this is not ending stage, thus a surrogate itself
        m = JumpModel()
        JuMP.@variable(m, x2)
        JuMP.@variable(m, a)
        JuMP.@constraint(m, a >=  x2)
        JuMP.@constraint(m, a >= -x2)
        JuMP.@variable(m, b, Bin)
        JuMP.@constraint(m, x2 == x1 + xi[2] + (2. * b - 1.))
        f2 = stage_c(2) * a
        # aftereffects
        JuMP.@variable(m, th2 >= 0.)
        lD = x_th_dv[2]["lag"]
        for (cx, ct, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
            JuMP.@constraint(m, cx * x2 + ct * th2 >= rhs)
        end
        lD = x_th_dv[2]["lip"]
        tmpl = length(lD["id"])
        JuMP.@variable(m, cx2[1:tmpl])
        JuMP.@variable(m, n1_cx2[1:tmpl])
        for (xcenter, pai, pai0, rhs, i) in zip(lD["xcenter"], lD["pai"], lD["pai0"], lD["rhs"], lD["id"])
            JuMP.@constraint(m, cx2[i] == x2 - xcenter)
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_cx2[i]), Cint(1), [column(cx2[i])], norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@constraint(m, pai0 * (th2 + rho * n1_cx2[i]) + pai * x2 >= rhs)
        end
        # objective part
        JuMP.@objective(m, Min, f2 + th2)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        JuMP.objective_value(m)
    end
    function Q2_ast(pai::Float64, pai0::Float64)
        m = JumpModel()
        # first stage variable and constraints
        JuMP.@variable(m, -20. <= x1 <= 20.)
        # second stage
        JuMP.@variable(m, x)
        JuMP.@variable(m, a)
        JuMP.@constraint(m, a >=  x)
        JuMP.@constraint(m, a >= -x)
        JuMP.@variable(m, b, Bin)
        JuMP.@constraint(m, x == x1 + xi[2] + (2. * b - 1.))
        f2 = stage_c(2) * a
        # aftereffects
        JuMP.@variable(m, th2 >= 0.)
        lD = x_th_dv[2]["lag"]
        for (cx, ct, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
            JuMP.@constraint(m, cx * x + ct * th2 >= rhs)
        end
        # objective part
        JuMP.@variable(m, o_2)
        JuMP.@constraint(m, o_2 >= f2 + th2)
        JuMP.@objective(m, Min, pai * x1 + pai0 * o_2)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        precise_Q_ast_val, cp, cp0 = JuMP.objective_value(m), JuMP.value(x1), JuMP.value(o_2)
    end
    function Q2_ast(rho::Float64, xcenter::Float64, pai::Float64, pai0::Float64)
        m = JumpModel()
        # first stage variable and constraints
        JuMP.@variable(m, -20. <= x1 <= 20.)
        # second stage
        JuMP.@variable(m, x2)
        JuMP.@variable(m, a)
        JuMP.@constraint(m, a >=  x2)
        JuMP.@constraint(m, a >= -x2)
        JuMP.@variable(m, b, Bin)
        JuMP.@constraint(m, x2 == x1 + xi[2] + (2. * b - 1.))
        f2 = stage_c(2) * a
        # aftereffects
        JuMP.@variable(m, th2 >= 0.)
        lD = x_th_dv[2]["lag"]
        for (cx, ct, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
            JuMP.@constraint(m, cx * x2 + ct * th2 >= rhs)
        end
        lD = x_th_dv[2]["lip"]
        tmpl = length(lD["id"])
        JuMP.@variable(m, cx2[1:tmpl])
        JuMP.@variable(m, n1_cx2[1:tmpl])
        for (xcenter, pai, pai0, rhs, i) in zip(lD["xcenter"], lD["pai"], lD["pai0"], lD["rhs"], lD["id"])
            JuMP.@constraint(m, cx2[i] == x2 - xcenter)
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_cx2[i]), Cint(1), [column(cx2[i])], norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@constraint(m, pai0 * (th2 + rho * n1_cx2[i]) + pai * x2 >= rhs)
        end
        # objective part
        JuMP.@variable(m, o_2)
        JuMP.@constraint(m, o_2 >= f2 + th2)
        JuMP.@variable(m, a1)
        JuMP.@constraint(m, a1 >= x1 - xcenter)
        JuMP.@constraint(m, a1 >= xcenter - x1)
        JuMP.@objective(m, Min, pai * x1 + pai0 * (o_2 + rho * a1))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        precise_Q_ast_val, cp, cp0 = JuMP.objective_value(m), JuMP.value(x1), JuMP.value(o_2)
    end
    function cut_gen(Q_ast::Function, Q::Function, x1::Float64, th1::Float64)
        # to get an extreme point
        m = JumpModel()
        JuMP.@variable(m, 0. <= pai0)
        JuMP.@variable(m, pai)
        JuMP.@variable(m, n1_pai)
        JuMP.@constraint(m, pai0 + n1_pai == 1.)
        errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(1), [column(pai)], norm_sense)
        @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
        JuMP.@objective(m, Max, 0. - x1 * pai - th1 * pai0)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        pai, pai0 = JuMP.value(pai), JuMP.value(pai0)
        @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
        tmp = [pai, pai0]
        rhs, cp, cp0 = Q_ast(pai, pai0)
        pai0 < 1e-4 && (cp0 = Q(cp))
        # to ensure the success of initialization
        m = JumpModel()
        JuMP.@variable(m, 0. <= pai0)
        JuMP.@variable(m, pai)
        JuMP.@variable(m, n1_pai)
        JuMP.@constraint(m, pai0 + n1_pai <= 1.)
        errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(1), [column(pai)], norm_sense)
        @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
        JuMP.@variable(m, phi)
        JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0) # ∵ only one extreme point, initially
        JuMP.@objective(m, Max, phi - x1 * pai - th1 * pai0) # phi is an overestimate of rhs
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))" # ✏️ if this test passed, we finish initialization
        # the formal program
        Qahat = Dict(
            "by_pai0"  => Float64[tmp[2]],
            "by_pai"   => Float64[tmp[1]],
            "cp"       => Float64[cp],
            "cp0"      => Float64[cp0],
            "id"       => Int[1]
        )
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
            JuMP.@variable(m, n1_pai)
            JuMP.@constraint(m, pai0 + n1_pai <= 1.)
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(1), [column(pai)], norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@variable(m, phi)
            JuMP.@objective(m, Max, phi - x1 * pai - th1 * pai0)
            for (cp, cp0) in zip(Qahat["cp"], Qahat["cp0"])
                JuMP.@constraint(m, phi <= cp * pai + cp0 * pai0)
            end
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
            ub = JuMP.objective_value(m) # the objBound
            ub < 1e-6 && return incumbent # fail to generate a cut
            pai, pai0 = JuMP.value(pai), JuMP.value(pai0)
            @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
            rhs, cp, cp0 = Q_ast(pai, pai0)
            pai0 < 1e-4 && (cp0 = Q(cp))
            if true
                push!(Qahat["by_pai0"], pai0)
                push!(Qahat["by_pai" ], pai)
                push!(Qahat["cp"     ], cp)
                push!(Qahat["cp0"    ], cp0)
                push!(Qahat["id"     ], length(Qahat["id"]) + 1)
            end
            lb = rhs - x1 * pai - th1 * pai0
            if incumbent["lb"] < lb
                incumbent["lb"], incumbent["pai0"], incumbent["rhs"] = lb, pai0, rhs
                incumbent["pai"] = pai
            end
            if incumbent["lb"] > (1. - .98) * ub + 1e-6 # it's pretty well to keep delta large, because it'll benefit sufficient exploring, there's no reason to stick to only one point
                incumbent["cut_gened"] = true
                return incumbent
            end
        end
    end
    function cut_gen(Q_ast::Function, Q::Function, rho::Float64, x1::Float64, th1::Float64)
        xcenter = x1
        # to get an extreme point
        m = JumpModel()
        JuMP.@variable(m, 0. <= pai0)
        JuMP.@variable(m, pai)
        JuMP.@variable(m, n1_pai)
        JuMP.@constraint(m, pai0 + n1_pai == 1.)
        errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(1), [column(pai)], norm_sense)
        @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
        JuMP.@objective(m, Max, 0. - x1 * pai - th1 * pai0)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        pai, pai0 = JuMP.value(pai), JuMP.value(pai0)
        @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
        tmp = [pai, pai0]
        rhs, x_val, Qx_val = Q_ast(rho, xcenter, pai, pai0)
        pai0 < 1e-4 && (Qx_val = Q(x_val))
        # to ensure the success of initialization
        m = JumpModel()
        JuMP.@variable(m, 0. <= pai0)
        JuMP.@variable(m, pai)
        JuMP.@variable(m, n1_pai)
        JuMP.@constraint(m, pai0 + n1_pai <= 1.)
        errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(1), [column(pai)], norm_sense)
        @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
        JuMP.@variable(m, phi)
        JuMP.@constraint(m, phi <= x_val * pai + (Qx_val + rho * abs(x_val - xcenter)) * pai0) # ∵ only one extreme point, initially
        JuMP.@objective(m, Max, phi - x1 * pai - th1 * pai0) # phi is an overestimate of rhs
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))" # ✏️ if this test passed, we finish initialization
        # the formal program
        Qahat = Dict( # inner data struct
            "by_pai0"     => Float64[tmp[2]],
            "by_pai"      => Float64[tmp[1]],
            "x_val"       => Float64[x_val],
            "Qx_val"      => Float64[Qx_val],
            "id"          => Int[1]
        )
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
            JuMP.@variable(m, n1_pai)
            JuMP.@constraint(m, pai0 + n1_pai <= 1.)
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(1), [column(pai)], norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@variable(m, phi)
            JuMP.@objective(m, Max, phi - x1 * pai - th1 * pai0)
            for (x_val, Qx_val) in zip(Qahat["x_val"], Qahat["Qx_val"])
                JuMP.@constraint(m, phi <= x_val * pai + (Qx_val + rho * abs(x_val - xcenter)) * pai0)
            end
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
            ub = JuMP.objective_value(m) # the objBound
            ub < 1e-6 && return incumbent # fail to generate a cut
            pai, pai0 = JuMP.value(pai), JuMP.value(pai0)
            @assert pai0 >= 0. "Gurobi have an violation about pai0, please check"
            rhs, x_val, Qx_val = Q_ast(rho, xcenter, pai, pai0)
            pai0 < 1e-4 && (Qx_val = Q(x_val))
            if true
                push!(Qahat["by_pai0"], pai0)
                push!(Qahat["by_pai" ], pai)
                push!(Qahat["x_val"  ], x_val)
                push!(Qahat["Qx_val" ], Qx_val)
                push!(Qahat["id"     ], length(Qahat["id"]) + 1)
            end
            lb = rhs - x1 * pai - th1 * pai0
            if incumbent["lb"] < lb
                incumbent["lb"], incumbent["pai0"], incumbent["rhs"] = lb, pai0, rhs
                incumbent["pai"] = pai
            end
            if incumbent["lb"] > (1. - .98) * ub + 1e-6 # it's pretty well to keep delta large, because it'll benefit sufficient exploring, there's no reason to stick to only one point
                incumbent["cut_gened"] = true
                return incumbent
            end
        end
    end
end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
norm_sense = Cdouble(1.0)

# num_decisions = 11 # number of formal decisions
# x0 = rand(Distributions.Uniform(-5., 5.))
# xi = [rand(Distributions.Uniform(-5., 5.)) for _ in 1:num_decisions]
x0 = 1.6169779972529312
xi = [3.760961946673934, -3.3367588557691774, -0.4625729914588472, -2.3338475912995738, -0.39159770602499044, -1.696214192324227,  3.224144216292162,  3.790779412949041, -1.5488703272238502,  4.521890360278045, -3.500405403145679]

rho = 1.9
x_th_dv = [cut_init() for _ in 1:2]

x1_candidate = [NaN]
ub = [Inf, Inf]
lb = [-Inf, -Inf]
x = [NaN, NaN]
th = [NaN, NaN]
f = [NaN, NaN, NaN]
curr_cut_cnt = [0]
while true # lag-cut generating
    m = JumpModel()
    JuMP.@variable(m, -20. <= x1 <= 20.)
    JuMP.@variable(m, a1)
    JuMP.@constraint(m, a1 >= x1-3.4)
    JuMP.@constraint(m, a1 >= 3.4-x1)
    JuMP.@variable(m, th1 >= 0.)
    lD = x_th_dv[1]["lag"]
    for (pai, pai0, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
        JuMP.@constraint(m, pai * x1 + pai0 * th1 >= rhs)
    end
    f1 = 1.3 * a1
    JuMP.@objective(m, Min, f1 + th1)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    x1, th1, f1 = JuMP.value(x1), JuMP.value(th1), JuMP.value(f1)
    x[1], th[1], f[1] = x1, th1, f1
    if true
        lb[end] = JuMP.objective_value(m)
        if lb[end] > lb[begin]
            lb[begin] = lb[end]
        elseif lb[end] < lb[begin] - 1e-6  
            @warn "cnt = $(curr_cut_cnt[begin]), maybe there exists some numerical problems, but keep on trainning."
        end
    end
    m = JumpModel()
    JuMP.@variable(m, x2) # actually there's no need to bound x2, due to chaining constr
    JuMP.@variable(m, a2)
    JuMP.@constraint(m, a2 >=  x2)
    JuMP.@constraint(m, a2 >= -x2)
    JuMP.@variable(m, b, Bin)
    JuMP.@constraint(m, x2 == x1 + xi[2] + (2. * b - 1.))
    f2 = stage_c(2) * a2
    JuMP.@variable(m, th2 >= 0.)
    lD = x_th_dv[2]["lag"]
    for (pai, pai0, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
        JuMP.@constraint(m, pai * x2 + pai0 * th2 >= rhs)
    end
    JuMP.@objective(m, Min, f2 + th2)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    x2, th2, f2 = JuMP.value(x2), JuMP.value(th2), JuMP.value(f2)
    x[2], th[2], f[2] = x2, th2, f2
    f[3] = Q3(x2)
    if true
        ub[end] = sum(f)
        if ub[end] < ub[begin]
            ub[begin] = ub[end]
            x1_candidate[begin] = x[1]
        end
        gap_abs = ub[begin] - lb[begin]
        gap = gap_abs / ub[begin]
        @info "cnt ▶ $(curr_cut_cnt[begin])" ub=ub[begin] lb=lb[begin] gap=gap
        if gap < 0.01/100
            @info "😊 gap <= 0.01%, check solutions" x1_candidate=x1_candidate[begin] val_candidate=ub[begin]
            break
        end
    end
    bv = falses(2)
    t = 2
    cdict = cut_gen(Q3_ast, Q3, x[t], th[t])
    if cdict["cut_gened"]
        @assert isapprox(cdict["pai0"] + LinearAlgebra.norm(cdict["pai"], norm_sense), 1.; atol = 1e-6) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
        push!(x_th_dv[t]["lag"]["pai"], cdict["pai"])
        push!(x_th_dv[t]["lag"]["pai0"], cdict["pai0"])
        push!(x_th_dv[t]["lag"]["rhs"], cdict["rhs"])
        push!(x_th_dv[t]["lag"]["id"], length(x_th_dv[t]["lag"]["id"]) + 1)
        bv[t] = true
    end
    t = 1
    cdict = cut_gen(Q2_ast, Q2, x[t], th[t])
    if cdict["cut_gened"]
        @assert isapprox(cdict["pai0"] + LinearAlgebra.norm(cdict["pai"], norm_sense), 1.; atol = 1e-6) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
        push!(x_th_dv[t]["lag"]["pai"], cdict["pai"])
        push!(x_th_dv[t]["lag"]["pai0"], cdict["pai0"])
        push!(x_th_dv[t]["lag"]["rhs"], cdict["rhs"])
        push!(x_th_dv[t]["lag"]["id"], length(x_th_dv[t]["lag"]["id"]) + 1)
        bv[t] = true
    end
    if any(bv)
        curr_cut_cnt[begin] += 1
    else
        @info "📘 break due to saturation of Lag-cuts"
        break
    end
end

while true # lip-cut generating
    m = JumpModel()
    JuMP.@variable(m, -20. <= x1 <= 20.)
    JuMP.@variable(m, a1)
    JuMP.@constraint(m, a1 >= x1-3.4)
    JuMP.@constraint(m, a1 >= 3.4-x1)
    JuMP.@variable(m, th1 >= 0.)
    lD = x_th_dv[1]["lag"]
    for (pai, pai0, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
        JuMP.@constraint(m, pai * x1 + pai0 * th1 >= rhs)
    end
    lD = x_th_dv[1]["lip"]
    tmpl = length(lD["id"])
    JuMP.@variable(m, cx1[1:tmpl])
    JuMP.@variable(m, n1_cx1[1:tmpl])
    for (xcenter, pai, pai0, rhs, i) in zip(lD["xcenter"], lD["pai"], lD["pai0"], lD["rhs"], lD["id"])
        JuMP.@constraint(m, cx1[i] == x1 - xcenter)
        errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_cx1[i]), Cint(1), [column(cx1[i])], norm_sense)
        @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
        JuMP.@constraint(m, pai0 * (th1 + rho * n1_cx1[i]) + pai * x1 >= rhs)
    end
    f1 = 1.3 * a1
    JuMP.@objective(m, Min, f1 + th1)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    x1, th1, f1 = JuMP.value(x1), JuMP.value(th1), JuMP.value(f1)
    x[1], th[1], f[1] = x1, th1, f1
    if true
        lb[end] = JuMP.objective_value(m)
        if lb[end] > lb[begin]
            lb[begin] = lb[end]
        elseif lb[end] < lb[begin] - 1e-6  
            @warn "cnt = $(curr_cut_cnt[begin]), maybe there exists some numerical problems, but keep on trainning."
        end
    end
    m = JumpModel()
    JuMP.@variable(m, x2) # actually there's no need to bound x2, due to chaining constr
    JuMP.@variable(m, a2)
    JuMP.@constraint(m, a2 >=  x2)
    JuMP.@constraint(m, a2 >= -x2)
    JuMP.@variable(m, b, Bin)
    JuMP.@constraint(m, x2 == x1 + xi[2] + (2. * b - 1.))
    f2 = stage_c(2) * a2
    JuMP.@variable(m, th2 >= 0.)
    lD = x_th_dv[2]["lag"]
    for (pai, pai0, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
        JuMP.@constraint(m, pai * x2 + pai0 * th2 >= rhs)
    end
    lD = x_th_dv[2]["lip"]
    tmpl = length(lD["id"])
    JuMP.@variable(m, cx2[1:tmpl])
    JuMP.@variable(m, n1_cx2[1:tmpl])
    for (xcenter, pai, pai0, rhs, i) in zip(lD["xcenter"], lD["pai"], lD["pai0"], lD["rhs"], lD["id"])
        JuMP.@constraint(m, cx2[i] == x2 - xcenter)
        errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_cx2[i]), Cint(1), [column(cx2[i])], norm_sense)
        @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
        JuMP.@constraint(m, pai0 * (th2 + rho * n1_cx2[i]) + pai * x2 >= rhs)
    end
    JuMP.@objective(m, Min, f2 + th2)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    x2, th2, f2 = JuMP.value(x2), JuMP.value(th2), JuMP.value(f2)
    x[2], th[2], f[2] = x2, th2, f2
    f[3] = Q3(x2)
    if true
        ub[end] = sum(f)
        if ub[end] < ub[begin]
            ub[begin] = ub[end]
            x1_candidate[begin] = x[1]
        end
        gap_abs = ub[begin] - lb[begin]
        gap = gap_abs / ub[begin]
        @info "cnt ▶ $(curr_cut_cnt[begin])" ub=ub[begin] lb=lb[begin] gap=gap
        if gap < 0.01/100
            @info "😊 gap <= 0.01%, check solutions" x1_candidate=x1_candidate[begin] val_candidate=ub[begin]
            break
        end
    end
    bv = falses(2)
    t = 2
    cdict = cut_gen(Q3_ast, Q3, rho, x[t], th[t])
    if cdict["cut_gened"]
        @assert isapprox(cdict["pai0"] + LinearAlgebra.norm(cdict["pai"], norm_sense), 1.; atol = 1e-6) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
        push!(x_th_dv[t]["lip"]["xcenter"], x[t])
        push!(x_th_dv[t]["lip"]["pai"], cdict["pai"])
        push!(x_th_dv[t]["lip"]["pai0"], cdict["pai0"])
        push!(x_th_dv[t]["lip"]["rhs"], cdict["rhs"])
        push!(x_th_dv[t]["lip"]["id"], length(x_th_dv[t]["lip"]["id"]) + 1)
        bv[t] = true
    end
    t = 1
    cdict = cut_gen(Q2_ast, Q2, rho, x[t], th[t])
    if cdict["cut_gened"]
        @assert isapprox(cdict["pai0"] + LinearAlgebra.norm(cdict["pai"], norm_sense), 1.; atol = 1e-6) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
        push!(x_th_dv[t]["lip"]["xcenter"], x[t])
        push!(x_th_dv[t]["lip"]["pai"], cdict["pai"])
        push!(x_th_dv[t]["lip"]["pai0"], cdict["pai0"])
        push!(x_th_dv[t]["lip"]["rhs"], cdict["rhs"])
        push!(x_th_dv[t]["lip"]["id"], length(x_th_dv[t]["lip"]["id"]) + 1)
        bv[t] = true
    end
    if any(bv)
        curr_cut_cnt[begin] += 1
    else
        @info "📕 break due to saturation of Lip-cuts"
        break
    end
end

if false # to give plots of 3-stage-nonconvex-programming
    f = Figure();
    axs = Axis.([f[i...] for i in Iterators.product([1,2],[1,2])]); # 2 * 2 subplots
    x2 = range(0., 7.; length=500);
    lines!(axs[4], x2, Q3.(x2); color = :olive) # aftereffect as a function of 2nd_stage_decisioin
    t = 2
    x1 = range(0., 7.; length=500);
    ytmp = [Q2(t, xi[t:end], s) for s in x1]; # aftereffect as a function of 1st_stage_decisioin
    lines!(axs[3], x1, ytmp; color = :olive)
    v1_bias = 5.9 # make it convenient to compare, only skewness is valid
    lines!(axs[3], [0., 3.4, 7.], v1_bias .+ v1.([0., 3.4, 7.]); color = :navy) # stage_1 immediate cost as a function of 1st_stage_decisioin
    t = 2
    x1 = range(0., 7.; length=800);
    ytmp = [v1(s) + Q2(t, xi[t:end], s) for s in x1];
    lines!(axs[2], x1, ytmp; color = :olive) # v_prim as a function of 1st_stage_decisioin
    t = 2
    x1 = range(2.4, 3.1; length=800); # a local magnification to facilitate observation
    ytmp = [v1(s) + Q2(t, xi[t:end], s) for s in x1];
    lines!(axs[1], x1, ytmp; color = :olive) # v_prim as a function of 1st_stage_decisioin
    # to mark out the large-scale-programming-brute-force optimal solution 
    opt_x, opt_v = v_opt()
    scatter!(axs[1], [opt_x], [opt_v]; color = :cyan, marker = '⋄') # this is the opt_solu and value, 4.3367 deviates from 3.4
    scatter!(axs[2], [opt_x], [opt_v]; color = :cyan, marker = '⋄') # this is the opt_solu and value, 4.3367 deviates from 3.4
end

if false
    function cut_fun(rho, xcenter, pai, pai0, rhs) # for drawing
        x -> (rhs - pai * x) / pai0 - rho * abs(x - xcenter)
    end
    function cut_check(rho, xcenter, pai, pai0, rhs, x, th)
        check = rhs - ( pai0 * (th + rho * abs(x - xcenter)) + pai * x )
        if check <= 1e-6
            println("this trial is above the cut")
        else
            println("this trial is violated, given the cut")
        end
        check
    end
end
    







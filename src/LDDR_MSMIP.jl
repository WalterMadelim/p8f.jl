import JuMP
import Gurobi
import Random
import Distributions
import Statistics
import LinearAlgebra
using Logging

# part 4.1 + 3.5 + 4.3 of LDDR-MSMIP
# first we use primal 4.1 to get a statistical UB, which is then used to bound NA-dual program at initializing phase
# we could also use EVPI to construct a statistical LB
# After solving NA dual, we get a LB that is comparable with EVPI. Results indicates that the former is superior
# We also get a dual solution upon solving NA dual, thus we derive the NA dual driven primal policy (policy_4p3)
# The comparing of primal policy is computationally demanding
# notice that it is announced in the paper that SW dual driven primal policy is better
# 28/01/24

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function get_logNormal_dist(E, s) # args is of the target distribution
    m, v = E, s^2
    Distributions.LogNormal(log((m^2)/sqrt(v+m^2)), sqrt(log(v/(m^2)+1)))
end

if true # Functions to which Lagrangian cut generation is related
    function Q2(t2, s, mat1, l_x_dict) # t2 := t+1, s in 1:S, mat1 is xm1, l_x_dict is the optimal dual weights
        D_s, Ups_s = D3[:, :, s], Ups3[:, :, s]
        m = JumpModel() # a multistage 1-scene program with penalized Obj
        JuMP.@variable(m, y[t2:T, 1:J], Bin)
        JuMP.@variable(m, 0. <= o[u = t2:T] <= o_UB[u])
        JuMP.@variable(m, x_H[t2:T, 1:J] >= 0.)
        JuMP.@variable(m, x_B[t2:T, 1:J] >= 0.)
        JuMP.@variable(m,   x[t2:T, 1:J] >= 0.)
        JuMP.@constraint(m, [u=t2:T, j=1:J], x_B[u,j] - x_H[u,j] + [1,-1,1.]' * (u == t2 ? mat1[j,:] : [x_H[u-1,j], x_B[u-1,j], x[u-1,j]]) == D_s[u,j]) # ★
        JuMP.@constraint(m, [u=t2:T], sum(T_Y[j] * y[u,j] + T_B * x[u,j] for j in 1:J) - o[u] <= time_UB[u])
        JuMP.@constraint(m, [u=t2:T, j=1:J], x[u,j] <= production_UB[j] * y[u,j])
        JuMP.@constraint(m, [u=t2:T, j=1:J], x_H[u,j] + x[u,j] <= inventory_UB[j])
        costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
        primal_obj = sum(c_O * o[u] + costs_oto(u) for u in t2:T)
        penal_with_dual_upper_vars = 0.
        if t2 <= T-1
            penal_with_dual_upper_vars = sum(
                    x[t, j] * sum(
                            l_x_dict[(t, j, u)] * ( D_s[u, j] - MU[u, j] * (1. + rhoY * rho^(u - t) * (Ups_s[t, j] - 1.)) )
                        for u in t+1:T
                    )
                for t in t2:T-1, j in 1:J
            )
        end
        JuMP.@objective(m, Min, primal_obj + penal_with_dual_upper_vars)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "in Q2: $(JuMP.termination_status(m))"
        JuMP.objective_value(m)
    end
    function Q2_ast(t2, s, l_x_dict, pai::Matrix, pai0)
        D_s, Ups_s = D3[:, :, s], Ups3[:, :, s]
        m = JumpModel()
        JuMP.@variable(m, y[t2:T, 1:J], Bin)
        JuMP.@variable(m, 0. <= o[u = t2:T] <= o_UB[u])
        # ⚠️ below 3 lines: `-1` introduce 1st stage state variables with lb
        JuMP.@variable(m, x_H[t2-1:T, 1:J] >= 0.)
        JuMP.@variable(m, x_B[t2-1:T, 1:J] >= 0.)
        JuMP.@variable(m,   x[t2-1:T, 1:J] >= 0.)
        JuMP.@constraint(m, [j=1:J], x_B[t2-1,j] <= sum(D_s[1:t2-1,j])) # ⚠️ first stage constraints
        JuMP.@constraint(m, [u=t2:T, j=1:J], x_B[u,j] - x_H[u,j] + x_H[u-1,j] - x_B[u-1,j] + x[u-1,j] == D_s[u,j]) # ★
        JuMP.@constraint(m, [u=t2:T], sum(T_Y[j] * y[u,j] + T_B * x[u,j] for j in 1:J) - o[u] <= time_UB[u])
        JuMP.@constraint(m, [u=t2:T, j=1:J], x[u,j] <= production_UB[j] * y[u,j])
        JuMP.@constraint(m, [u=t2-1:T, j=1:J], x_H[u,j] + x[u,j] <= inventory_UB[j]) # ⚠️ here `-1` introduce bounds on 1st stage state variables
        costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
        primal_obj = sum(c_O * o[u] + costs_oto(u) for u in t2:T)
        penal_with_dual_upper_vars = 0.
        if t2 <= T-1
            penal_with_dual_upper_vars = sum(
                    x[t, j] * sum(
                            l_x_dict[(t, j, u)] * ( D_s[u, j] - MU[u, j] * (1. + rhoY * rho^(u - t) * (Ups_s[t, j] - 1.)) )
                        for u in t+1:T
                    )
                for t in t2:T-1, j in 1:J
            )
        end
        o_2 = primal_obj + penal_with_dual_upper_vars
        xm1 = [x_H[t2-1, :] x_B[t2-1, :] x[t2-1, :]]
        JuMP.@objective(m, Min, sum( pai .* xm1 ) + pai0 * o_2)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "in Q2_ast: $(JuMP.termination_status(m))"
        precise_Q_ast_val, cp, cp0 = JuMP.objective_value(m), JuMP.value.(xm1), JuMP.value(o_2)
    end
    function cut_gen(Q_ast::Function, Q::Function, t, s, l_x_dict, x1::Matrix, th1::Float64)
        t2 = t + 1
        if isempty(Qahat[t,s]["id"]) # to get an extreme point of `K^s`
            m = JumpModel()
            JuMP.@variable(m, 0. <= pai0)
            JuMP.@variable(m, pai[1:J,1:3])
            JuMP.@variable(m, n1_pai)
            JuMP.@constraint(m, pai0 + n1_pai == 1.) # ⚠️ only here we use ==
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(length(pai)), column.(reshape(pai, (:,))), norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@objective(m, Max, 0. - sum(x1 .* pai) - th1 * pai0)
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "in cut_gen [1] $(JuMP.termination_status(m))"
            tmp = [JuMP.value.(pai), JuMP.value(pai0)]
            tmp[2] < 0. && (tmp[2] = 0.)
            rhs, cp, cp0 = Q_ast(t2, s, l_x_dict, tmp[1], tmp[2])
            tmp[2] < 1e-4 && (cp0 = Q(t2, s, cp, l_x_dict))
            JuMP.@variable(m, phi) # continue with the model `m`
            JuMP.@constraint(m, phi <= sum(cp .* pai) + cp0 * pai0) # ∵ only one extreme point, initially
            JuMP.@objective(m, Max, phi - sum(x1 .* pai) - th1 * pai0) # phi is an overestimate of rhs
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "in cut_gen [2] $(JuMP.termination_status(m))" # ✏️ if this test passed, we finish initialization
            push!(Qahat[t,s]["cp" ], cp)
            push!(Qahat[t,s]["cp0"], cp0)
            push!(Qahat[t,s]["id" ], length(Qahat[t,s]["id"]) + 1)
        end
        # the formal program
        incumbent = Dict(
            "ub" => Inf,
            "lb" => -Inf,
            "pai" => NaN * ones(J,3),
            "pai0" => NaN,
            "rhs" => NaN,
            "cut_gened" => false
        )
        while true
            m = JumpModel()
            JuMP.@variable(m, 0. <= pai0)
            JuMP.@variable(m, pai[1:J,1:3])
            JuMP.@variable(m, n1_pai)
            JuMP.@constraint(m, pai0 + n1_pai <= 1.)
            errcode_norm = Gurobi.GRBaddgenconstrNorm(JuMP.backend(m), "", column(n1_pai), Cint(length(pai)), column.(reshape(pai, (:,))), norm_sense)
            @assert errcode_norm == 0 "Gurobi's norm_1 constr fail"
            JuMP.@variable(m, phi)
            JuMP.@objective(m, Max, phi - sum(x1 .* pai) - th1 * pai0)
            for (cp, cp0) in zip(Qahat[t,s]["cp"], Qahat[t,s]["cp0"])
                JuMP.@constraint(m, phi <= sum(cp .* pai) + cp0 * pai0)
            end
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "in cut_gen [loop] $(JuMP.termination_status(m))"
            ub = JuMP.objective_value(m) # the objBound
            ub < incumbent["ub"] && (incumbent["ub"] = ub)
            incumbent["ub"] < 1e-3 && return incumbent # fail to generate a cut
            pai, pai0 = JuMP.value.(pai), JuMP.value(pai0) # get a feas. solu
            pai0 < 0. && (pai0 = 0.)
            rhs, cp, cp0 = Q_ast(t2, s, l_x_dict, pai, pai0) # encounter an ext point of `K^s`
            pai0 < 1e-4 && (cp0 = Q(t2, s, cp, l_x_dict))
            push!(Qahat[t,s]["cp" ], cp)
            push!(Qahat[t,s]["cp0"], cp0)
            push!(Qahat[t,s]["id" ], length(Qahat[t,s]["id"]) + 1)
            lb = rhs - sum(x1 .* pai) - th1 * pai0 # get a feas. value
            if lb > incumbent["lb"]
                incumbent["lb"], incumbent["pai0"], incumbent["rhs"] = lb, pai0, rhs
                incumbent["pai"] .= pai
            end
            if incumbent["lb"] > (1. - .98) * incumbent["ub"] + 1e-6 # it's pretty well to keep delta large, because it'll benefit sufficient exploring, there's no reason to stick to only one point
                incumbent["cut_gened"] = true
                return incumbent
            end
        end
    end
end

global_logger(ConsoleLogger(Info))
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
norm_sense = Cdouble(1.0)
GRB_ENV = Gurobi.Env()

if true # 📕 data for the program
    alpha = .05
    z_alpha = Distributions.quantile(Distributions.Normal(), 1. - alpha) # used in one-side confidence bound
    # N = 10000 raw: u = (1.0939493482904157, 0.0035990318291414638)
    # N = 10000 raw: l = (0.9048899519779566, 0.0023215053447264793)
    statistical_ub = 1.0975483801195571 # due to policy 4p1, at N = 10000
    lb_by_NA_dual = 0.9689142967808022 # at N = 5000
    statistical_lb = 0.9025684466332301 # due to EVPI, at N = 10000
    SCALE = 61888 # to alleviate numerical issues
    T, J, S = 4, 3, 500 # S is the samples taken in SAA
    MU = Float64[ # serves as the Expectation of D
        74  140 88
        64  99  105
        77  128 95
        42  120 78
    ] # This is a scaling factor
    d_epsilon = get_logNormal_dist(1., .5)
    d_delta = [get_logNormal_dist(MU[t, j], .2 * t * MU[t, j]) for t in eachindex(eachrow(MU)), j in eachindex(eachcol(MU))] # but the first line (t=1) is unused
    rho, rhoY = .6, .2
    EDj_bar = Statistics.mean.(eachcol(MU)) # indexed by j
    c_H, c_O = 15., 100.
    c_B = 30. * ones(T) # indexed by t
    c_B[T] *= 5
    c_Y = 1.2 * 4 * 15 * EDj_bar # indexed by j
    c_H, c_O, c_B, c_Y = c_H / SCALE, c_O / SCALE, c_B / SCALE, c_Y / SCALE
    T_B = 1.0
    T_Y = .25 * EDj_bar # indexed by j, in (27c)
    production_UB = 6. * EDj_bar # indexed by j, in (27d)
    inventory_UB = 10. * EDj_bar # indexed by j, RHS of (27e, 27f)
    time_UB = 1.5 * sum.(eachrow(MU)) # indexed by t, RHS of (27c)
    o_UB = .25 * time_UB # indexed by t, RHS of (27g)
    obj3a_UB = statistical_ub
    l_x_dict = Dict((1, 3, 4) => 1.0137155977993396e-5, (2, 2, 4) => 6.028446919920274e-6, (2, 3, 4) => 1.011054996349409e-5, (1, 2, 3) => -2.1744587199982812e-7, (1, 1, 2) => 7.809629676413858e-7, (3, 2, 4) => 7.697355275195144e-6, (1, 3, 3) => 8.896077643660707e-7, (1, 1, 4) => 1.3127828313518984e-5, (2, 2, 3) => 4.608947397779395e-8, (3, 3, 4) => 1.0997277056759108e-5, (2, 3, 3) => 9.520538860409112e-7, (2, 1, 4) => 1.53764650057964e-5, (1, 2, 2) => 1.8059381618977696e-6, (1, 1, 3) => -6.313658736049452e-7, (1, 3, 2) => 1.796661348124759e-6, (1, 2, 4) => 5.806376765557988e-6, (3, 1, 4) => 1.599354661023327e-5, (2, 1, 3) => 2.013670619596678e-6) # a candidate dual weight
end

function compare_4p1_and_4p3(seed::Int)
    # 📕 we start folding horizon process
    Random.seed!(seed)
    for t in 1:T-1 # the folding horizon process for policy 4p3, with code of policy 4p1 plugged in
        if t >= 2 # generate the real realized ξ_t
            for j in 1:J
                epsilon, delta = rand(d_epsilon), rand(d_delta[t,j])
                tmp = rho * Ups_s[t-1, j] + (1-rho) * epsilon
                D_s[t, j], Ups_s[t, j] = rhoY * tmp * MU[t, j] + (1-rhoY) * delta, tmp
            end
        end
        for s in 1:S # write the info that is conditioned on
            Ups3[t,:,s] .= Ups_s[t,:]
            D3[t,:,s] .= D_s[t,:]
        end
        for s in 1:S # 💡 ∀s, sample (overwrite) `info >= t+1` conditionally on `info <= t`
            for u in t+1:T, j in 1:J # `u ∈ t+1:T` indicating the "2nd stage"
                epsilon, delta = rand(d_epsilon), rand(d_delta[u,j])
                tmp = rho * Ups3[u-1, j, s] + (1-rho) * epsilon
                D3[u, j, s], Ups3[u, j, s] = rhoY * tmp * MU[u, j] + (1-rhoY) * delta, tmp
            end
        end
        for u in t+1:T, j in 1:J # 🔭 this loop is for policy 4p1, calculate 𝔼_{ξᵗ}{ r.v. > t }
            # 💡 we use the same D_s because it doesn't affect policy 4p3
            D_s[u, j] = MU[u, j] * (1. + rhoY * rho^(u-t) * (Ups_s[t, j] - 1.))
        end
        lb, ub = [-Inf, -Inf], [Inf, Inf]
        for ite in 1:typemax(Int) # iteratively solving a 2-stage SIP at stage t
            m = JumpModel() # the Benders master problem
            JuMP.@variable(m, y[1:J], Bin)
            JuMP.@variable(m, 0. <= o <= o_UB[t])
            JuMP.@variable(m, x_H[1:J] >= 0.)
            JuMP.@variable(m, x_B[1:J] >= 0.)
            JuMP.@variable(m, x[1:J] >= 0.) # these 5 lines are decisions of (25)
            JuMP.@constraint(m, sum(T_Y[j] * y[j] + T_B * x[j] for j in 1:J) - o <= time_UB[t])
            JuMP.@constraint(m, [j=1:J], x[j] <= production_UB[j] * y[j])
            JuMP.@constraint(m, [j=1:J], x_H[j] + x[j] <= inventory_UB[j])
            JuMP.@constraint(m, [j=1:J], x_B[j] - x_H[j] + (t == 1 ? 0. : [1,-1,1.]' * xm1_final[j,:] ) == D_s[t,j]) # ★ vm1 is delivered from last stage ★ RHS has been revealed at this moment
            JuMP.@variable(m, th[1:S] >= 0.)
            for s in 1:S
                lD = cut_mat[t,s]
                for (pai, pai0, rhs) in zip(lD["pai"], lD["pai0"], lD["rhs"])
                    JuMP.@constraint(m, sum(pai .* [x_H x_B x]) + pai0 * th[s] >= rhs)
                end
            end
            immediate_cost = c_O * o + sum(c_B[t] * x_B[j] + c_H * x_H[j] + c_Y[j] * y[j] for j in 1:J)
            after_cost = sum(1/S * th[s] for s in 1:S)
            JuMP.@objective(m, Min, immediate_cost + after_cost) # will derive an ObjBound
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "in `ite` = ($ite) $(JuMP.termination_status(m))"
            state_trial = JuMP.value.([x_H x_B x]) # 💡 reshape([x_H x_B x], (:,)) to turn it into an vector 💡 reshape(ans, (:,3)) to recover
            th_s_trial = JuMP.value.(th)
            cost_1 = JuMP.value(immediate_cost)
            cost_2 = sum(1/S * Q2(t+1, s, state_trial, l_x_dict) for s in 1:S)
            ub[end], lb[end] = cost_1 + cost_2, cost_1 + JuMP.value(after_cost)
            lb[end] > lb[begin] && (lb[begin] = lb[end])
            if ub[end] < ub[begin] # current decision is better
                ub[begin] = ub[end]
                cost_4p3[t] = cost_1
                xm1_holder .= state_trial
            end
            if true # concluding session for the 2-stage iterative solving process
                ObjValue, ObjBound = ub[begin], lb[begin]
                rel_gap = abs((ObjBound - ObjValue) / ObjValue)
                @info "▶ t = $t // ite = $(ite)" ObjValue ObjBound rel_gap
                if rel_gap < 3 / 100
                    @info "😊 2-stage problem convergent."
                    break
                end
            end
            bv = falses(S)
            for s in 1:S # generate one cut per scene
                print("\rs = $s")
                cdict = cut_gen(Q2_ast, Q2, t, s, l_x_dict, state_trial, th_s_trial[s])
                if cdict["cut_gened"]
                    @assert isapprox(cdict["pai0"] + LinearAlgebra.norm(cdict["pai"], norm_sense), 1.; atol = 0.01) "cut coeff's bounding restriction pai0 + ||pai|| <= 1. is not active!"
                    push!(cut_mat[t,s]["pai"], cdict["pai"])
                    push!(cut_mat[t,s]["pai0"], cdict["pai0"])
                    push!(cut_mat[t,s]["rhs"], cdict["rhs"])
                    push!(cut_mat[t,s]["id"], length(cut_mat[t,s]["id"]) + 1)
                    bv[s] = true
                end
            end
            print("\r")
            if !any(bv) # no more new cuts can be added
                @info "▶▶▶ t = $t // ite = $(ite), break due to cut saturation."
                break
            end
        end # end of loop on adding Lagrangian cuts
        xm1_final .= xm1_holder
        @info "(policy 4p3) ◀ decision phase t = $t" linking_decision=xm1_final inducing_cost=cost_4p3[t]
        if true # this block is for policy 4p1
            m = JumpModel()
            JuMP.@variable(m, y[t:T, 1:J], Bin)
            JuMP.@variable(m, 0. <= o[u = t:T] <= o_UB[u])
            JuMP.@variable(m,   x[t:T, 1:J] >= 0.)
            JuMP.@variable(m, x_B[t:T, 1:J] >= 0.)
            JuMP.@variable(m, x_H[t:T, 1:J] >= 0.)
            JuMP.@constraint(m, [u=t:T], sum(T_Y[j] * y[u,j] + T_B * x[u,j] for j in 1:J) - o[u] <= time_UB[u])
            JuMP.@constraint(m, [u=t:T, j=1:J], x[u,j] <= production_UB[j] * y[u,j])
            JuMP.@constraint(m, [u=t:T, j=1:J], x_H[u,j] + x[u,j] <= inventory_UB[j])
            JuMP.@constraint(m, [j=1:J], x_B[t,j] - x_H[t,j] + [1,-1,1.]' * xm1_4p1[j,:] == D_s[t,j]) # this RHS is by sampling
            t != T && JuMP.@constraint(m, [u=t+1:T, j=1:J], x_B[u,j] - x_H[u,j] + x_H[u-1,j] - x_B[u-1,j] + x[u-1,j] == D_s[u,j]) # these RHS's are by calculating cond_𝔼
            costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
            JuMP.@objective(m, Min, sum(c_O * o[u] + costs_oto(u) for u in t:T))
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
            cost_4p1[t] = c_O * JuMP.value(o[t]) + sum(c_B[t] * JuMP.value(x_B[t,j]) + c_H * JuMP.value(x_H[t,j]) + c_Y[j] * JuMP.value(y[t,j]) for j in 1:J)
            xm1_4p1 .= JuMP.value.([x_H[t,:] x_B[t,:] x[t,:]])
        end
        @info "(policy 4p1) ◀ decision phase t = $t" linking_decision=xm1_4p1 inducing_cost=cost_4p1[t]
    end # end of loop on the folding horizon process
    t = T
    if t == T # the final stage, which is a deterministic optimization problem
        for j in 1:J
            epsilon, delta = rand(d_epsilon), rand(d_delta[t,j]) # after this line, we have no further randomness
            tmp = rho * Ups_s[t-1, j] + (1-rho) * epsilon
            D_s[t, j], Ups_s[t, j] = rhoY * tmp * MU[t, j] + (1-rhoY) * delta, tmp
        end
        m = JumpModel()
        JuMP.@variable(m, y[1:J], Bin)
        JuMP.@variable(m, 0. <= o <= o_UB[T])
        JuMP.@variable(m,   x[1:J] >= 0.)
        JuMP.@variable(m, x_B[1:J] >= 0.)
        JuMP.@variable(m, x_H[1:J] >= 0.)
        JuMP.@constraint(m, sum(T_Y[j] * y[j] + T_B * x[j] for j in 1:J) - o <= time_UB[T])
        JuMP.@constraint(m, [j=1:J], x[j] <= production_UB[j] * y[j])
        JuMP.@constraint(m, [j=1:J], x_H[j] + x[j] <= inventory_UB[j])
        JuMP.@constraint(m, tbrpl[j=1:J], x_B[j] - x_H[j] + [1,-1,1.]' * xm1_final[j,:] == D_s[T,j]) # this RHS is by sampling
        JuMP.@objective(m, Min, c_O * o + sum(c_B[T] * x_B[j] + c_H * x_H[j] + c_Y[j] * y[j] for j in 1:J))
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        cost_4p3[t] = JuMP.objective_value(m)
        xm1_final .= JuMP.value.([x_H x_B x])
        @info "(policy 4p3) ◀ decision phase t = $t" linking_decision=xm1_final inducing_cost=cost_4p3[t]
        JuMP.delete(m, tbrpl)
        JuMP.unregister(m, :tbrpl)
        JuMP.@constraint(m, [j=1:J], x_B[j] - x_H[j] + [1,-1,1.]' * xm1_4p1[j,:] == D_s[T,j]) # this RHS is by sampling
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        cost_4p1[t] = JuMP.objective_value(m)
        xm1_4p1 .= JuMP.value.([x_H x_B x])
        @info "(policy 4p1) ◀ decision phase t = $t" linking_decision=xm1_4p1 inducing_cost=cost_4p1[t]
    end
    @info " ■ 4p1 and 4p3" cost_4p1 cost_4p3
    (sum(cost_4p1), sum(cost_4p3))
end

# 🏹 to compare policy 4.1 and 4.3, we sample a path and compare their costs, 
# step 1, run the following `if-end` 
# step 2, give a rand(Int) as arg, then eval the function
if true # containers for the whole folding horizon process
    D_s, Ups_s = deepcopy(MU), ones(T,J) # this represents the real realized path
    D3, Ups3 = zeros(T, J, S), ones(T, J, S) # these are created for the prediction purpose in policy 4p3
    Qahat = [Dict(
            "cp"       => Matrix{Float64}[],
            "cp0"      => Float64[],
            "id"       => Int[]
    ) for t in 1:T-1, s in 1:S]
    cut_mat = [Dict(
        "pai" => Matrix{Float64}[],
        "pai0" => Float64[],
        "rhs" => Float64[],
        "id" => Int[]
    ) for t in 1:T-1, s in 1:S] # ⚠️ there isn't a cut-generating phase at t = T
    cost_4p1 = NaN * ones(T)
    cost_4p3 = NaN * ones(T) # ⚠️ the end entry is filled at stage (T-1)
    xm1_holder = NaN * ones(J, 3) # a container for the linking state variables
    xm1_final = NaN * ones(J, 3)
    xm1_4p1 = zeros(J, 3)
end
compare_4p1_and_4p3(1)
(1.0826447853628987, 1.030998981967765)

compare_4p1_and_4p3(2)
(1.1613419656256803, 0.9411703920966215)

compare_4p1_and_4p3(3)
(0.973764355855891, 0.9658351428518281)

compare_4p1_and_4p3(4)
(0.9242871949043029, 0.9321374671432369)

compare_4p1_and_4p3(9999)
(0.915811140431844, 0.8788546913458437)

if false # view the samplers
    f = Figure();
    axs = Axis.([f[i...] for i in Iterators.product([1,2,3,4],[1,2,3])]);
    for t in 1:4, j in 1:3
        xt = range(0, 3. * MU[t, j]; length = 100);
        lines!(axs[t, j], xt, Distributions.pdf.(d_delta[t, j], xt))
        scatter!(axs[t, j], [MU[t, j]], [0.]; color = :cyan)
    end
end
if true # 📀 [Important] preprocessings
    function one_side_bound_raw(sampler::Function, N)
        v = [sampler() for _ in 1:N]
        x̄ = Statistics.mean(v)
        s = Statistics.std(v; mean = x̄)
        x̄, z_alpha * s / sqrt(N) # (67702.00986320838, 222.71804425951458) at N = 10000
    end
    # N = 10000
    # julia> one_side_bound_raw(a_sample_value_by_policy_4p1, N)
    # (1.128366831053581, 0.0037119674043307573)
    # julia> one_side_bound_raw(a_perfect_info_sample_value, N)
    # (0.9333638166090396, 0.0023945551307127952)
    function a_sample_value_by_policy_4p1()
        costs = zeros(T)
        x_m1 = zeros(J)
        x_B_m1 = zeros(J)
        x_H_m1 = zeros(J)
        D_s = deepcopy(MU)
        (Ups_s = deepcopy(D_s); Ups_s .= 1.)
        for t in 1:T
            if t >= 2
                for j in 1:J # sampling to generate ξₜ
                    epsilon, delta = rand(d_epsilon), rand(d_delta[t,j])
                    Ups_s[t, j] = rho * Ups_s[t-1, j] + (1. - rho) * epsilon
                    D_s[t, j] = rhoY * Ups_s[t, j] * MU[t, j] + (1. - rhoY) * delta
                end
                for u in t+1:T, j in 1:J # calculate 𝔼_{ξᵗ}{ r.v. > t }
                    D_s[u, j] = MU[u, j] * (1. + rhoY * rho^(u-t) * (Ups_s[t, j] - 1.))
                end
            end
            m = JumpModel()
            JuMP.@variable(m, y[t:T, 1:J], Bin)
            JuMP.@variable(m, 0. <= o[u = t:T] <= o_UB[u])
            JuMP.@variable(m,   x[t:T, 1:J] >= 0.)
            JuMP.@variable(m, x_B[t:T, 1:J] >= 0.)
            JuMP.@variable(m, x_H[t:T, 1:J] >= 0.)
            JuMP.@constraint(m, [u=t:T], sum(T_Y[j] * y[u,j] + T_B * x[u,j] for j in 1:J) - o[u] <= time_UB[u])
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
        JuMP.@constraint(m, [u=t:T], sum(T_Y[j] * y[u,j] + T_B * x[u,j] for j in 1:J) - o[u] <= time_UB[u])
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
    N = 10000
    one_side_bound_raw(a_sample_value_by_policy_4p1, N) |> println
    one_side_bound_raw(a_perfect_info_sample_value, N) |> println
end
if true # 📀 [Important] NA Dual Programming Related
    if true # data and containers for the level method
        D3 = zeros(T, J, S)
        U3 = deepcopy(D3)
        for s in 1:S
            D_s = deepcopy(MU)
            (Ups_s = deepcopy(D_s); Ups_s .= 1.)
            for t in 2:T, j in 1:J
                epsilon, delta = rand(d_epsilon), rand(d_delta[t,j])
                Ups_s[t, j] = rho * Ups_s[t-1, j] + (1-rho) * epsilon
                D_s[t, j] = rhoY * Ups_s[t, j] * MU[t, j] + (1-rhoY) * delta
            end
            D3[:, :, s] .= D_s
            U3[:, :, s] .= Ups_s
        end
        lb = -Inf * ones(2); # ObjValue
        ub = Inf * ones(2); # ObjBound
        Vhat = [Dict{String, Array{Float64}}[] for _ in 1:S] # list of extreme points of MIP subproblems for each block 
        cost_vector = zeros(S)
        # the opt solution when S = 5000
        l_x_2[1] = Dict((1, 3, 4) => 1.0137155977993396e-5, (2, 2, 4) => 6.028446919920274e-6, (2, 3, 4) => 1.011054996349409e-5, (1, 2, 3) => -2.1744587199982812e-7, (1, 1, 2) => 7.809629676413858e-7, (3, 2, 4) => 7.697355275195144e-6, (1, 3, 3) => 8.896077643660707e-7, (1, 1, 4) => 1.3127828313518984e-5, (2, 2, 3) => 4.608947397779395e-8, (3, 3, 4) => 1.0997277056759108e-5, (2, 3, 3) => 9.520538860409112e-7, (2, 1, 4) => 1.53764650057964e-5, (1, 2, 2) => 1.8059381618977696e-6, (1, 1, 3) => -6.313658736049452e-7, (1, 3, 2) => 1.796661348124759e-6, (1, 2, 4) => 5.806376765557988e-6, (3, 1, 4) => 1.599354661023327e-5, (2, 1, 3) => 2.013670619596678e-6)
        l_x_2 = [Dict((1, 1, 1) => 0.) for _ in 1:2]
    end
    for lv_ite in 1:typemax(Int)
        m = JumpModel() # upper level
        if true # to be copied
            JuMP.@variable(m, l_x[t = 1:T-1, j = 1:J, u = t+1:T]) # dual weights
            JuMP.@variable(m, th[1:S])
            for s in 1:S # cutting planes of θ
                D_s = D3[:, :, s] # load random data
                Ups_s = U3[:, :, s] # used to calculate closed-form conditional Expectations
                for coeff_dict in Vhat[s]
                    o = coeff_dict["o"]
                    x = coeff_dict["x"]
                    y = coeff_dict["y"]
                    x_B = coeff_dict["x_B"]
                    x_H = coeff_dict["x_H"]
                    costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
                    primal_obj = sum(
                            c_O * o[u] + costs_oto(u) 
                        for u in 1:T
                    )
                    penal_with_dual_upper_vars = sum(
                            x[t, j] * sum(
                                    l_x[t, j, u] * ( D_s[u, j] - MU[u, j] * (1. + rhoY * rho^(u - t) * (Ups_s[t, j] - 1.)) )
                                for u in t+1:T
                            )
                        for t in 1:T-1, j in 1:J
                    )
                    JuMP.@constraint(m, 1e-4 * th[s] <= 1e-4 * primal_obj + 1e-4 * penal_with_dual_upper_vars) # ⚠️ we must scale coefficients
                end
            end
            JuMP.@variable(m, obj3a <= obj3a_UB)
            JuMP.@constraint(m, obj3a == fill(1/S, S)' * th)
        end
        JuMP.@objective(m, Max, obj3a)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
        ub[end] = JuMP.objective_value(m) # ObjBound
        ub[end] < ub[begin] && (ub[begin] = ub[end])
        if lb[begin] != -Inf
            if true # concluding session
                ObjValue, ObjBound = lb[begin], ub[begin]
                rel_gap = abs((ObjBound - ObjValue) / ObjValue)
                @info "▶ level_ite = $(lv_ite)" ObjBound ObjValue rel_gap
                if rel_gap < 0.01 / 100
                    @info "😊 dual problem convergent"
                    break
                end
            end
            JuMP.@constraint(m, obj3a >= .7 * ub[begin] + .3 * lb[begin]) # (EC.4)
            JuMP.@variable(m, a_l_x[t = 1:T-1, j = 1:J, u = t+1:T]) # abs's
            JuMP.@constraint(m, [t = 1:T-1, j = 1:J, u = t+1:T], a_l_x[t, j, u] >= l_x[t,j,u] - l_x_2[2][(t,j,u)])
            JuMP.@constraint(m, [t = 1:T-1, j = 1:J, u = t+1:T], a_l_x[t, j, u] >= l_x_2[2][(t,j,u)] - l_x[t,j,u])
            JuMP.@objective(m, Min, sum(a_l_x))
            # ✏️ if we use || ⋅ ||_2
            # JuMP.@objective(m, Min, sum( ( l_x[t,j,u] - l_x_2[2][(t,j,u)] ) ^ 2  for t = 1:T-1, j = 1:J, u = t+1:T ))
            # JuMP.set_attribute(m, "BarHomogeneous", 1) # [Gurobi exclusive] to delay or eliminate the circumstance that (QP) termination_status == NUMERICAL_ERROR
            JuMP.optimize!(m)
            tmp = JuMP.termination_status(m)
            tmp == JuMP.LOCALLY_SOLVED && @warn ">> level_ite = $(lv_ite), regularization problem locally solved."
            tmp != JuMP.OPTIMAL && error("in regularization problem, terminating with $tmp.")
        end
        l_x_2[2] = JuMP.value.(l_x).data # solution acquisition (whatever lb takes)
        for s in 1:S # block ≡ sample path
            D_s = D3[:, :, s] # load random data
            Ups_s = U3[:, :, s] # used to calculate closed-form conditional Expectations
            m = JumpModel()
            JuMP.@variable(m, y[1:T, 1:J], Bin)
            JuMP.@variable(m, 0. <= o[u = 1:T] <= o_UB[u])
            JuMP.@variable(m,   x[1:T, 1:J] >= 0.)
            JuMP.@variable(m, x_B[1:T, 1:J] >= 0.)
            JuMP.@variable(m, x_H[1:T, 1:J] >= 0.)
            JuMP.@constraint(m, [u=1:T], sum(T_Y[j] * y[u,j] + T_B * x[u,j] for j in 1:J) - o[u] <= time_UB[u])
            JuMP.@constraint(m, [u=1:T, j=1:J], x[u,j] <= production_UB[j] * y[u,j])
            JuMP.@constraint(m, [u=1:T, j=1:J], x_H[u,j] + x[u,j] <= inventory_UB[j])
            JuMP.@constraint(m, [j=1:J], x_B[1,j] - x_H[1,j] + 0. - 0. + 0. == D_s[1,j])
            JuMP.@constraint(m, [u=2:T, j=1:J], x_B[u,j] - x_H[u,j] + x_H[u-1,j] - x_B[u-1,j] + x[u-1,j] == D_s[u,j])
            if true # dual values present only in obj 
                costs_oto = u -> sum(c_B[u] * x_B[u,j] + c_H * x_H[u,j] + c_Y[j] * y[u,j] for j in 1:J)
                primal_obj = sum(
                        c_O * o[u] + costs_oto(u) 
                    for u in 1:T
                )
                # This is an illustration
                # t = 2
                # term = x[t, j] * (
                #     l_x[2, j, 3] * ( D_s[3, j] - E_{ xi_3, xi_4 | xi_1, xi_2 }{ D_s[3, j] } ) +
                #     l_x[2, j, 4] * ( D_s[4, j] - E_{ xi_3, xi_4 | xi_1, xi_2 }{ D_s[4, j] } ) +
                # )
                penal_with_dual_upper_vars = sum(
                        x[t, j] * sum(
                                l_x_2[2][(t, j, u)] * ( D_s[u, j] - MU[u, j] * (1. + rhoY * rho^(u - t) * (Ups_s[t, j] - 1.)) )
                            for u in t+1:T
                        )
                    for t in 1:T-1, j in 1:J
                )
                JuMP.@objective(m, Min, primal_obj + penal_with_dual_upper_vars)
            end
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL "$(JuMP.termination_status(m))"
            push!(Vhat[s], Dict(
                "o"   => JuMP.value.(o),
                "x"   => JuMP.value.(x),
                "y"   => JuMP.value.(y),  
                "x_B" => JuMP.value.(x_B),
                "x_H" => JuMP.value.(x_H)
                ))
            cost_vector[s] = JuMP.objective_value(m)
        end
        lb[end] = fill(1/S, S)' * cost_vector # a feasible ObjValue
        if lb[end] > lb[begin]
            lb[begin] = lb[end] # update incumbent ObjValue
            l_x_2[1] = l_x_2[2] # update incumbent dual weights
        end
    end
end

import JuMP
import Gurobi
import Random
import Distributions
import Statistics
using Logging
# using CairoMakie

# 26/01/24
# solving the NA dual problem with ||⋅||_1 level method 
# sample size S = 5000
# detailed log @eof

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
    # N = 10000 raw: u = (1.0939493482904157, 0.0035990318291414638)
    # N = 10000 raw: l = (0.9048899519779566, 0.0023215053447264793)
    statistical_ub = 1.0975483801195571 # due to policy 4p1, at N = 10000
    statistical_lb = 0.9025684466332301 # due to EVPI, at N = 10000
    SCALE = 61888 # to alleviate numerical issues
    T, J, S = 4, 3, 5000 # S is the samples taken in SAA
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
end

if false # view the samplers
    f = Figure();
    axs = Axis.([f[i...] for i in Iterators.product([1,2,3,4],[1,2,3])]);
    for t in 1:4, j in 1:3
        xt = range(0, 3. * MU[t, j]; length = 100);
        lines!(axs[t, j], xt, Distributions.pdf.(d_delta[t, j], xt))
        scatter!(axs[t, j], [MU[t, j]], [0.]; color = :cyan)
    end
end

if false # 💡 preprocessings
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
        x_m1 = [0. for j in 1:J]
        x_B_m1 = [0. for j in 1:J]
        x_H_m1 = [0. for j in 1:J]
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
        m = JumpModel() 
        # lv_ite == 4 && JuMP.unset_silent(m)
        if true # this is a copy
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
                    JuMP.@constraint(m, 1e-4 * th[s] <= 1e-4 * primal_obj + 1e-4 * penal_with_dual_upper_vars)
                end
            end
            JuMP.@variable(m, obj3a <= obj3a_UB)
            JuMP.@constraint(m, obj3a == fill(1/S, S)' * th)
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

# # # # # # # # # # # # # # # # # # 📚 at N = 5000, the logs of level method
# ┌ Info: ▶ level_ite = 2
# │   ObjBound = 1.0975483801195571
# │   ObjValue = 0.9061053943674257
# └   rel_gap = 0.21128114559541114
# ┌ Info: ▶ level_ite = 3
# │   ObjBound = 1.0975483801195571
# │   ObjValue = 0.9061053943674257
# └   rel_gap = 0.21128114559541114
# ┌ Info: ▶ level_ite = 4
# │   ObjBound = 1.0975483801195571
# │   ObjValue = 0.9061053943674257
# └   rel_gap = 0.21128114559541114
# ┌ Info: ▶ level_ite = 5
# │   ObjBound = 1.0975483801195571
# │   ObjValue = 0.9061053943674257
# └   rel_gap = 0.21128114559541114
# ┌ Info: ▶ level_ite = 6
# │   ObjBound = 0.9801017927882563
# │   ObjValue = 0.9061053943674257
# └   rel_gap = 0.08166422899677063
# ┌ Info: ▶ level_ite = 7
# │   ObjBound = 0.9742421183515708
# │   ObjValue = 0.9303700444634049
# └   rel_gap = 0.047155509949236764
# ┌ Info: ▶ level_ite = 8
# │   ObjBound = 0.9723456713299082
# │   ObjValue = 0.9571333818045802
# └   rel_gap = 0.015893594157845362
# ┌ Info: ▶ level_ite = 9
# │   ObjBound = 0.9704933024377124
# │   ObjValue = 0.9658682462461753
# └   rel_gap = 0.004788495956371125
# ┌ Info: ▶ level_ite = 10
# │   ObjBound = 0.969448830684741
# │   ObjValue = 0.9684978578410901
# └   rel_gap = 0.0009819049530689827
# ┌ Info: ▶ level_ite = 11
# │   ObjBound = 0.968985022428189
# │   ObjValue = 0.9689278905472288
# └   rel_gap = 5.8964017361353295e-5
# [ Info: 😊 dual problem convergent
# 
# julia> l_x_2[1] # the candidate dual solution
# Dict{Tuple{Int64, Int64, Int64}, Float64} with 18 entries:
#   (1, 3, 4) => 1.00721e-5
#   (2, 2, 4) => 6.02901e-6
#   (2, 3, 4) => 1.00535e-5
#   (1, 2, 3) => -2.07887e-7
#   (1, 1, 2) => 8.84024e-7
#   (3, 2, 4) => 7.6823e-6
#   (1, 3, 3) => 8.86545e-7
#   (1, 1, 4) => 1.32586e-5
#   (2, 2, 3) => 6.0429e-8
#   (3, 3, 4) => 1.09555e-5
#   (2, 3, 3) => 9.33593e-7
#   (2, 1, 4) => 1.54552e-5
#   (1, 2, 2) => 1.8014e-6
#   (1, 1, 3) => -6.70951e-7
#   (1, 3, 2) => 1.77746e-6
#   (1, 2, 4) => 5.81301e-6
#   (3, 1, 4) => 1.60817e-5
#   (2, 1, 3) => 2.01066e-6
# ✏️ the magnitude is small, because corr constrs' coefficients are large.

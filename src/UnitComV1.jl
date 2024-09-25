import LinearAlgebra
import Distributions
import PowerModels # to parse Matpower *.m data files
import Random
import Gurobi
import JuMP
using Logging


# bus 1 is reference bus
# Generator 1 is on ref bus to assure global feasibility
# Generator 1's minimum output power is 0, it has no ramping constraints, it is always on
# there is no wind farm or load on the ref bus
# version 1.2: The code is correct, by scrutinizing
# 24/9/24

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function JumpModel(i)
    if i == 0 # the most frequently used
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    elseif i == 1 # generic convex conic program
        m = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 # if you need Gurobi callback
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function load_data()
    # network_data = PowerModels.parse_file("data/case6ww.m")
    # basic_net_data = PowerModels.make_basic_network(network_data)
    # F = PowerModels.calc_basic_ptdf_matrix(basic_net_data)
    S_BASE = 100 # MVA
    F = [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    Bdict = Dict(
        "f" =>  Int[1,1,1,2,2,2,2,3,3,4,5],
        "t" =>  Int[2,4,5,3,4,5,6,5,6,5,6],
        "BC" => [4,6,4,4,6,3,9,7,8,2,4]/10
    )
    Wdict = Dict(
        "id" => Int[1, 2],
        "n" => Int[2, 3],
        "CW" => [500., 400],
        "MAX" => [1.8, 1.7]
    )
    Ldict = Dict(
        "id" => Int[1,2,3],
        "n" => Int[4,5,6],
        "CL" => 1.8 * [3000., 2583, 2708],
        "MAX" => [1, 1.2, 1]
    )
    Gdict = Dict(
        "id" => Int[1, 2, 3], # the 1st one is slack generator
        "n" => Int[1, 2, 3], # the 1st generator resides at bus 1
        "ZS" => [1., 0, 0], # the 1st generator is always on
        "ZP" => [0.5, 0, 0], # Pzero, consistent with IS
        "C2" => [53.3, 88.9, 74.1],
        "C1" => [1166.9, 1033.3, 1083.3],
        "C0" => [0., 200, 240], # the 1st generator has no generation cost if output power = 0
        "CR" => [83., 74, 77],
        "CST" => [210., 200, 240],
        "CSH" => [210., 200, 240],
        "PI" => [0., 0.375, 0.45], # the 1st generator doesn't have a positive output lower bound
        "PS" => [2, 1.5, 1.8],
        "RU" => [.6, .6, .6],
        "SU" => [.6, .6, .6],
        "RD" => [.6, .6, .6],
        "SD" => [.6, .6, .6],
        "UT" => Int[3, 3, 3],
        "DT" => Int[3, 3, 3]
    )
    G = length(Gdict["n"])
    W = length(Wdict["n"])
    L = length(Ldict["n"])
    B, N = size(F)
    SRD = 1.5
    return B, G, W, L, F, SRD, Bdict, Gdict, Wdict, Ldict
end

B, G, W, L, F, SRD, Bdict, Gdict, Wdict, Ldict = load_data()
T = 8

Y = [Wdict["MAX"][c] for t in 1:T, c in 1:W]
Z = [Ldict["MAX"][c] for t in 1:T, c in 1:L]

υ = JumpModel(0)
JuMP.@variable(υ, u[1:T, 2:G], Bin)
JuMP.@variable(υ, v[1:T, 2:G], Bin)
JuMP.@variable(υ, x[0:T, 2:G], Bin)
# 1st-stage cost
JuMP.@expression(υ, CGst[t = 1:T, g = 2:G], Gdict["CST"][g] * u[t, g])
JuMP.@expression(υ, CGsh[t = 1:T, g = 2:G], Gdict["CSH"][g] * v[t, g])
# linking constr, but authentic 1st stage
JuMP.@constraint(υ, state_logic[t = 1:T, g = 2:G], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
# minimum up-down time
JuMP.@constraint(υ, up1[g = 2:G, t = 1:T-Gdict["UT"][g]+1], sum(x[i, g] for i in t:t+Gdict["UT"][g]-1) >= Gdict["UT"][g] * u[t, g])
JuMP.@constraint(υ, up2[g = 2:G, t = T-Gdict["UT"][g]+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0.)
JuMP.@constraint(υ, dw1[g = 2:G, t = 1:T-Gdict["DT"][g]+1], sum(1. - x[i, g] for i in t:t+Gdict["DT"][g]-1) >= Gdict["DT"][g] * v[t, g])
JuMP.@constraint(υ, dw2[g = 2:G, t = T-Gdict["DT"][g]+1:T], sum(1. - x[i, g] - v[t, g] for i in t:T) >= 0.)
# 2nd-stage continuous variables
JuMP.@variable(υ, ϖ[1:T, 1:W] >= 0.) # wind curtail
JuMP.@variable(υ, ζ[1:T, 1:L] >= 0.) # load shed
JuMP.@constraint(υ, [t = 1:T, w in 1:W], ϖ[t, w] <= Y[t, w])
JuMP.@constraint(υ, [t = 1:T, l in 1:L], ζ[t, l] <= Z[t, l])
JuMP.@variable(υ, ρ[1:T, 1:G] >= 0.) # spinning reserve
JuMP.@variable(υ, p[0:T, 2:G]) # power output of the Generator 2:G
JuMP.@variable(υ, ϱ[1:T] >= 0.) # power output of the slack generator, has liability for the power balance
JuMP.@variable(υ, ϕ[1:T, 2:G] >= 0.) # epi-variable of Cost_Generators, only for 2:G
# ★ Linking ★ 
JuMP.@constraint(υ, [t = 1:T, g = 2:G], -Gdict["RD"][g] * x[t, g] - Gdict["SD"][g] * v[t, g] <= p[t, g] - p[t-1, g])
JuMP.@constraint(υ, [t = 1:T, g = 2:G], p[t, g] - p[t-1, g] <= Gdict["RU"][g] * x[t-1, g] + Gdict["SU"][g] * u[t, g])
# generator output region
JuMP.@constraint(υ, [t = 1:T], sum(ρ[t, :]) >= SRD) # system-level
JuMP.@constraint(υ, [t = 1:T], ϱ[t] <= Gdict["PS"][1] - ρ[t, 1])
JuMP.@constraint(υ, [t = 1:T, g = 2:G], Gdict["PI"][g] * x[t, g] <= p[t, g])
JuMP.@constraint(υ, [t = 1:T, g = 2:G], p[t, g] <= Gdict["PS"][g] * x[t, g] - ρ[t, g])
# line flow heat restriction, this constraint can be tight
JuMP.@expression(υ, line_flow[t = 1:T, b = 1:B], sum(F[b, Wdict["n"][w]] * (Y[t, w] - ϖ[t, w]) for w in 1:W) + 
                                                sum(F[b, Gdict["n"][g]] * p[t, g] for g in 2:G)
                                                - sum(F[b, Ldict["n"][l]] * (Z[t, l] - ζ[t, l]) for l in 1:L))
JuMP.@constraint(υ, heat_bound[t = 1:T, b = 1:B], -Bdict["BC"][b] <= line_flow[t, b] <= Bdict["BC"][b])
# power balance system level
JuMP.@constraint(υ, power_balance[t = 1:T], sum(Y[t, :]) - sum(ϖ[t, :]) + sum(p[t, :]) + ϱ[t] == sum(Z[t, :]) - sum(ζ[t, :]))
# ★ Cost ★
θ = (g, p) -> Gdict["C2"][g] * p^2 + Gdict["C1"][g] * p + Gdict["C0"][g]
JuMP.@constraint(υ, [t = 1:T, g = 2:G], ϕ[t, g] >= θ(g, p[t, g]) - (1. - x[t, g]) * θ(g, Gdict["PS"][g]))
JuMP.@expression(υ, CGres[t = 1:T, g = 1:G], Gdict["CR"][g] * ρ[t, g]) # reserve cost
JuMP.@expression(υ, CW[t = 1:T, w = 1:W], Wdict["CW"][w] * ϖ[t, w])
JuMP.@expression(υ, CL[t = 1:T, l = 1:L], Ldict["CL"][l] * ζ[t, l])
JuMP.@objective(υ, Min, sum(
                    sum(CGsh[t, g] + CGst[t, g] for g in 2:G) # first stage
                    + sum(CGres[t, g] for g in 1:G) # reserve cost
                    + sum(ϕ[t, g] for g in 2:G) + θ(1, ϱ[t])
                    + sum(CW[t, :]) + sum(CL[t, :]) 
                                                for t in 1:T))
for g in 2:G
    JuMP.fix(x[0, g], Gdict["ZS"][g]; force = true)
    JuMP.fix(p[0, g], Gdict["ZP"][g]; force = true)
end
JuMP.set_attribute(υ, "NonConvex", 0)
JuMP.optimize!(υ)
status = JuMP.termination_status(υ)
@assert status == JuMP.OPTIMAL
JuMP.objective_value(υ)






JuMP.value.(x)
JuMP.value.(p)

JuMP.value.(ρ)
JuMP.value.(ϖ)
JuMP.value.(ζ)

JuMP.value.(ϕ)
JuMP.value.(CGres)
JuMP.value.(CGst)
JuMP.value.(CGsh)
JuMP.value.(CW)
JuMP.value.(CL)


julia> JuMP.objective_value(υ)
20993.6846938024





























# x_ = JuMP.value.(x)
# u_ = JuMP.value.(u)
# v_ = JuMP.value.(v)
# ρ_ = JuMP.value.(ρ)
# p_ = JuMP.value.(p)

# function check_candidate_feasibility(x_, u_, v_, ρ_, p_)
#     υ = JuMP.Model()
#     JuMP.@variable(υ, x[t = 1:T, g = 1:G], Bin, start = x_[t, g])
#     JuMP.@variable(υ, u[t = 1:T, g = 1:G], Bin, start = u_[t, g])
#     JuMP.@variable(υ, v[t = 1:T, g = 1:G], Bin, start = v_[t, g])
#     JuMP.@variable(υ, ρ[t = 1:T, g = 1:G] >= 0., start = ρ_[t, g]) # spinning reserve
#     JuMP.@variable(υ, p[t = 1:T, g = 1:G], start = p_[t, g]) # in p.u.
#     JuMP.@constraint(υ, sys_spinning_reserve[t = 1:T], sum(ρ[t, :]) >= SYS_SPINNING_RESERVE_DEMAND)
#     JuMP.@constraint(υ, power_output_LB[t = 1:T, g = 1:G], p[t, g] >= P_MIN[g] * x[t, g])
#     JuMP.@constraint(υ, power_output_UB[t = 1:T, g = 1:G], p[t, g] + ρ[t, g] <= P_MAX[g] * x[t, g])
#     JuMP.@constraint(υ, power_balance[t = 1:T], sum(p[t, :]) == Demand[t])
#     JuMP.@constraint(υ, t1_status[g = 1:G], x[1, g] - 1. == u[1, g] - v[1, g]) # t = 0, units are all running
#     JuMP.@constraint(υ, t2_end_status[t = 2:T, g = 1:G], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
#     pdict = Dict(i => JuMP.start_value(i) for i in JuMP.all_variables(υ))
#     PF_constraint_violations = JuMP.primal_feasibility_report(υ, pdict)
# end





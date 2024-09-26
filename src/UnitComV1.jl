import LinearAlgebra
import Distributions
# import PowerModels # to parse Matpower *.m data files
import Random
import Gurobi
import JuMP
using Logging

# bus 1 is reference bus
# Generator 1 is on ref bus to assure global feasibility
# Generator 1's minimum output power is 0, it has no ramping constraints, it is always on
# there is no wind farm or load on the ref bus
# version 1.2: The code is correct, by scrutinizing
# 24/9/26

PTH = 0.0001 # probability threshold
Random.seed!(23)
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

function ip(x, y) return LinearAlgebra.dot(x, y) end
function θ(g, p) return Gdict["C2"][g] * p^2 + Gdict["C1"][g] * p + Gdict["C0"][g] end
B, G, W, L, F, SRD, Bdict, Gdict, Wdict, Ldict = load_data()
MY = [0.654662 0.641636; 0.656534 0.819678; 0.511227 0.392445; 0.435808 0.123746; 0.100096 0.35382; 0.782138 0.542857; 0.798238 0.76052; 0.39923 0.309604]
MZ = [[0.7310,0.4814,0.6908,0.4326,0.1753,0.8567,0.8665,0.6107] [0.7010,0.5814,0.3908,0.1326,0.4153,0.7567,0.8565,0.5107] [0.2010,0.6814,0.1908,0.4326,0.8153,0.7567,0.6565,0.7107]]
T = 8
function lenCutDict(dict) return length(dict["cn"]) end

function pushCutsZ(lD, pu, pv, px, pY, pZ, cn, u, v, x, Y, Z) # Bender's
    push!(lD["pu"], pu)
    push!(lD["pv"], pv)
    push!(lD["px"], px)
    push!(lD["pY"], pY)
    push!(lD["pZ"], pZ)
    push!(lD["cn"], cn)
    push!(lD["u"], u) # at trial point
    push!(lD["v"], v)
    push!(lD["x"], x)
    push!(lD["Y"], Y)
    push!(lD["Z"], Z)
    push!(lD["bkcn"], cn - (ip(pu, u) + ip(pv, v) + ip(px, x) + ip(pZ, Z) + ip(pY, Y)))
end
cutDictZ = Dict( # from Bender's cut, used by the latter DRC
    "pu" => JuMP.Containers.DenseAxisArray[], # also as bk's slope
    "pv" => JuMP.Containers.DenseAxisArray[], # also as bk's slope
    "px" => JuMP.Containers.DenseAxisArray[], # also as bk's slope
    "pY" => Matrix[], # also as bk's slope
    "pZ" => Matrix[], # i.e. ak
    "cn" => Float64[], # raw const from Bender's cut
    "u" => JuMP.Containers.DenseAxisArray[],
    "v" => JuMP.Containers.DenseAxisArray[],
    "x" => JuMP.Containers.DenseAxisArray[],
    "Y" => Matrix[],
    "Z" => Matrix[],
    "bkcn" => Float64[] # const part of bk
)
function pushCutsY(lD, pu, pv, px, pY, cn, u, v, x, Y) # latter DR functional's sup's cut
    push!(lD["pu"], pu)
    push!(lD["pv"], pv)
    push!(lD["px"], px)
    push!(lD["pY"], pY)
    push!(lD["cn"], cn)
    push!(lD["u"], u) # at trial point
    push!(lD["v"], v)
    push!(lD["x"], x)
    push!(lD["Y"], Y)
end
cutDictY = Dict( # from sup's cut, used by the former DRC
    "pu" => JuMP.Containers.DenseAxisArray[], # also as bk's slope
    "pv" => JuMP.Containers.DenseAxisArray[], # also as bk's slope
    "px" => JuMP.Containers.DenseAxisArray[], # also as bk's slope
    "pY" => Matrix[], # i.e. ak
    "cn" => Float64[], # also is precisely "bkcn"
    "u" => JuMP.Containers.DenseAxisArray[],
    "v" => JuMP.Containers.DenseAxisArray[],
    "x" => JuMP.Containers.DenseAxisArray[],
    "Y" => Matrix[]
)
cutDict1 = Dict( # 💡 Only for Benders' feasibility cut
    "pu" => JuMP.Containers.DenseAxisArray[],
    "pv" => JuMP.Containers.DenseAxisArray[],
    "px" => JuMP.Containers.DenseAxisArray[],
    "cn" => Float64[],
    "u" => JuMP.Containers.DenseAxisArray[],
    "v" => JuMP.Containers.DenseAxisArray[],
    "x" => JuMP.Containers.DenseAxisArray[]
)

function f(uu, vv, xx, YY::Matrix, ZZ::Matrix) # YY, ZZ should be in [0, 1]
    υ = JumpModel(0)
    JuMP.@variable(υ, u[1:T, 2:G])
    JuMP.@variable(υ, v[1:T, 2:G])
    JuMP.@variable(υ, x[0:T, 2:G])
    JuMP.@variable(υ, Y[1:T, 1:W])
    JuMP.@variable(υ, Z[1:T, 1:L])
    function rY(t, w) return Wdict["MAX"][w] * Y[t, w] end
    function rZ(t, l) return Ldict["MAX"][l] * Z[t, l] end
    # copy constrs
    JuMP.@constraint(υ, pu[t = 1:T, g = 2:G], u[t, g] == uu[t, g])
    JuMP.@constraint(υ, pv[t = 1:T, g = 2:G], v[t, g] == vv[t, g])
    JuMP.@constraint(υ, px[t = 0:T, g = 2:G], x[t, g] == xx[t, g])
    JuMP.@constraint(υ, pY[t = 1:T, w = 1:W], Y[t, w] == YY[t, w])
    JuMP.@constraint(υ, pZ[t = 1:T, l = 1:L], Z[t, l] == ZZ[t, l])
    # curtail and shedding
    JuMP.@variable(υ, ϖ[1:T, 1:W] >= 0.)
    JuMP.@variable(υ, ζ[1:T, 1:L] >= 0.)
    JuMP.@expression(υ, CW[t = 1:T, w = 1:W], Wdict["CW"][w] * ϖ[t, w])
    JuMP.@expression(υ, CL[t = 1:T, l = 1:L], Ldict["CL"][l] * ζ[t, l])
    JuMP.@constraint(υ, [t = 1:T, w in 1:W], ϖ[t, w] <= rY(t, w))
    JuMP.@constraint(υ, [t = 1:T, l in 1:L], ζ[t, l] <= rZ(t, l))
    # generations
    JuMP.@variable(υ, ρ[1:T, 1:G] >= 0.)
    JuMP.@expression(υ, CGres[t = 1:T, g = 1:G], Gdict["CR"][g] * ρ[t, g])
    JuMP.@variable(υ, ϱ[1:T] >= 0.) # power output of the slack generator, has liability for the power balance
    JuMP.@expression(υ, CGgen1[t = 1:T], θ(1, ϱ[t]))
    JuMP.@variable(υ, p[0:T, 2:G]) # power output of the Generator 2:G
    JuMP.@variable(υ, ϕ[1:T, 2:G] >= 0.) # epi-variable of Cost_Generators, only for 2:G
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], ϕ[t, g] >= θ(g, p[t, g]) - (1. - x[t, g]) * θ(g, Gdict["PS"][g]))
    # ★ Linking ★ 
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], -Gdict["RD"][g] * x[t, g] - Gdict["SD"][g] * v[t, g] <= p[t, g] - p[t-1, g])
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], p[t, g] - p[t-1, g] <= Gdict["RU"][g] * x[t-1, g] + Gdict["SU"][g] * u[t, g])
    for g in 2:G
        JuMP.fix(p[0, g], Gdict["ZP"][g]; force = true)
    end
    JuMP.@constraint(υ, [t = 1:T], sum(ρ[t, :]) >= SRD)
    JuMP.@constraint(υ, [t = 1:T], ϱ[t] <= Gdict["PS"][1] - ρ[t, 1])
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], Gdict["PI"][g] * x[t, g] <= p[t, g])
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], p[t, g] <= Gdict["PS"][g] * x[t, g] - ρ[t, g])
    JuMP.@expression(υ, line_flow[t = 1:T, b = 1:B], sum(F[b, Wdict["n"][w]] * (rY(t, w) - ϖ[t, w]) for w in 1:W) + 
                                                sum(F[b, Gdict["n"][g]] * p[t, g] for g in 2:G)
                                                - sum(F[b, Ldict["n"][l]] * (rZ(t, l) - ζ[t, l]) for l in 1:L))
    JuMP.@constraint(υ, TBD1[t = 1:T, b = 1:B], -Bdict["BC"][b] <= line_flow[t, b] <= Bdict["BC"][b])
    JuMP.@constraint(υ, TBD2[t = 1:T], sum(rY(t, w) - ϖ[t, w] for w in 1:W) + sum(p[t, :]) + ϱ[t] == sum(rZ(t, l) - ζ[t, l] for l in 1:L))
    JuMP.@objective(υ, Min, sum(CW) + sum(CL) + sum(CGres) + sum(CGgen1) + sum(ϕ))
    JuMP.set_attribute(υ, "NonConvex", 0)
    JuMP.set_attribute(υ, "QCPDual", 1) # Gurobi's QCPDual is good
    JuMP.optimize!(υ)
    status = JuMP.termination_status(υ)
    if status == JuMP.INFEASIBLE_OR_UNBOUNDED # means that (u, v, x) is improper
        # JuMP.delete.(υ, TBD1)
        # JuMP.unregister(υ, :TBD1)
        # JuMP.delete.(υ, TBD2)
        # JuMP.unregister(υ, :TBD2)
        # JuMP.@variable(υ, s1[t = 1:T, b = 1:B] >= 0.)
        # JuMP.@variable(υ, s2[t = 1:T, b = 1:B] >= 0.)
        # JuMP.@variable(υ, s3[t = 1:T] >= 0.)
        # JuMP.@variable(υ, s4[t = 1:T] >= 0.)
        # JuMP.@constraint(υ, [t = 1:T, b = 1:B], -Bdict["BC"][b] <= line_flow[t, b] + s1[t, b])
        # JuMP.@constraint(υ, [t = 1:T, b = 1:B], line_flow[t, b] <= Bdict["BC"][b] + s2[t, b])
        # JuMP.@constraint(υ, [t = 1:T], sum(rY(t, w) - ϖ[t, w] for w in 1:W) + sum(p[t, :]) + ϱ[t] + s3[t] - s4[t] == sum(rZ(t, l) - ζ[t, l] for l in 1:L))
        # JuMP.@objective(υ, Min, sum(s1) + sum(s2) + sum(s3) + sum(s4))
        # JuMP.optimize!(υ)
        # status = JuMP.termination_status(υ)        
        JuMP.delete.(υ, TBD2)
        JuMP.unregister(υ, :TBD2)
        JuMP.@variable(υ, s3[t = 1:T] >= 0.)
        JuMP.@variable(υ, s4[t = 1:T] >= 0.)
        JuMP.@constraint(υ, [t = 1:T], sum(rY(t, w) - ϖ[t, w] for w in 1:W) + sum(p[t, :]) + ϱ[t] + s3[t] - s4[t] == sum(rZ(t, l) - ζ[t, l] for l in 1:L))
        JuMP.@objective(υ, Min, sum(s3) + sum(s4))
        JuMP.optimize!(υ)
        status = JuMP.termination_status(υ)
        @assert status == JuMP.OPTIMAL "in f(), the Relaxed problem is still non-optimal!"
        push!(cutDict1["pu"], JuMP.dual.(pu))
        push!(cutDict1["pv"], JuMP.dual.(pv))
        push!(cutDict1["px"], JuMP.dual.(px))
        push!(cutDict1["cn"], JuMP.objective_value(υ))
        push!(cutDict1["u"], uu)
        push!(cutDict1["v"], vv)
        push!(cutDict1["x"], xx)
        return nothing
    end
    @assert status == JuMP.OPTIMAL "in f(), status = $status"
    (JuMP.dual.(pu),
     JuMP.dual.(pv),
     JuMP.dual.(px),
     JuMP.dual.(pY)::Matrix,
     JuMP.dual.(pZ)::Matrix,
     JuMP.objective_value(υ),
     uu,
     vv,
     xx,
     YY,
     ZZ)
end

function RCZ(u, v, x, Y, lD, MZ) # OK
    K = lenCutDict(lD) # should be cutDictZ
    function a(k)::Matrix return lD["pZ"][k] end
    function b(k, u, v, x, Y)::Float64
        pu, pv, px, pY = lD["pu"][k], lD["pv"][k], lD["px"][k], lD["pY"][k]
        return ip(pu, u) + ip(pv, v) + ip(px, x) + ip(pY, Y) + lD["bkcn"][k]
    end
    υ = JumpModel(0) # a Robust Counterpart
    JuMP.@variable(υ, α)
    JuMP.@variable(υ, β[1:T, 1:L])
    JuMP.@objective(υ, Min, α + ip(MZ, β))
    JuMP.@variable(υ, ς[1:K, 1:T, 1:L] >= 0.) 
    JuMP.@constraint(υ, η[k = 1:K], α >= b(k, u, v, x, Y) + sum(ς[k, :, :]))
    JuMP.@constraint(υ, ξ[k = 1:K, t = 1:T, l = 1:L], β[t, l] + ς[k, t, l] >= a(k)[t, l])
    JuMP.optimize!(υ)
    status = JuMP.termination_status(υ)
    @assert status == JuMP.OPTIMAL "in RCZ, status = $status"
    η = JuMP.dual.(η)::Vector
    ξ = JuMP.dual.(ξ)::Array
    # cut generation  θ(u, v, x, Y) >= cn + <pu, u> + <pv, v> + <px, x> + <pY, Y>
    pu = sum(η[k] .* lD["pu"][k] for k in 1:K) # 💡 this sum is w.r.t matrix
    pv = sum(η[k] .* lD["pv"][k] for k in 1:K) # 💡 this sum is w.r.t matrix
    px = sum(η[k] .* lD["px"][k] for k in 1:K) # 💡 this sum is w.r.t matrix
    pY = sum(η[k] .* lD["pY"][k] for k in 1:K) # 💡 this sum is w.r.t matrix
    cn = sum(η[k]  * lD["bkcn"][k] + ip(ξ[k, :, :], a(k)) for k in 1:K)
    # eliminate 0's AFTER cut generation ⚠️
    bitvec = η .> PTH
    η, ξ = η[bitvec], ξ[bitvec, :, :]
    Ztrial = [ξ[i, r, c] / η[i] for i in eachindex(η), r in 1:size(ξ)[2], c in 1:size(ξ)[3]]
    (pu, pv, px, pY, cn, u, v, x, Y), (η, Ztrial)
end

function RCY(u, v, x, lD, MY) # OK
    K = lenCutDict(lD) # should be cutDictY, see the cut generation part in RCZ()
    function a(k)::Matrix return lD["pY"][k] end
    function b(k, u, v, x)::Float64
        pu, pv, px = lD["pu"][k], lD["pv"][k], lD["px"][k]
        return ip(pu, u) + ip(pv, v) + ip(px, x) + lD["cn"][k]
    end
    υ = JumpModel(0)
    JuMP.@variable(υ, α)
    JuMP.@variable(υ, β[1:T, 1:W])
    JuMP.@objective(υ, Min, α + ip(MY, β))
    JuMP.@variable(υ, ς[1:K, 1:T, 1:W] >= 0.)
    JuMP.@constraint(υ, η[k = 1:K], α >= b(k, u, v, x) + sum(ς[k, :, :]))
    JuMP.@constraint(υ, ξ[k = 1:K, t = 1:T, w = 1:W], β[t, w] + ς[k, t, w] >= a(k)[t, w])
    JuMP.optimize!(υ)
    status = JuMP.termination_status(υ)
    @assert status == JuMP.OPTIMAL "in RCY, status = $status"
    η = JuMP.dual.(η)::Vector
    ξ = JuMP.dual.(ξ)::Array
    # eliminate 0's
    bitvec = η .> PTH
    η, ξ = η[bitvec], ξ[bitvec, :, :]
    Ytrial = [ξ[i, r, c] / η[i] for i in eachindex(η), r in 1:size(ξ)[2], c in 1:size(ξ)[3]]
    (η, Ytrial)
end

function master_plus_RCY(lD, MY)
    K = lenCutDict(lD) # should be cutDictY, see the cut generation part in RCZ()
    function a(k)::Matrix return lD["pY"][k] end
    function b(k, u, v, x)
        pu, pv, px = lD["pu"][k], lD["pv"][k], lD["px"][k]
        return ip(pu, u) + ip(pv, v) + ip(px, x) + lD["cn"][k]
    end
    υ = JumpModel(0)
    JuMP.@variable(υ, u[1:T, 2:G], Bin)
    JuMP.@variable(υ, v[1:T, 2:G], Bin)
    JuMP.@variable(υ, x[0:T, 2:G], Bin)
    # 1st-stage cost
    JuMP.@expression(υ, CGst[t = 1:T, g = 2:G], Gdict["CST"][g] * u[t, g])
    JuMP.@expression(υ, CGsh[t = 1:T, g = 2:G], Gdict["CSH"][g] * v[t, g])
    JuMP.@expression(υ, COST1, sum(CGst) + sum(CGsh))
    # linking constr, but authentic 1st stage
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
    for g in 2:G
        JuMP.fix(x[0, g], Gdict["ZS"][g]; force = true)
    end
    # minimum up-down time
    JuMP.@constraint(υ, [g = 2:G, t = 1:T-Gdict["UT"][g]+1], sum(x[i, g] for i in t:t+Gdict["UT"][g]-1) >= Gdict["UT"][g] * u[t, g])
    JuMP.@constraint(υ, [g = 2:G, t = T-Gdict["UT"][g]+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0.)
    JuMP.@constraint(υ, [g = 2:G, t = 1:T-Gdict["DT"][g]+1], sum(1. - x[i, g] for i in t:t+Gdict["DT"][g]-1) >= Gdict["DT"][g] * v[t, g])
    JuMP.@constraint(υ, [g = 2:G, t = T-Gdict["DT"][g]+1:T], sum(1. - x[i, g] - v[t, g] for i in t:T) >= 0.)
    D1 = cutDict1
    for (pu, pv, px, cn, ut, vt, xt) in zip(D1["pu"], D1["pv"], D1["px"], D1["cn"], D1["u"], D1["v"], D1["x"])
        JuMP.@constraint(υ, 0. >= cn + ip(pu, u .- ut) + ip(pv, v .- vt) + ip(px, x .- xt)) # Bender's feas cut
    end
    # RCY's part
    JuMP.@variable(υ, α)
    JuMP.@variable(υ, β[1:T, 1:W])
    JuMP.@objective(υ, Min, COST1 + α + ip(MY, β)) # 📕 joint optimization
    JuMP.@variable(υ, ς[1:K, 1:T, 1:W] >= 0.)
    JuMP.@constraint(υ, [k = 1:K], α >= b(k, u, v, x) + sum(ς[k, :, :]))
    JuMP.@constraint(υ, [k = 1:K, t = 1:T, w = 1:W], β[t, w] + ς[k, t, w] >= a(k)[t, w])
    # JuMP.set_attribute(υ, "Presolve", 0)
    JuMP.optimize!(υ)
    status = JuMP.termination_status(υ)
    @assert status == JuMP.OPTIMAL "in master_plus_RCY, status = $status"
    return JuMP.value.(u), JuMP.value.(v), JuMP.value.(x), JuMP.value.(COST1), JuMP.objective_value(υ)
end

function initial_master()
    υ = JumpModel(0)
    JuMP.@variable(υ, u[1:T, 2:G], Bin)
    JuMP.@variable(υ, v[1:T, 2:G], Bin)
    JuMP.@variable(υ, x[0:T, 2:G], Bin)
    # 1st-stage cost
    JuMP.@expression(υ, CGst[t = 1:T, g = 2:G], Gdict["CST"][g] * u[t, g])
    JuMP.@expression(υ, CGsh[t = 1:T, g = 2:G], Gdict["CSH"][g] * v[t, g])
    JuMP.@expression(υ, COST1, sum(CGst) + sum(CGsh))
    # linking constr, but authentic 1st stage
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
    for g in 2:G
        JuMP.fix(x[0, g], Gdict["ZS"][g]; force = true)
    end
    # minimum up-down time
    JuMP.@constraint(υ, [g = 2:G, t = 1:T-Gdict["UT"][g]+1], sum(x[i, g] for i in t:t+Gdict["UT"][g]-1) >= Gdict["UT"][g] * u[t, g])
    JuMP.@constraint(υ, [g = 2:G, t = T-Gdict["UT"][g]+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0.)
    JuMP.@constraint(υ, [g = 2:G, t = 1:T-Gdict["DT"][g]+1], sum(1. - x[i, g] for i in t:t+Gdict["DT"][g]-1) >= Gdict["DT"][g] * v[t, g])
    JuMP.@constraint(υ, [g = 2:G, t = T-Gdict["DT"][g]+1:T], sum(1. - x[i, g] - v[t, g] for i in t:T) >= 0.)
    JuMP.@objective(υ, Min, COST1)
    JuMP.optimize!(υ)
    status = JuMP.termination_status(υ)
    @assert status == JuMP.OPTIMAL " In master(): $status"
    JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
end

# TODO modify the trial point process, night 9/26

# ------------ Initialization 📚 ------------
u, v, x = initial_master()
# execute these TWICE
Y, Z = rand(T, W), rand(T, L)
res = f(u, v, x, Y, Z)
@assert typeof(res) != Nothing " Initialization fails! Change a random seed. "
pushCutsZ(cutDictZ, res...)
Y, Z = rand(T, W), rand(T, L)
res = f(u, v, x, Y, Z)
@assert typeof(res) != Nothing " Initialization fails! Change a random seed. "
pushCutsZ(cutDictZ, res...)

# execute these TWICE
Y = rand(T, W)
res, _ = RCZ(u, v, x, Y, cutDictZ, MZ)
pushCutsY(cutDictY, res...)
Y = rand(T, W)
res, _ = RCZ(u, v, x, Y, cutDictZ, MZ)
pushCutsY(cutDictY, res...)

# ------------ begin the ALGORITHM formally 📚 ------------
for ite in 1:20000
    continue_flag = [false]
    u, v, x, COST1, lb = master_plus_RCY(cutDictY, MY)
    etaY, Yt = RCY(u, v, x, cutDictY, MY)
    COST2 = [0.]
    for (i, prob_i) in enumerate(etaY)
        Y = Yt[i, :, :]
        _, (etaZ, Zt) = RCZ(u, v, x, Y, cutDictZ, MZ)
        for (j, prob_j) in enumerate(etaZ)
            Z = Zt[j, :, :]
            res = f(u, v, x, Y, Z) # evaluate the 2nd stage cost
            if typeof(res) === Nothing
                continue_flag[1] = true
                break
            end
            COST2[1] += prob_i * prob_j * res[6]
            pushCutsZ(cutDictZ, res...)
        end
        continue_flag[1] && break
        res, _ = RCZ(u, v, x, Y, cutDictZ, MZ)
        pushCutsY(cutDictY, res...)
    end
    continue_flag[1] && continue
    @info ">>> lb = $lb | $(COST1 + COST2[1]) = ub"
end

# results:
[ Info: >>> lb = -5468.246511361018 | 16184.321989708544 = ub
[ Info: >>> lb = 7364.285899366696 | 23812.150770031898 = ub
[ Info: >>> lb = 25114.20724134319 | 27099.389714819896 = ub
[ Info: >>> lb = 27880.331470794557 | 29039.88858157922 = ub
[ Info: >>> lb = 29238.993159805977 | 29484.599249407347 = ub
[ Info: >>> lb = 29594.503515765795 | 30212.69384249913 = ub
[ Info: >>> lb = 31032.89850718894 | 32041.34835631958 = ub
[ Info: >>> lb = 32136.99075569302 | 32241.48193777314 = ub
[ Info: >>> lb = 32279.513110971344 | 32395.54031545915 = ub
[ Info: >>> lb = 32428.665456285198 | 32489.349025488034 = ub
[ Info: >>> lb = 32526.795086753842 | 32565.999252176916 = ub
[ Info: >>> lb = 32576.257915188362 | 32731.68833815564 = ub
[ Info: >>> lb = 32746.41078443818 | 32768.31306022063 = ub
[ Info: >>> lb = 32773.131734556424 | 32779.75555764785 = ub
[ Info: >>> lb = 32779.99802922258 | 32780.33439852922 = ub
[ Info: >>> lb = 32780.36444506726 | 32780.618103085224 = ub
[ Info: >>> lb = 32780.61812268202 | 32780.77290094291 = ub
[ Info: >>> lb = 32780.77292099248 | 32780.77290033713 = ub
[ Info: >>> lb = 32780.772921157324 | 32780.7815871818 = ub


# change single cut strategy to multi-cut version






# ------------ end the ALGORITHM formally 📚 ------------
# ------------ end the ALGORITHM formally 📚 ------------
# ------------ end the ALGORITHM formally 📚 ------------
# ------------ end the ALGORITHM formally 📚 ------------
# ------------ end the ALGORITHM formally 📚 ------------
# ------------ end the ALGORITHM formally 📚 ------------
# ------------ end the ALGORITHM formally 📚 ------------








# begin algo
for i in 1:50000
    ut, vt, xt, COST1, lb = master()
    Y = rand(T, W)
    Z = rand(T, L)
    res = f(ut, vt, xt, Y, Z)
    # @info "cmp" lb = lb  ub = COST1 + cn
    pushCutsZ(cutDictZ, res...)
end

julia> JuMP.objective_value(υ)
32947.96025336013





function f(t, xt, p0t, p1t, z) # assume now Y is constant // index (t) // trial is (xt, p0t, p1t) // uncertainty is (z)
    # t: at stage t ∈ 1:T
    # xt: x[t, :]
    # p0t: p0[t, :]
    # p1t: p1[t, :, :, :]
    # z should be full z[1:T, 1:L]
    function pt(g) return p0t[g] + sum(p1t[g, τ, l] * z(τ, l) for τ in 1:T, l in 1:L) end
    υ = JumpModel(0) # at stage t ∈ 1:T
    JuMP.@variable(υ, ϖ[1:W] >= 0.) # wind curtail
    JuMP.@variable(υ, ζ[1:L] >= 0.) # load shed
    JuMP.@constraint(υ, [w = 1:W], ϖ[w] <= Y[t, w])
    JuMP.@constraint(υ, [l = 1:L], ζ[l] <= z[t, l])
    JuMP.@variable(υ, ρ[1:G] >= 0.) # reserve
    JuMP.@variable(υ, ϱ >= 0.) # slack power by Gen #1
    JuMP.@constraint(υ, sum(ρ) >= SRD) # system-level sufficient reserve at stage t
    JuMP.@constraint(υ, ρ[1] <= Gdict["PS"][1] - ϱ)
    JuMP.@constraint(υ, [g = 2:G], ρ[g] <= Gdict["PS"][g] * xt[g] - pt(g))
    JuMP.@expression(υ, line_flow[b = 1:B],   sum(F[b, Wdict["n"][w]] * (Y[t, w] - ϖ[w]) for w in 1:W)
                                            + sum(F[b, Gdict["n"][g]] * pt(g)            for g in 2:G)
                                            - sum(F[b, Ldict["n"][l]] * (z[t, l] - ζ[l]) for l in 1:L))
    JuMP.@constraint(υ, [b = 1:B], -Bdict["BC"][b] <= line_flow[b] <= Bdict["BC"][b])
    JuMP.@constraint(υ, sum(Y[t, :]) - sum(ϖ) + sum(pt(g) for g in 2:G) + ϱ == sum(z[t, :]) - sum(ζ))
    # only cost at stage t
    JuMP.@expression(υ, CR[g = 1:G], Gdict["CR"][g] * ρ[g])
    JuMP.@expression(υ, CW[w = 1:W], Wdict["CW"][w] * ϖ[w])
    JuMP.@expression(υ, CL[l = 1:L], Ldict["CL"][l] * ζ[l])
    JuMP.@objective(υ, Min, θ(1, ϱ) + sum(CR) + sum(CW) + sum(CL))
    JuMP.set_attribute(υ, "NonConvex", 0)
    JuMP.optimize!(υ)
    status = JuMP.termination_status(υ)
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(υ)
end



import Statistics
import Distributions

function gen_covmat(N::Int, d::Int) # cov at d = 3 >> cov at d = 300
    Σ = Statistics.cov(randn(N + d, N))
    @assert minimum(LinearAlgebra.eigen(Σ).values) > 0
    Σ
end

A = gen_covmat(8, 10)

function gen_valid_e(d)
    while true
        e = rand(d)
        if minimum(e) >= 0 && maximum(e) <= 1
            return e
        end
    end
end
function gen_sample()
    # data furnishing:
    # mean vector along the time axis should be specified
    # covariance matrix is randomly generated by using the 'gen_covmat' function above
    m1 = [0.7310,0.4814,0.6908,0.4326,0.1753,0.8567,0.8665,0.6107]
    covmat1 = 0.02 * [1.0275641071270425 -0.21521375062984943 -0.10245618227502372 0.2736442096080011 0.1935143957813836 0.1163562754916817 0.11006764381192213 0.32217900149439127; -0.21521375062984943 0.8013520640849399 -0.07653762897281689 0.42311258113769884 -0.08247008996625137 0.12827264196216165 -0.013906395918773724 -0.372504987824754; -0.10245618227502372 -0.07653762897281689 1.0828875925632344 -0.36011147351418554 0.2325457245541218 0.12795866926476612 0.5516926277019879 -0.16479288057424205; 0.2736442096080011 0.42311258113769884 -0.36011147351418554 0.6632796791884628 0.05179161732790655 0.3704424029591757 -0.3781539158150975 -0.2879625749085806; 0.1935143957813836 -0.08247008996625137 0.2325457245541218 0.05179161732790655 0.452259165276217 0.27875881891668036 -0.09133184339413176 0.26892579999270316; 0.1163562754916817 0.12827264196216165 0.12795866926476612 0.3704424029591757 0.27875881891668036 1.5547427522554451 -0.36800776415607667 -0.3693489187340054; 0.11006764381192213 -0.013906395918773724 0.5516926277019879 -0.3781539158150975 -0.09133184339413176 -0.36800776415607667 1.586434788868833 0.02332408987960274; 0.32217900149439127 -0.372504987824754 -0.16479288057424205 -0.2879625749085806 0.26892579999270316 -0.3693489187340054 0.02332408987960274 0.914525523585337]
    m2 = [0.7010,0.5814,0.3908,0.1326,0.4153,0.7567,0.8565,0.5107]
    covmat2 = 0.02 * [1.1297131964778115 -0.19430347894392902 -0.22611009760629502 -0.6662853774669376 -0.05801643559820969 -0.491736002353186 0.3556786721313869 0.23415972195877455; -0.19430347894392902 0.6901166924049221 0.3250169978105876 0.10344026865996811 0.1836746787626621 0.40036590319353765 -0.25080655949861147 0.00453970813018895; -0.22611009760629502 0.3250169978105876 0.9736426085087982 0.3257892180401471 0.5582182127537276 0.33485374916006305 -0.1528249394026537 -0.049752640012164534; -0.6662853774669376 0.10344026865996811 0.3257892180401471 1.0588376134720985 0.13503901871060137 0.46576995546312555 0.11736824878146586 -0.05617277220690781; -0.05801643559820969 0.1836746787626621 0.5582182127537276 0.13503901871060137 0.5628217162949489 0.114854858344524 -0.12470207464300379 0.15003561626474068; -0.491736002353186 0.40036590319353765 0.33485374916006305 0.46576995546312555 0.114854858344524 1.7075069672680152 -0.3602783058022874 0.3221419245092686; 0.3556786721313869 -0.25080655949861147 -0.1528249394026537 0.11736824878146586 -0.12470207464300379 -0.3602783058022874 0.6918539673534526 -0.42345126951551665; 0.23415972195877455 0.00453970813018895 -0.049752640012164534 -0.05617277220690781 0.15003561626474068 0.3221419245092686 -0.42345126951551665 1.7045672714652822]
    m3 = [0.2010,0.6814,0.1908,0.4326,0.8153,0.7567,0.6565,0.7107]
    covmat3 = 0.02 * [1.0647369807200258 0.1097563221175112 0.01329163886127755 0.09080652747965 -0.05734563884681584 0.13085820508616589 0.03850582808426042 -0.054748330589408166; 0.1097563221175112 1.0216028480028418 -0.07355410300358128 0.2007288310009736 0.20199648229746692 -0.10705602449542131 -0.041251037569431534 0.029273814777955823; 0.01329163886127755 -0.07355410300358128 1.0796987347186362 0.018510348029372767 -0.07865815328835442 0.1416057210973469 0.047232589822427506 0.15187428738059738; 0.09080652747965 0.2007288310009736 0.018510348029372767 0.8589038053631057 0.08338192532606366 -0.024673140599074137 0.14313735275310138 -0.2213197130130684; -0.05734563884681584 0.20199648229746692 -0.07865815328835442 0.08338192532606366 0.9365055285384479 -0.0607755054086011 -0.16188792416229125 0.1879239241651221; 0.13085820508616589 -0.10705602449542131 0.1416057210973469 -0.024673140599074137 -0.0607755054086011 1.025665883587342 0.15522639694673324 -0.07203668261448558; 0.03850582808426042 -0.041251037569431534 0.047232589822427506 0.14313735275310138 -0.16188792416229125 0.15522639694673324 0.8362885643022493 -0.09981345626783832; -0.054748330589408166 0.029273814777955823 0.15187428738059738 -0.2213197130130684 0.1879239241651221 -0.07203668261448558 -0.09981345626783832 1.0659711463172497]
    # use MvNormal distribution as the generator of load curve, for each location
    d1 = Distributions.MvNormal(m1, covmat1)
    d2 = Distributions.MvNormal(m2, covmat2)
    d3 = Distributions.MvNormal(m3, covmat3)
    l1 = gen_valid_e(d1)
    l2 = gen_valid_e(d2)
    l3 = gen_valid_e(d3)
    [l1 l2 l3]
end






















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




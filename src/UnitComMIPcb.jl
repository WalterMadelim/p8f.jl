import LinearAlgebra
import Distributions
import Statistics
import Random
import Gurobi
import JuMP
using Logging
GRB_ENV = Gurobi.Env()

# if you use (u v x and beta_1) (MILP) aggregate, after 6 hours, lb rises from -340 to -228, 15217 cuts of beth1
# one-shot MILP is typically (a bit) faster than pureIP equipped with Benders decomposition, but the advantage of the latter is that in Gurobi's MILP it would have the primal value column
# 18/12/24

UPDTH = 1e-5 # only a update greater than this threshold will be performed
B1BND, B2BND = 6.0, 3.6

# lines are unlimited and ramp rate are not restricted in `case118.m`
# NOTE: when (G+1 = 37), the main bottleneck is `master MIP`, rather than argmaxZ, the latter has only one line logging before converging to optimal.
ip(x, y)       = LinearAlgebra.dot(x, y)
rd6(f)         = round(f; digits = 6)
get_safe_bin(x) = Bool.(round.(JuMP.value.(x)))
jo(ø) = JuMP.objective_value(ø)
jv(x) = JuMP.value.(x)
jd(x) = JuMP.dual.(x)
brcs(v) = ones(T) * transpose(v) # to broadcast those timeless cost coeffs
macro assert_optimal() return esc(:(status == JuMP.OPTIMAL || error("$status"))) end
macro add_β1() return esc(:(JuMP.@variable(ø, β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))]))) end
macro add_β2() return esc(:(JuMP.@variable(ø, β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]))) end
macro addMatVarViaCopy(x, xΓ) return esc(:(JuMP.@variable(ø, $x[eachindex(eachrow($xΓ)), eachindex(eachcol($xΓ))]))) end
macro addMatCopyConstr(cpx, x, xΓ) return esc(:(JuMP.@constraint(ø, $cpx[i = eachindex(eachrow($x)), j = eachindex(eachcol($x))], $x[i, j] == $xΓ[i, j]))) end
macro optimise() return esc(:((_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø)))) end
macro reoptimise()
    return esc(quote
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
    end)
end
is_finite(r) = -Inf < r < Inf
rgapsimple(l, u) = abs(u - l) / max(abs(l), abs(u))
function rgap(l, u)
    is_finite(l) && is_finite(u) && return rgapsimple(l, u)
    return Inf
end
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # JuMP.set_attribute(ø, "QCPDual", 1)
    elseif i == 1 
        ø = JuMP.Model(MosekTools.Optimizer) # vio = JuMP.get_attribute(ø, Gurobi.ModelAttribute("MaxVio")) 🍀 we suggest trial-and-error to decide if a constr is tight, rather than using this cmd
    elseif i == 2 
        ø = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(ø) # JuMP.unset_silent(ø)
    return ø
end
function swap_i_j_in(v, i, j) return v[i], v[j] = v[j], v[i] end
function is_in(MY, yM) # strengthened version
    NY = size(yM, 3)
    ø = JumpModel(0)
    JuMP.@variable(ø, c[1:NY] >= 0)
    JuMP.@constraint(ø, sum(c) == 1)
    JuMP.@constraint(ø, sum(yM[:, :, i] * c[i] for i in 1:NY) .== MY)
    @optimise()
    status != JuMP.OPTIMAL && return false
    c = jv(c)
    @assert all(c .>= 0)
    @assert isapprox(sum(c), 1.0; atol=1e-7)
    return true
end
function is_MY_in_int_yM(distance = 0.007)
    for i in eachindex(MY)
        (a = zero(MY); a[i] = distance)
        @assert is_in(MY + a, yM)
        @assert is_in(MY - a, yM)
    end
end

ℶ1, ℶ2, Δ2 = let
    ℶ1 = Dict(
        # "x" =>  BitMatrix[], # contain x only, where u, v can be decoded from
        # "rv" => Int[], # index of Y
        "st" => Bool[],
        "cn" => Float64[],
        "pu" => Matrix{Float64}[],
        "pv" => Matrix{Float64}[],
        "px" => Matrix{Float64}[],
        "pβ" => Matrix{Float64}[] # slope of β1
    )
    ℶ2 = Dict(
        # "rv" is negation of pβ, thus dropped
        "st" => Bool[],
        "cn" => Float64[],
        "pu" => Matrix{Float64}[],
        "pv" => Matrix{Float64}[],
        "px" => Matrix{Float64}[],
        "pY" => Matrix{Float64}[],
        "pβ" => Matrix{Float64}[] # slope of β2
    )
    Δ2 = Dict( # 🌸 used in argmaxY
        "f" => Float64[],
        "u" => Matrix{Float64}[],
        "v" => Matrix{Float64}[],
        "x" => Matrix{Float64}[],
        "Y" => Int[],
        "β" => Matrix{Float64}[] # β2
    )
    ℶ1, ℶ2, Δ2
end
Random.seed!(3) # include("src/FM.jl") # Node 69 is going to be swapped with Node 118

POWER_DEN, COST_DEN = 100, 1500
T = 4 # 🫖
B = 186
# 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 
NW = [1, 6, 9, 18, 19, 41, 43, 62, 63, 72, 80]; # node number of wind units
W = 11
begin
    YABSMAX = 1/POWER_DEN * [65, 626, 742, 212, 63, 512, 66, 153, 87, 45, 40]
    Pr = Distributions.Uniform.(0.2 * rand(W), YABSMAX / 1.1)
    Ybaseline = [rand(Pr[w]) for t in 1:T, w in 1:W]
    YMIN, YMAX = 0.9 * Ybaseline, 1.1 * Ybaseline # used to construct vertices
    all0() = Bool.(zero(YMIN))
    vertexList = [all0() for _ in eachindex(YMIN)]
    for i in eachindex(YMIN)
        vertexList[i][i] = true
    end
    pushfirst!(vertexList, all0())
    for i in 1:1+length(YMIN)
        push!(vertexList, .!(vertexList[i]))
    end
    ref = vertexList[1]
    yM = [(ref[i,j] ? YMAX[i,j] : YMIN[i,j]) for i in eachindex(eachrow(YMIN)), j in eachindex(eachcol(YMIN))]
    yM = reshape(yM, (T, W, 1))
    popfirst!(vertexList)
    for ref in vertexList
        yM = cat(yM, [(ref[i,j] ? YMAX[i,j] : YMIN[i,j]) for i in eachindex(eachrow(YMIN)), j in eachindex(eachcol(YMIN))]; dims = 3)
    end # ✅ after this line, yM is already gened
    NY = size(yM, 3)
    PrVec = rand(Distributions.Uniform(.4, .6), NY)
    PrVec = PrVec / sum(PrVec)
    MY = sum(yM[:, :, i] * PrVec[i] for i in 1:NY)
    @assert all(MY .< YMAX) && all(MY .> YMIN)
    function is_strict_in(MY, yM)
        NY = size(yM, 3)
        ø = JumpModel(0)
        JuMP.@variable(ø, c[1:NY] >= 1e-5)
        JuMP.@constraint(ø, sum(c) == 1)
        JuMP.@constraint(ø, sum(yM[:, :, i] * c[i] for i in 1:NY) .== MY)
        @optimise()
        status == JuMP.OPTIMAL && return true
        return false
    end
    @assert is_strict_in(MY, yM)
end # ✅  at this line, we have a valid (MY, yM)
# 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 
NL = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 66, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118]; # node number of loads
LM = [51, 20, 39, 39, 52, 19, 28, 70, 47, 34, 14, 90, 25, 11, 60, 45, 18, 14, 10, 7, 13, 71, 17, 24, 43, 59, 23, 59, 33, 31, 27, 66, 37, 96, 18, 16, 53, 28, 34, 20, 87, 17, 17, 18, 23, 113, 63, 84, 12, 12, 277, 78, 77, 39, 28, 66, 12, 6, 68, 47, 68, 61, 71, 39, 130, 54, 20, 11, 24, 21, 48, 163, 10, 65, 12, 30, 42, 38, 15, 34, 42, 37, 22, 5, 23, 38, 31, 43, 50, 2, 8, 39, 68, 6, 8, 22, 184, 20, 33];
(flds = LM .<= 80; rvlds = LM .> 80);
fNL, fLM = NL[flds], LM[flds] 
rNL, rLM = NL[rvlds], LM[rvlds]
(fLM /= POWER_DEN; rLM /= POWER_DEN);
fL, rL = 90, 9 # Load = fixed load + random load
function gen_load_pattern(fLM)
    fL = length(fLM)
    fLP = rand(Distributions.Uniform(0.7 * fLM[1], 1.4 * fLM[1]), T, 1)
    for i in 2:fL
        fLP = [fLP rand(Distributions.Uniform(0.7 * fLM[i], 1.4 * fLM[i]), T)]
    end
    return fLP
end
fLP = gen_load_pattern(fLM)
rLP = MZ = gen_load_pattern(rLM)
# 📕 (minimum(fLP), maximum(fLP), minimum(MZ), maximum(MZ)) = (0.02, 0.99, 0.62, 2.98)
# 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 
G = 36 # 36 ordinary generators + 1 slack generator
UT = DT = 3
NG = [6, 8, 10, 12, 24, 25, 26, 27, 40, 42, 46, 49, 54, 59, 61, 62, 65, 66, 69, 72, 73, 76, 77, 80, 87, 89, 90, 91, 99, 100, 103, 104, 107, 110, 111, 113, 116];
PS = 1/POWER_DEN * [100, 100, 550, 185, 100, 320, 414, 100, 100, 100, 119, 304, 148, 255, 260, 100, 491, 492, 805, 100, 100, 100, 100, 577, 104, 707, 100, 100, 100, 352, 140, 100, 100, 100, 136, 100, 100];
PI = 1/POWER_DEN * [0, 10, 55, 20, 10, 30, 40, 10, 10, 10, 12, 30, 15, 25, 25, 10, 50, 50, 80, 10, 10, 0, 0, 60, 10, 70, 10, 10, 0, 35, 0, 0, 0, 0, 14, 10, 10];
C1 = POWER_DEN/COST_DEN * [20.22, 28.4, 29.36, 36.85, 26.64, 26.64, 26.64, 45.58, 37.34, 33.39, 33.39, 25.87, 53.21, 26.57, 31.77, 34.6, 24.44, 30.81, 48.13, 24.44, 24.01, 1.29, 31.45, 45.66, 30.21, 30.09, 33.39, 30.09, 1.29, 28.99, 2.0, 2.0, 1.29, 1.29, 33.39, 30.87, 26.64];
C0 = 1/COST_DEN * [0, 300, 1600, 70, 300, 800, 1100, 500, 400, 300, 400, 800, 200, 700, 800, 100, 1200, 1500, 4000, 250, 250, 0, 0, 300, 320, 2200, 340, 300, 0, 1000, 0, 0, 0, 0, 500, 300, 300];
CST = 1/COST_DEN * [2000, 3000, 16000, 7000, 2700, 8000, 11000, 5000, 4000, 3000, 4000, 8000, 4000, 7000, 8000, 2000, 12000, 15000, 40000, 2500, 2500, 20, 3200, 30000, 3200, 22000, 3400, 3000, 0, 10000, 0, 0, 0, 0, 5000, 3100, 2700];
EM = 1.1 * (C1 .* PS .+ C0); # 1.1 is a safe coefficient
i_2bswapped = findfirst(n -> n == 69, NG); # paired with G+1
swap_i_j_in(NG, i_2bswapped, G+1)
swap_i_j_in(PS, i_2bswapped, G+1)
swap_i_j_in(PI, i_2bswapped, G+1)
swap_i_j_in(C1, i_2bswapped, G+1)
swap_i_j_in(C0, i_2bswapped, G+1)
swap_i_j_in(CST, i_2bswapped, G+1)
swap_i_j_in(EM, i_2bswapped, G+1)
sigGenIndVec = findall(p -> p >= sort(PS)[end-9], PS)
NG  =  NG[sigGenIndVec]
PS  =  PS[sigGenIndVec]
PI  =  PI[sigGenIndVec]
C1  =  C1[sigGenIndVec]
C0  =  C0[sigGenIndVec]
CST = CST[sigGenIndVec]
EM  =  EM[sigGenIndVec] # ❌⚠️ is it valid to use continuous relaxation, if we've used logical big-M constraint including EM
G = length(sigGenIndVec) - 1
ZS = trues(G+1)
ZP = (PI + PS)/2
CL = 1.1 * maximum(C1)
CG = 0.8 * CL
cno = CL * sum(fLP) # doesn't participate in optimization

macro set_β1_bound()
    return esc(quote
    JuMP.set_lower_bound.(β1, -B1BND)
    JuMP.set_upper_bound.(β1,  B1BND)
    end)
end
macro delete_β1_bound()
    return esc(quote
    JuMP.delete_lower_bound.(β1)
    JuMP.delete_upper_bound.(β1)
    end)
end
macro Zfeas_code()
    return esc(quote
    JuMP.@variable(ø, 0.9 * MZ[t, l] <= Z[t = 1:T, l = 1:rL] <= 1.1 * MZ[t, l])
    JuMP.@variable(ø, adZ[t = 1:T, l = 1:rL])
    JuMP.@constraint(ø, [t = 1:T, l = 1:rL], adZ[t, l] >= Z[t, l] - MZ[t, l])
    JuMP.@constraint(ø, [t = 1:T, l = 1:rL], adZ[t, l] >= MZ[t, l] - Z[t, l])
    JuMP.@constraint(ø, [t = 1:T], sum(adZ[t, :] ./ (0.1 * MZ[t, :])) <= rL/3)
    end)
end
macro uvxfeas_code_con()
    return esc(quote
    JuMP.@variable(ø, 0 <= u[t = 1:T, g = 1:G+1] <= 1)
    JuMP.@variable(ø, 0 <= v[t = 1:T, g = 1:G+1] <= 1)
    JuMP.@variable(ø, 0 <= x[t = 1:T, g = 1:G+1] <= 1)
    JuMP.@constraint(ø, x .- vcat(transpose(ZS), x)[1:end-1, :] .== u .- v)
    JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-UT+1], sum(x[i, g] for i in t:t+UT-1) >= UT * u[t, g])
    JuMP.@constraint(ø, [g = 1:G+1, t = T-UT+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0)
    JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-DT+1], sum(1 - x[i, g] for i in t:t+DT-1) >= DT * v[t, g])
    JuMP.@constraint(ø, [g = 1:G+1, t = T-DT+1:T], sum(1 - x[i, g] - v[t, g] for i in t:T) >= 0)
    end)
end
macro uvxfeas_code_int()
    return esc(quote
    JuMP.@variable(ø, u[t = 1:T, g = 1:G+1], Bin)
    JuMP.@variable(ø, v[t = 1:T, g = 1:G+1], Bin)
    JuMP.@variable(ø, x[t = 1:T, g = 1:G+1], Bin)
    JuMP.@constraint(ø, x .- vcat(transpose(ZS), x)[1:end-1, :] .== u .- v)
    JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-UT+1], sum(x[i, g] for i in t:t+UT-1) >= UT * u[t, g])
    JuMP.@constraint(ø, [g = 1:G+1, t = T-UT+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0)
    JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-DT+1], sum(1 - x[i, g] for i in t:t+DT-1) >= DT * v[t, g])
    JuMP.@constraint(ø, [g = 1:G+1, t = T-DT+1:T], sum(1 - x[i, g] - v[t, g] for i in t:T) >= 0)
    end)
end
macro primobj_code()
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G+1]  >= 0) # effective power output
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W]    >= 0) # effective wind output
    JuMP.@variable(ø, fζ[t = 1:T, l = 1:fL]  >= 0) # effective fixed load
    JuMP.@variable(ø, rζ[t = 1:T, l = 1:rL]  >= 0) # effective realized random load
    JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0) # generation cost epi-variable
    JuMP.@constraint(ø, Dbal[t = 1:T], sum(fζ[t, :]) + sum(rζ[t, :]) == sum(ϱ[t, :]) + sum(ϖ[t, :]))
    JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= C1[g] * p[t, g] + C0[g] - EM[g] * (1 - x[t, g]))
    JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
    JuMP.@constraint(ø, Dfzt[t = 1:T, l = 1:fL], fLP[t, l] >= fζ[t, l]) # fLP is a fixed param
    JuMP.@constraint(ø, Drzt[t = 1:T, l = 1:rL], Z[t, l] >= rζ[t, l])
    JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G+1], p[t, g] >= ϱ[t, g])
    JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
    JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
    JuMP.@expression(ø, gencost, sum(pe))
    JuMP.@expression(ø, gccost, CG * sum(p .- ϱ))
    JuMP.@expression(ø, fLsCost2, -CL * sum(fζ))
    JuMP.@expression(ø, rLsCost, -CL * sum(rζ)) # want to split it in the opponent's problem
    JuMP.@expression(ø, OFC, CL * sum(Z)) # 🥑 Z is fixed as a param during cut generation
    JuMP.@expression(ø, primobj, gencost + gccost + fLsCost2 + rLsCost + OFC)
    end)
end
macro dualobj_code()
    return esc(quote
    JuMP.@variable(ø, Dbal[t = 1:T])
    JuMP.@variable(ø, 0 <= De[t = 1:T, g = 1:G+1] <= 1)
    JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0)
    JuMP.@variable(ø, Dfzt[t = 1:T, l = 1:fL] >= 0)
    JuMP.@variable(ø, Drzt[t = 1:T, l = 1:rL] >= 0)
    JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@constraint(ø, p[t = 1:T, g = 1:G+1], CG + C1[g] * De[t, g] - Dvr[t, g] + Dps[t, g] - Dpi[t, g] == 0)
    JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G+1], -CG + Dvr[t, g] + Dbal[t] >= 0)
    JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + Dbal[t] >= 0)
    JuMP.@constraint(ø, fζ[t = 1:T, l = 1:fL], -CL - Dbal[t] + Dfzt[t, l] >= 0)
    JuMP.@constraint(ø, rζ[t = 1:T, l = 1:rL], -CL - Dbal[t] + Drzt[t, l] >= 0)
    JuMP.@expression(ø, OFC, CL * sum(Z)) # 🥑
    JuMP.@expression(ø, dualobj, OFC
        + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
        + sum(x[t, g] * (PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) for t in 1:T, g in 1:G+1)
        - ip(Dvp, Y) - ip(Dfzt, fLP) - ip(Drzt, Z)
    )
    end)
end
function primobj_value(u, v, x, Y, Z) # f
    ø = JumpModel(0)
    @primobj_code()
    JuMP.@objective(ø, Min, primobj)
    @optimise()
    @assert_optimal()
    JuMP.objective_value(ø)
end
function dualobj_value(u, v, x, Y, Z) # f
    ø = JumpModel(0)
    @dualobj_code()
    JuMP.@objective(ø, Max, dualobj)
    @optimise()
    @assert_optimal()
    JuMP.objective_value(ø)
end
function master_con() # when ℶ1 is empty
    ø = JumpModel(0)
    @uvxfeas_code_con()
    JuMP.@expression(ø, o1, ip(brcs(CST), u))
    @add_β1()
    @set_β1_bound()
    JuMP.@expression(ø, o2, ip(MY, β1))
    JuMP.@objective(ø, Min, o1 + o2)
    @optimise()
    @assert_optimal()
    u = jv(u)
    v = jv(v)
    x = jv(x)
    β1 = jv(β1)
    return u, v, x, β1
end
function readCutℶ2(ℶ2)
    stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ2["st"], ℶ2["cn"], ℶ2["pu"], ℶ2["pv"], ℶ2["px"], ℶ2["pY"], ℶ2["pβ"]
    R2 = length(cnV2)
    return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
end
function get_trial_β2_oℶ2() # when ℶ2 is empty
    ø = JumpModel(0)
    @add_β2()
    JuMP.@objective(ø, Min, ip(MZ, β2))
    (JuMP.set_lower_bound.(β2, -B2BND); JuMP.set_upper_bound.(β2, B2BND))
    vldtV[2] = false
    @optimise()
    @assert_optimal()
    return β2, oℶ2 = jv(β2), -Inf
end
function get_trial_β2_oℶ2(ℶ2, u, v, x, Y) # invoke next to argmaxY
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCutℶ2(ℶ2)
    ø = JumpModel(0)
    @add_β2()
    JuMP.@variable(ø, o2)
    for r in 1:R2
        if stV2[r]
            tmp = [(cnV2[r], 1), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], Y), (pβ2V2[r], β2)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o2 >= cut_expr)
        end
    end
    JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        @assert status == JuMP.DUAL_INFEASIBLE
        (JuMP.set_lower_bound.(β2, -B2BND); JuMP.set_upper_bound.(β2, B2BND))
        @optimise()
        @assert_optimal()
        vldtV[2] = false
    end
    return β2, oℶ2 = jv(β2), JuMP.value(o2)
end
function argmaxZ(u, v, x, Y, β2) # 💻 Feat
    ø = JumpModel(2)
    @Zfeas_code() # def of Z included
    @dualobj_code()
    JuMP.@objective(ø, Max, -ip(Z, β2) + dualobj) # dualobj is f's
    t_start = time()
    @optimise()
    t_interval = time() - t_start
    t_interval > 3600 && @error("⋅⋅⋅⋅⋅$t_interval⋅⋅⋅⋅⋅")
    @assert_optimal()
    return jv(Z)
end
phi_2(u, v, x, Y, Z, β2) = -ip(β2, Z) + primobj_value(u, v, x, Y, Z) # ✅ phi_2 is eval by def, Not an estimate via Δ2 ⚠️
function evalPush_Δ2(u, v, x, yM, iY, Z, β2)
    φ2_via_model = ub_φ2(u, v, x, iY, β2)
    φ2_via_eval = phi_2(u, v, x, yM[:, :, iY], Z, β2)
    φ2_via_eval < φ2_via_model - UPDTH || return true # Δ2 is saturated
    push!(Δ2["f"], φ2_via_eval)
    push!(Δ2["u"], u)
    push!(Δ2["v"], v)
    push!(Δ2["x"], x)
    push!(Δ2["Y"], iY)
    push!(Δ2["β"], β2)
    return false # Δ2 is Not saturated (= strongly updated)
end
function gencut_f_uvxY(Z, uΓ, vΓ, xΓ, YΓ) # Ben cut
    ø = JumpModel(0) # if we don't have Quad
    @addMatVarViaCopy(u, uΓ)
    @addMatVarViaCopy(v, vΓ)
    @addMatVarViaCopy(x, xΓ)
    @addMatVarViaCopy(Y, YΓ)
    @addMatCopyConstr(cpu, u, uΓ)
    @addMatCopyConstr(cpv, v, vΓ)
    @addMatCopyConstr(cpx, x, xΓ)
    @addMatCopyConstr(cpY, Y, YΓ)
    @primobj_code()
    JuMP.@objective(ø, Min, primobj) # obj must be the convex function you want to build CTPLN model for
    @optimise()
    @assert_optimal()
    obj = jo(ø)
    pu  = jd(cpu)
    pv  = jd(cpv)
    px  = jd(cpx)
    pY  = jd(cpY)
    cn = obj - ip(pu, uΓ) - ip(pv, vΓ) - ip(px, xΓ) - ip(pY, YΓ) 
    return cn, pu, pv, px, pY
end
function gencut_ℶ2(Z, yM, iY, of, u, v, x) # decorator
    cn, pu, pv, px, pY = gencut_f_uvxY(Z, u, v, x, yM[:, :, iY])
    pβ2 = -Z
    return cn, pu, pv, px, pY, pβ2
end
function tryPush_ℶ2(Z, yM, iY, oℶ2, u, v, x, β2) # 👍 use this directly
    cn, pu, pv, px, pY, pβ2 = gencut_ℶ2(Z, yM, iY, NaN, u, v, x) # you'll always gen a cut with cn being finite
    new_oℶ2 = cn + ip(pu, u) + ip(pv, v) + ip(px, x) + ip(pY, yM[:, :, iY]) + ip(pβ2, β2)
    new_oℶ2 > oℶ2 + UPDTH || (cutSuccV[2] = false; return)
    push!(ℶ2["st"], true)
    push!(ℶ2["cn"], cn)
    push!(ℶ2["pu"], pu)
    push!(ℶ2["pv"], pv)
    push!(ℶ2["px"], px)
    push!(ℶ2["pY"], pY)
    push!(ℶ2["pβ"], pβ2)
end
macro λmethod_code()
    return esc(quote
        i_vec = findall(r -> r == iY, Δ2["Y"])
        isempty(i_vec) && return Inf
        R2 = length(i_vec)
        uV2, vV2, xV2 = Δ2["u"][i_vec], Δ2["v"][i_vec], Δ2["x"][i_vec]
        β2V2, fV2 = Δ2["β"][i_vec], Δ2["f"][i_vec]
        ø = JumpModel(0)
        JuMP.@variable(ø, λ[1:R2] >= 0)
        JuMP.@constraint(ø, sum(λ) == 1)
        JuMP.@constraint(ø,  sum(uV2[r] * λ[r] for r in 1:R2) .==  u)
        JuMP.@constraint(ø,  sum(vV2[r] * λ[r] for r in 1:R2) .==  v)
        JuMP.@constraint(ø,  sum(xV2[r] * λ[r] for r in 1:R2) .==  x)
    end)
end
ub_φ1(u, v, x, yM, iY, β1) = -ip(β1, yM[:, :, iY]) + ub_psi(u, v, x, iY)
function ub_psi(u, v, x, iY)
    @λmethod_code()
    @add_β2()
    JuMP.@constraint(ø, sum(β2V2[r] * λ[r] for r in 1:R2) .== β2)
    JuMP.@objective(ø, Min, ip(MZ, β2) + ip(fV2, λ))
    @optimise()
    status != JuMP.OPTIMAL && return Inf
    return JuMP.objective_value(ø)
end
function ub_φ2(u, v, x, iY, β2) # used in admission ctrl during recruitment of Δ2
    @λmethod_code()
    JuMP.@constraint(ø, sum(β2V2[r] * λ[r] for r in 1:R2) .== β2)
    JuMP.@objective(ø, Min, ip(fV2, λ))
    @optimise()
    status != JuMP.OPTIMAL && return Inf
    return JuMP.objective_value(ø)
end
# 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃
vldtV, cutSuccV = trues(2), trues(2)
function initialize_Δ2_and_ℶ2()
    u, v, x, β1 = master_con()
    iY = rand(1:size(yM, 3))
    vldtV[2] = true
    β2, oℶ2 = get_trial_β2_oℶ2()
    Z = argmaxZ(u, v, x, yM[:, :, iY], β2)
    evalPush_Δ2(u, v, x, yM, iY, Z, β2) # primobj_value(u, v, x, yM[:, :, iY], Z)
    cutSuccV[2] = true
    tryPush_ℶ2(Z, yM, iY, oℶ2, u, v, x, β2)
    @assert(length(Δ2["f"]) == 1 && length(ℶ2["cn"]) == 1)
    @info " 🥑 Δ2 and ℶ2 is ready"
end
initialize_Δ2_and_ℶ2()
# 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃 🍃
function argmaxindY(u, v, x, yM, β1)
    (NY = size(yM, 3); fullVec = zeros(NY))
    indexVector = Random.shuffle(1:NY) # 🫖 crucial 🫖
    for iY in indexVector
        val = ub_φ1(u, v, x, yM, iY, β1)
        val == Inf && return iY
        fullVec[iY] = val
    end
    return findmax(fullVec)[2]
end
function gencut_ψ_uvx(yM, iY, uΓ, vΓ, xΓ)
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCutℶ2(ℶ2)
    # R2 == 0 && return -Inf, zero(uΓ), zero(vΓ), zero(xΓ) # cn, pu, pv, px
    ø = JumpModel(0)
    @addMatVarViaCopy(u, uΓ)
    @addMatVarViaCopy(v, vΓ)
    @addMatVarViaCopy(x, xΓ)
    @addMatCopyConstr(cpu, u, uΓ)
    @addMatCopyConstr(cpv, v, vΓ)
    @addMatCopyConstr(cpx, x, xΓ)
    @add_β2()
    JuMP.@variable(ø, o2)
    for r in 1:R2
        if stV2[r]
            Y = yM[:, :, iY] # fixed parameter
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], Y), (pβ2V2[r], β2)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o2 >= cut_expr)
        end
    end
    JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    # JuMP.unset_silent(ø)
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        status == JuMP.DUAL_INFEASIBLE && return -Inf, zero(uΓ), zero(vΓ), zero(xΓ) # cn, pu, pv, px
        error("$status")
    end
    obj = jo(ø)
    pu  = jd(cpu)
    pv  = jd(cpv)
    px  = jd(cpx)
    cn = obj - ip(pu, uΓ) - ip(pv, vΓ) - ip(px, xΓ)
    return cn, pu, pv, px
end
function gencut_ℶ1(yM, iY, oψ, u, v, x) # decorator
    cn, pu, pv, px = gencut_ψ_uvx(yM, iY, u, v, x)
    pβ1 = -yM[:, :, iY]
    return cn, pu, pv, px, pβ1
end
function tryPush_ℶ1(yM, iY, oℶ1, u, v, x, β1)
    cn, pu, pv, px, pβ1 = gencut_ℶ1(yM, iY, NaN, u, v, x)
    cn == -Inf && (cutSuccV[1] = false; return false) # (No push! to ℶ1; No saturation)
    new_oℶ1 = cn + ip(pu, u) + ip(pv, v) + ip(px, x) + ip(pβ1, β1)
    new_oℶ1 > oℶ1 + UPDTH || (cutSuccV[1] = false; return true) # (No push! to ℶ1 due to {saturation = true})
    push!(ℶ1["st"], true)
    push!(ℶ1["cn"], cn)
    push!(ℶ1["pu"], pu)
    push!(ℶ1["pv"], pv)
    push!(ℶ1["px"], px)
    push!(ℶ1["pβ"], pβ1)
    return false # No saturation
end
# 🍃 🍃  🍃 🍃  🍃 🍃  🍃 🍃  🍃 🍃 
function initialize_ℶ1()
    u, v, x, β1 = master_con()
    while true # if B2BND = 3.2, you will converge to a stagnant state by @info message. If B2BND = 3.6, you'll have vldtV[2] = true and leave successfully 
        iY = argmaxindY(u, v, x, yM, β1)
        vldtV[2] = true
        β2, oℶ2 = get_trial_β2_oℶ2(ℶ2, u, v, x, yM[:, :, iY])
        Z = argmaxZ(u, v, x, yM[:, :, iY], β2)
        evalPush_Δ2(u, v, x, yM, iY, Z, β2)
        @debug "iY = $iY, vldtV[2] = $(vldtV[2])" β2
        cutSuccV[2] = true
        tryPush_ℶ2(Z, yM, iY, oℶ2, u, v, x, β2)
        cutSuccV[1] = true
        saturated = tryPush_ℶ1(yM, iY, -Inf, u, v, x, β1)
        saturated && @error "ℶ1 is saturated during its initialization"
        cutSuccV[1] && break
    end
    @assert length(ℶ1["cn"]) == 1
    @info " 🥑 ℶ1 is ready"
end
initialize_ℶ1()
function build_sufficient_ℶ1()
    ø = JumpModel(0)
    @uvxfeas_code_con()
    JuMP.@expression(ø, o1, ip(brcs(CST), u))
    @add_β1()
    JuMP.@expression(ø, o2, ip(MY, β1))
    JuMP.@variable(ø, o3)
    ı = 1
    cut_expr = ℶ1["cn"][ı] + ip(ℶ1["pu"][ı], u) + ip(ℶ1["pv"][ı], v) + ip(ℶ1["px"][ı], x) + ip(ℶ1["pβ"][ı], β1)
    JuMP.@constraint(ø, o3 >= cut_expr)
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    while true # this loop may take long time but the lower bound is growing continually, at night of 15/12/24
        @optimise()
        status == JuMP.OPTIMAL && break
        @set_β1_bound()
        @optimise()
        @assert_optimal()
        u_, v_, x_, β1_ = jv(u), jv(v), jv(x), jv(β1)
        iY = argmaxindY(u_, v_, x_, yM, β1_)
        vldtV[2] = true
        β2, oℶ2 = get_trial_β2_oℶ2(ℶ2, u_, v_, x_, yM[:, :, iY])
        Z = argmaxZ(u_, v_, x_, yM[:, :, iY], β2)
        evalPush_Δ2(u_, v_, x_, yM, iY, Z, β2)
        iY = argmaxindY(u_, v_, x_, yM, β1_)
        cutSuccV[2] = true
        tryPush_ℶ2(Z, yM, iY, oℶ2, u_, v_, x_, β2)
        cutSuccV[1] = true
        tryPush_ℶ1(yM, iY, -Inf, u_, v_, x_, β1_)
            # beta1 = round.(β1_; digits = 1)
            # u_1, v_1, x_1 = round.(u_; digits = 1), round.(v_; digits = 1), round.(x_; digits = 1)
            # beta2 = round.(β2; digits = 1)
            lb = cno + jo(ø)
            @info "ite = $ı" iY vldtV[2] lb;
        if cutSuccV[1]
            ı += 1
            cut_expr = ℶ1["cn"][ı] + ip(ℶ1["pu"][ı], u) + ip(ℶ1["pv"][ı], v) + ip(ℶ1["px"][ı], x) + ip(ℶ1["pβ"][ı], β1)
            JuMP.@constraint(ø, o3 >= cut_expr)
        end
        @delete_β1_bound()
    end
    @info "🥑 β1 now has auto-bound"
end
build_sufficient_ℶ1()
# 🍃 🍃  🍃 🍃  🍃 🍃  🍃 🍃  🍃 🍃  MAIN integer Pragram
function get_β2_oℶ2(ℶ2, u, v, x, Y) # A concise version
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCutℶ2(ℶ2)
    ø = JumpModel(0)
    @add_β2()
    JuMP.@variable(ø, o2)
    for r in 1:R2
        tmp = [(cnV2[r], 1), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], Y), (pβ2V2[r], β2)]
        cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
        JuMP.drop_zeros!(cut_expr)
        JuMP.@constraint(ø, o2 >= cut_expr)
    end
    JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    @optimise()
    @assert_optimal()
    return β2, oℶ2 = jv(β2), JuMP.value(o2)
end
function Mi_master_without_callback() # as a template for reference
    ø = JumpModel(0)
    @uvxfeas_code_int()
    JuMP.@expression(ø, o1, ip(brcs(CST), u))
    @add_β1()
    JuMP.@expression(ø, o2, ip(MY, β1))
    JuMP.@variable(ø, o3)
    ini_len = length(ℶ1["cn"])
    ı = 0
    while ı+1 <= ini_len
        ı += 1
        cut_expr = ℶ1["cn"][ı] + ip(ℶ1["pu"][ı], u) + ip(ℶ1["pv"][ı], v) + ip(ℶ1["px"][ı], x) + ip(ℶ1["pβ"][ı], β1)
        JuMP.@constraint(ø, o3 >= cut_expr)
    end
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    JuMP.unset_silent(ø)
    ubs = Inf
    while true
        @optimise()
        if status == JuMP.OPTIMAL
            o1po2, lbℶ1 = JuMP.value(o1) + JuMP.value(o2), JuMP.value(o3)
            u_, v_, x_, β1_ = jv(u), jv(v), jv(x), jv(β1)
        else
            error("intMaster(): Sstatus")
        end
        iY = argmaxindY(u_, v_, x_, yM, β1_)
        ubℶ1 = ub_φ1(u_, v_, x_, yM, iY, β1_)
        β2, oℶ2 = get_β2_oℶ2(ℶ2, u_, v_, x_, yM[:, :, iY])
        Z = argmaxZ(u_, v_, x_, yM[:, :, iY], β2)
        Δ2_saturated = evalPush_Δ2(u_, v_, x_, yM, iY, Z, β2)
            ubℶ1_invalid = ub_φ1(u_, v_, x_, yM, iY, β1_)
            ub = o1po2 + ubℶ1
            ub_invalid = o1po2 + ubℶ1_invalid
            lb = o1po2 + lbℶ1
            ubs = min(ubs, ub)
            @info "($ı)[+$cno] lb = $lb | $ubs = ubs `$ub_invalid`"
        cutSuccV[2] = true
        tryPush_ℶ2(Z, yM, iY, oℶ2, u_, v_, x_, β2)
        cutSuccV[1] = true
        ℶ1_saturated = tryPush_ℶ1(yM, iY, lbℶ1, u_, v_, x_, β1_)
        if cutSuccV[1]
            ı += 1
            cut_expr = ℶ1["cn"][ı] + ip(ℶ1["pu"][ı], u) + ip(ℶ1["pv"][ı], v) + ip(ℶ1["px"][ı], x) + ip(ℶ1["pβ"][ı], β1)
            JuMP.@constraint(ø, o3 >= cut_expr)
        end
        if Δ2_saturated
            if ℶ1_saturated
                if cutSuccV[2]
                    @info "ℶ1, Δ2 S⋅A⋅T"
                else
                    @info "ℶ2, ℶ1, Δ2 S⋅A⋅T, thus break"
                    break # if ain't, the logging afterwards will be unaltered ⭐
                end
            else
                @assert cutSuccV[1] "No saturation and No push! to ℶ1"
                cutSuccV[2] || @info "ℶ2, Δ2 S⋅A⋅T"
            end
        else
            if ℶ1_saturated
                if cutSuccV[2]
                    @info "ℶ1 S⋅A⋅T"
                else
                    @info "ℶ2, ℶ1 S⋅A⋅T"
                end
            else
                @assert cutSuccV[1] "No saturation and No push! to ℶ1"
                cutSuccV[2] || @info "ℶ2 S⋅A⋅T"
            end
        end
    end
end
function my_callback_function(cb_data, cb_where::Cint)
    jvcb_scalar(x) = JuMP.callback_value(cb_data, x)
    jvcb(x) = jvcb_scalar.(x)
    cb_where == Gurobi.GRB_CB_MIPSOL || return
    Gurobi.load_callback_variable_primal(cb_data, cb_where)
    o1po2, lbℶ1 = jvcb_scalar(o1) + jvcb_scalar(o2), jvcb_scalar(o3)
    u_, v_, x_, β1_ = jvcb(u), jvcb(v), jvcb(x), jvcb(β1)
    while true # must generate violating cut, or terminate
        iY = argmaxindY(u_, v_, x_, yM, β1_)
        ubℶ1 = ub_φ1(u_, v_, x_, yM, iY, β1_)
        β2, oℶ2 = get_β2_oℶ2(ℶ2, u_, v_, x_, yM[:, :, iY])
        Z = argmaxZ(u_, v_, x_, yM[:, :, iY], β2)
        Δ2_saturated = evalPush_Δ2(u_, v_, x_, yM, iY, Z, β2)
        cutSuccV[2] = true
        tryPush_ℶ2(Z, yM, iY, oℶ2, u_, v_, x_, β2)
        cutSuccV[1] = true
        ℶ1_saturated = tryPush_ℶ1(yM, iY, lbℶ1, u_, v_, x_, β1_)
        if ℶ1_saturated
            if Δ2_saturated && cutSuccV[2] == false
                @info "🥑 ℶ2, ℶ1, Δ2 S⋅A⋅T, thus return without a violating cut"
                return
            end
        else
            cutSuccV[1] || @error "ℶ1 is unupdated when it is unsaturated"
            ı = ℶ1CntV[1]
            ı += 1
            cut_expr = ℶ1["cn"][ı] + ip(ℶ1["pu"][ı], u) + ip(ℶ1["pv"][ı], v) + ip(ℶ1["px"][ı], x) + ip(ℶ1["pβ"][ı], β1)
            JuMP.MOI.submit(ø, JuMP.MOI.LazyConstraint(cb_data), JuMP.@build_constraint(o3 >= cut_expr))
            ℶ1CntV[1] = ı
            return
        end
    end
end
ℶ1CntV = [length(ℶ1["cn"])]
UPDTH = 1.0 # only a update greater than this threshold will be performed
# .....................................
ø = JumpModel(2)
@uvxfeas_code_int()
JuMP.@expression(ø, o1, ip(brcs(CST), u))
@add_β1()
JuMP.@expression(ø, o2, ip(MY, β1))
JuMP.@variable(ø, o3)
ı, ini_len = 0, length(ℶ1["cn"])
while ı+1 <= ini_len
    ı += 1
    cut_expr = ℶ1["cn"][ı] + ip(ℶ1["pu"][ı], u) + ip(ℶ1["pv"][ı], v) + ip(ℶ1["px"][ı], x) + ip(ℶ1["pβ"][ı], β1)
    JuMP.@constraint(ø, o3 >= cut_expr)
end
# ℶ1CntV ≡ [ı] at this line
JuMP.@objective(ø, Min, o1 + o2 + o3) # cno also
JuMP.MOI.set(ø, JuMP.MOI.RawOptimizerAttribute("LazyConstraints"), 1)
JuMP.MOI.set(ø, Gurobi.CallbackFunction(), my_callback_function)
JuMP.unset_silent(ø)
@optimise() # --> goto callback
@warn "Mi_master_with_callback is over!"


Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (win64 - Windows 11.0 (22631.2))

CPU model: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, instruction set [SSE2|AVX|AVX2|AVX512]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 388 rows, 165 columns and 21570 nonzeros
Model fingerprint: 0x18ba1999
Variable types: 45 continuous, 120 integer (120 binary)
Coefficient statistics:
  Matrix range     [4e-04, 3e+01]
  Objective range  [8e-02, 3e+01]
  Bounds range     [0e+00, 0e+00]
  RHS range        [1e+00, 5e+02]
Presolve removed 44 rows and 14 columns
Presolve time: 0.02s
Presolved: 344 rows, 151 columns, 17884 nonzeros
Variable types: 56 continuous, 95 integer (80 binary)

Root relaxation: objective -3.403515e+02, 229 iterations, 0.02 seconds (0.02 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 -340.35147    0   55          - -340.35147      -     -    2s
     0     0 -339.00894    0   58          - -339.00894      -     -    2s
     0     0 -338.76498    0   58          - -338.76498      -     -    2s
     0     0 -338.76498    0   58          - -338.76498      -     -    2s
     0     0 -338.39511    0   51          - -338.39511      -     -    3s
     0     0 -338.35769    0   52          - -338.35769      -     -    3s
     0     0 -337.68513    0   52          - -337.68513      -     -    3s
     0     2 -337.65284    0   52          - -337.65284      -     -    3s
   307   276 -310.79819   11    3          - -330.43305      -   8.6    5s
  1350   971 -298.67104    9   35          - -323.90707      -   7.6   10s
  2000  1404 -277.65712   25   11          - -321.09019      -   8.0   15s
  2766  1712 -295.24604   31    -          - -310.05105      -   7.7   20s
  3330  1936 -294.70967   21    9          - -307.71500      -   7.6   25s
  3959  2185 -267.28335   27    -          - -305.83740      -   7.5   30s
  4704  2421 -256.32039   35    -          - -303.85340      -   7.3   35s
  5208  2625 -276.67956   18   24          - -302.73568      -   7.4   40s
  5882  2873 -276.12003   26   14          - -301.01585      -   7.4   45s
  6490  3274 -262.12382   32    6          - -300.11466      -   7.5   51s
  7157  3788 -273.58079   36    -          - -298.97026      -   7.6   55s
  7728  4164 -274.43631   25   14          - -298.36505      -   7.6   60s
  8596  4724 -259.33107   29    -          - -297.37041      -   7.7   66s
  9307  5183 -267.66271   25    8          - -296.44366      -   7.8   71s
 10145  5678 -248.74200   32    6          - -295.46903      -   7.9   76s
 11026  6318 -237.49885   30    -          - -294.77052      -   7.9   82s
 11404  6569 -242.17155   34    -          - -294.57428      -   7.9   86s
 12067  7005 -276.35423   22   11          - -294.03238      -   8.0   90s
 12872  7496 -230.29567   33    -          - -293.27566      -   8.1   98s
 13227  7729 -269.61609   23   12          - -293.07250      -   8.1  101s
 13561  7916 -257.75206   26    5          - -292.79176      -   8.2  105s
 14201  8330 -261.14246   22   19          - -291.98355      -   8.3  111s
 14581  8586 -255.93113   28    4          - -291.56853      -   8.3  116s
 14947  8792 -260.61958   38    -          - -291.30024      -   8.4  120s
 15557  9217 -272.52792   22   27          - -290.65727      -   8.5  127s
 15911  9455 -218.27156   37    -          - -290.33981      -   8.5  131s
 16275  9644 -255.81807   30    2          - -290.05857      -   8.5  135s
 16885 10072 -256.37397   22   15          - -289.29252      -   8.6  143s
 17247 10280 -252.83227   20    8          - -288.95357      -   8.6  146s
 17575 10479 -253.59352   28   13          - -288.73597      -   8.6  151s
 17880 10654 -249.06997   31    -          - -288.40538      -   8.7  156s
 18152 10830 -250.46259   24    9          - -288.12366      -   8.7  160s
 18824 11239 -233.46490   35    -          - -287.63551      -   8.7  168s
 19145 11452 -226.02793   34    4          - -287.23662      -   8.8  172s
 19405 11610 -248.04972   28    3          - -287.04889      -   8.8  176s
 19700 11802 -246.00017   26   16          - -286.88112      -   8.8  181s
 20281 12143 -238.84849   34    3          - -286.25816      -   8.9  190s
 20934 12513 -226.95833   32    4          - -285.53644      -   9.0  199s
 21205 12715 -273.63283   25   19          - -285.40308      -   9.0  204s
 21580 12919 -221.20676   34    5          - -285.24802      -   9.0  209s
 21885 13087 -254.62462   25   13          - -285.04810      -   9.0  216s
 22089 13224 -230.72493   30    3          - -284.86981      -   9.1  221s
 22405 13427 -239.92226   34    6          - -284.74672      -   9.1  225s
 22737 13625 -243.23126   31   15          - -284.53944      -   9.1  230s
 23054 13793 -230.95542   27   15          - -284.32098      -   9.1  237s
 23262 13965 -237.23045   33    -          - -284.10501      -   9.2  243s
 23576 14144 -241.46164   29    4          - -283.93992      -   9.2  247s
 23861 14308 -233.00136   31    3          - -283.80861      -   9.2  255s
 24118 14470 -219.82098   31    -          - -283.49387      -   9.3  262s
 24441 14671 -222.75997   29    9          - -283.21312      -   9.3  267s
 24697 14835 -223.91523   36    -          - -282.95927      -   9.3  273s
 24992 15022 -239.59729   23   16          - -282.81847      -   9.3  277s
 25265 15180 -236.03640   34    4          - -282.63600      -   9.3  282s
 25376 15222 -252.54051   24   19          - -282.60852      -   9.3  285s
 25776 15515 -227.24338   31    -          - -282.22713      -   9.4  293s
 26059 15700 -260.57494   22   26          - -282.07370      -   9.4  299s
 26343 15863 -224.03346   31    5          - -281.91863      -   9.4  304s
 26612 15996 -243.40208   32    3          - -281.67892      -   9.5  310s
 26794 16151 -244.18628   23    4          - -281.52747      -   9.5  316s
 27054 16323 -213.81265   30    3          - -281.34867      -   9.5  321s
 27343 16489 -238.76778   29   11          - -281.26626      -   9.5  326s
 27607 16637 -211.61651   31    -          - -281.05558      -   9.5  331s
 27880 16827 -254.45379   23   16          - -280.88260      -   9.6  340s
 28163 16985 -224.00895   28    4          - -280.64594      -   9.6  346s
 28368 17113 -226.94100   31    8          - -280.46924      -   9.6  350s
 28494 17159 -197.99162   36    -          - -280.44663      -   9.6  356s
 28913 17474 -239.39630   24    9          - -279.98227      -   9.7  365s
 29188 17607 -196.19040   34    -          - -279.88262      -   9.7  370s
 29403 17774 -205.58576   28    -          - -279.73403      -   9.7  376s
 29677 17934 -229.75418   24   17          - -279.50332      -   9.7  380s
 29935 18105 -258.36994   28   14          - -279.34495      -   9.7  388s
 30234 18263 -214.37331   31    -          - -279.23668      -   9.8  395s
 30457 18434 -234.70309   32    -          - -279.15833      -   9.8  400s
 30701 18587 -242.65934   23   13          - -279.05860      -   9.8  407s
 30971 18726 -242.78130   27   12          - -278.88838      -   9.8  415s
 31205 18892 -209.71106   32    2          - -278.70587      -   9.9  422s
 31484 19052 -247.29714   25   18          - -278.56176      -   9.9  430s
 31741 19203 -222.63921   29    -          - -278.31343      -   9.9  440s
 31955 19370 -259.58024   25   16          - -278.14489      -   9.9  446s
 32252 19537 -263.26483   21   26          - -278.06264      -   9.9  453s
 32515 19690 -227.79659   25   13          - -277.93287      -  10.0  459s
 32743 19865 -237.40996   29    6          - -277.78434      -  10.0  468s
 32972 19985 -238.08570   31    -          - -277.68353      -  10.0  476s
 33245 20147 -248.83714   22   23          - -277.51758      -  10.0  484s
 33469 20282 -212.02689   26    5          - -277.41509      -  10.0  491s
 33747 20475 -211.82163   33    -          - -277.30519      -  10.0  498s
 34003 20585 -227.02345   27   21          - -277.13606      -  10.1  506s
 34252 20771 -197.76018   34    -          - -277.05879      -  10.1  513s
 34502 20931 -217.08688   33    -          - -276.94095      -  10.1  523s
 34765 21061 -216.90662   29    9          - -276.78699      -  10.1  531s
 34999 21243 -230.06538   29    -          - -276.67846      -  10.1  541s
 35223 21345 -232.42911   28    -          - -276.60274      -  10.2  546s
 35356 21403 -255.17163   26   22          - -276.58184      -  10.2  552s
 35490 21505 -242.13946   32    -          - -276.46506      -  10.2  556s
 35678 21581 -234.97387   24   13          - -276.46126      -  10.2  560s
 36005 21823 -229.98410   34    -          - -276.36823      -  10.2  573s
 36226 21969 -228.85489   24   14          - -276.23416      -  10.2  581s
 36357 22011 -229.96780   26    9          - -276.16776      -  10.3  586s
 36662 22257 -237.01026   26   12          - -276.01730      -  10.3  595s
 36913 22380 -232.81779   36    -          - -275.90014      -  10.3  601s
 37118 22527 -209.94569   31    -          - -275.82609      -  10.3  613s
 37363 22671 -224.64673   36    -          - -275.69396      -  10.4  622s
 37517 22729 -235.88962   28    3          - -275.53417      -  10.4  630s
 37807 22979 -239.19455   27   10          - -275.44260      -  10.4  638s
 38058 23134 -241.87378   31    -          - -275.33860      -  10.4  646s
 38332 23287 -226.08215   29    -          - -275.23154      -  10.4  654s
 38532 23424 -239.55710   28    9          - -275.11755      -  10.5  664s
 38772 23590 -232.11556   28    6          - -274.99632      -  10.5  673s
 38817 23606 -221.80197   34    -          - -274.97622      -  10.5  676s
 39017 23716 -252.79150   24   10          - -274.87867      -  10.5  684s
 39203 23879 -241.75488   24   10          - -274.77939      -  10.5  691s
 39456 24008 -222.62266   35    -          - -274.65299      -  10.5  704s
 39662 24144 -234.15834   26   13          - -274.53316      -  10.6  715s
 39792 24191 -231.40819   33    8          - -274.48425      -  10.6  721s
 39913 24301 -218.91366   36    -          - -274.42518      -  10.6  727s
 40107 24409 -244.04687   28    4          - -274.34841      -  10.6  740s
 40366 24566 -240.25570   31    6          - -274.25001      -  10.6  750s
 40626 24721 -230.86997   23   24          - -274.11009      -  10.6  763s
 40677 24736 -224.58456   36    -          - -274.08452      -  10.6  766s
 40785 24831 -239.27296   35    -          - -274.00895      -  10.7  771s
 40929 24888 -244.54660   32    5          - -273.99764      -  10.7  777s
 41070 25016 -229.36505   29    8          - -273.87902      -  10.7  787s
 41330 25162 -229.34548   33    4          - -273.82222      -  10.7  796s
 41603 25332 -222.84385   34    4          - -273.69543      -  10.7  810s
 41816 25441 -240.08296   23    5          - -273.59031      -  10.7  819s
 42024 25599 -241.31734   24   20          - -273.52268      -  10.7  831s
 42164 25648 -227.93118   29    6          - -273.50259      -  10.7  837s
 42272 25747 -210.36150   32    2          - -273.44902      -  10.7  840s
 42523 25900 -233.04436   30    -          - -273.34399      -  10.8  853s
 42718 25999 -245.41029   33    2          - -273.29683      -  10.8  861s
 42905 26080 -235.20154   28    3          - -273.27349      -  10.8  867s
 43111 26239 -232.41898   28    -          - -273.23784      -  10.8  876s
 43229 26315 -225.56273   34    6          - -273.18360      -  10.8  882s
 43467 26464 -247.21417   23   31          - -273.10789      -  10.8  891s
 43645 26584 -231.11578   23    5          - -273.02280      -  10.8  899s
 43768 26630 -246.67286   26   20          - -272.99332      -  10.8  906s
 43886 26722 -201.76210   31    8          - -272.90638      -  10.8  914s
 43951 26747 -216.49674   36    -          - -272.89360      -  10.8  916s
 44147 26861 -241.90524   36    2          - -272.87498      -  10.8  924s
 44406 27028 -255.26103   23   16          - -272.74433      -  10.8  936s
 44585 27169 -194.90339   31    6          - -272.68530      -  10.9  943s
 44819 27306 -225.32899   29    6          - -272.61575      -  10.9  954s
 45066 27463 -207.96971   33    -          - -272.55366      -  10.9  965s
 45293 27579 -238.61015   27   20          - -272.48409      -  10.9  975s
 45471 27702 -242.27986   32   10          - -272.42136      -  10.9  984s
 45734 27872 -218.76817   25   19          - -272.36879      -  10.9  993s
 45982 28004 -213.45096   35    -          - -272.34981      -  10.9 1008s
 46176 28126 -229.01519   33    -          - -272.25994      -  11.0 1018s
 46415 28275 -209.74033   26    2          - -272.21607      -  11.0 1029s
 46478 28301 -229.74725   34    -          - -272.20931      -  11.0 1033s
 46625 28402 -227.82512   25    5          - -272.11773      -  11.0 1040s
 46867 28560 -221.40686   32    5          - -272.01819      -  11.0 1052s
 47058 28679 -140.89632   35    -          - -271.97895      -  11.0 1064s
 47156 28717 -228.53819   30    5          - -271.94018      -  11.0 1070s
 47288 28826 -216.71129   36    4          - -271.83912      -  11.0 1078s
 47529 28962 -238.00429   25    5          - -271.77733      -  11.0 1091s
 47740 29098 -203.88859   29    -          - -271.72245      -  11.1 1104s
 47927 29233 -227.91899   28   15          - -271.65593      -  11.1 1115s
 48213 29402 -223.75758   36    5          - -271.60647      -  11.1 1127s
 48433 29526 -240.78136   27    3          - -271.54287      -  11.1 1136s
 48615 29654 -209.67782   33    4          - -271.51760      -  11.1 1140s
 48862 29814 -237.87206   27   14          - -271.43902      -  11.1 1153s
 48925 29838 -226.92908   33    6          - -271.43381      -  11.1 1157s
 49088 29922 -215.76530   30    -          - -271.41025      -  11.1 1165s
 49298 30068 -229.78249   23   30          - -271.33092      -  11.1 1176s
 49515 30201 -189.80096   36    -          - -271.26386      -  11.1 1191s
 49689 30308 -220.21216   32    8          - -271.19212      -  11.2 1199s
 49786 30347 -231.69324   24   15          - -271.18056      -  11.2 1206s
 49903 30451 -225.27402   26    6          - -271.10257      -  11.2 1212s
 50098 30565 -233.95881   24   16          - -271.06729      -  11.2 1223s
 50322 30711 -226.93643   25    5          - -271.01672      -  11.2 1236s
 50559 30846 -250.38233   25   29          - -270.92094      -  11.2 1245s
 50714 30964 -197.81846   32    -          - -270.86543      -  11.2 1253s
 50828 31006 -206.42345   31    3          - -270.84474      -  11.2 1261s
 50946 31104 -227.97342   26    9          - -270.78924      -  11.2 1270s
 51164 31217 -198.04319   30    2          - -270.74728      -  11.2 1282s
 51215 31237 -232.42002   32    2          - -270.74313      -  11.2 1285s
 51386 31360 -223.63456   24   14          - -270.65029      -  11.2 1298s
 51610 31474 -237.26500   35    4          - -270.60965      -  11.3 1306s
 51786 31603 -259.00115   22   21          - -270.56027      -  11.3 1322s
 52012 31715 -228.41634   34    8          - -270.46879      -  11.3 1335s
 52226 31862 -230.01414   26   23          - -270.40199      -  11.3 1351s
 52375 31915 -202.30931   30    4          - -270.34420      -  11.3 1361s
 52425 31976 -200.15833   30    -          - -270.34102      -  11.3 1365s
 52654 32094 -208.95244   34    -          - -270.30370      -  11.3 1377s
 52868 32231 -224.80606   32    5          - -270.19862      -  11.3 1391s
 53036 32344 -207.34724   35    -          - -270.16781      -  11.4 1403s
 53250 32464 -238.35319   29    5          - -270.08541      -  11.4 1413s
 53452 32602 -206.95093   32   11          - -270.01612      -  11.4 1430s
 53682 32742 -198.33582   35    4          - -269.92629      -  11.4 1443s
 53881 32864 -215.13802   30    -          - -269.88737      -  11.4 1460s
 54076 33000 -226.15282   31   10          - -269.85438      -  11.4 1474s
 54299 33118 -240.36612   25   25          - -269.80872      -  11.4 1487s
 54483 33232 -226.56617   33    5          - -269.77388      -  11.4 1500s
 54682 33348 -219.01403   28   18          - -269.67939      -  11.5 1518s
 54889 33450 -213.34250   33    -          - -269.57554      -  11.5 1534s
 55068 33580 -226.77081   30    7          - -269.48075      -  11.5 1548s
 55277 33701 -211.54960   28   11          - -269.42496      -  11.5 1562s
 55476 33812 -183.51244   36    -          - -269.38532      -  11.5 1578s
 55658 33918 -215.41947   34    -          - -269.31553      -  11.5 1590s
 55853 34046 -176.21550   30    5          - -269.24128      -  11.5 1605s
 56053 34154 -225.30328   28    7          - -269.19846      -  11.6 1619s
 56095 34169 -220.10616   31   16          - -269.19828      -  11.6 1622s
 56202 34292 -213.88914   35    6          - -269.17782      -  11.6 1633s
 56245 34305 -207.63112   27    7          - -269.17695      -  11.6 1635s
 56407 34414 -223.57665   35    4          - -269.10790      -  11.6 1644s
 56469 34438 -210.08802   27    4          - -269.10745      -  11.6 1646s
 56624 34528 -210.61417   35    -          - -269.04405      -  11.6 1657s
 56845 34659 -213.06916   33    3          - -268.98912      -  11.6 1673s
 56887 34673 -227.83310   29   10          - -268.97895      -  11.6 1675s
 57041 34775 -212.23101   35    4          - -268.91577      -  11.6 1692s
 57202 34873 -203.79777   34    4          - -268.84790      -  11.6 1707s
 57377 34950 -238.58645   29    8          - -268.79347      -  11.6 1717s
 57434 35013 -204.56944   35    5          - -268.78753      -  11.6 1720s
 57640 35125 -212.77394   28    3          - -268.73023      -  11.6 1738s
 57828 35221 -215.93496   27   10          - -268.70221      -  11.7 1758s
 58027 35353 -219.39684   36    -          - -268.58551      -  11.7 1779s
 58216 35469 -209.52252   31    -          - -268.54041      -  11.7 1795s
 58293 35497 -216.51628   30    -          - -268.53179      -  11.7 1802s
 58402 35592 -231.81830   29    4          - -268.43981      -  11.7 1813s
 58619 35696 -199.87169   29    -          - -268.34767      -  11.7 1828s
 58750 35783 -233.73289   26   23          - -268.30776      -  11.7 1842s
 58843 35817 -248.97072   26   23          - -268.28549      -  11.7 1848s
 58952 35909 -216.05487   29    4          - -268.25656      -  11.7 1861s
 59169 36034 -217.16471   30    8          - -268.19829      -  11.7 1880s
 59384 36143 -192.07727   33    -          - -268.15463      -  11.7 1898s
 59418 36160 -200.79386   34    -          - -268.15293      -  11.7 1902s
 59558 36242 -213.94307   32    8          - -268.09828      -  11.7 1911s
 59650 36277 -203.72772   32    -          - -268.09725      -  11.7 1918s
 59758 36364 -236.24761   27   21          - -268.05234      -  11.7 1929s
 59847 36401 -229.16025   25   23          - -268.02879      -  11.8 1943s
 59941 36489 -231.06564   29    8          - -267.99251      -  11.8 1953s
 60046 36529 -225.43048   29   14          - -267.97387      -  11.8 1956s
 60171 36615 -192.16765   33    5          - -267.96726      -  11.8 1960s
 60377 36732 -219.58431   31    -          - -267.89026      -  11.8 1982s
 60492 36774 -225.31591   33    -          - -267.85554      -  11.8 1993s
 60525 36823 -217.91053   29    -          - -267.85082      -  11.8 2002s
 60669 36881 -214.35696   34    4          - -267.77381      -  11.8 2012s
 60722 36933 -203.41943   32    -          - -267.71496      -  11.8 2018s
 60924 37062 -209.87219   31   11          - -267.65282      -  11.8 2034s
 60998 37088 -205.37057   34    6          - -267.64756      -  11.8 2043s
 61088 37175 -232.55062   25   27          - -267.60029      -  11.8 2052s
 61188 37210 -229.23395   25   16          - -267.58838      -  11.8 2061s
 61278 37281 -205.42888   36    8          - -267.55087      -  11.8 2071s
 61443 37367 -214.69733   31    -          - -267.48559      -  11.8 2093s
 61583 37483 -240.65224   24   10          - -267.44422      -  11.8 2107s
 61794 37620 -214.23802   32   11          - -267.37527      -  11.9 2122s
 61984 37726 -199.92911   32    7          - -267.34846      -  11.9 2136s
 62195 37846 -247.54363   26   24          - -267.28721      -  11.9 2153s
 62378 37954 -218.10950   21   21          - -267.23598      -  11.9 2168s
 62427 37969 -199.32509   32   10          - -267.22730      -  11.9 2171s
 62556 38069 -197.14347   33    6          - -267.19513      -  11.9 2186s
 62737 38176 -203.63620   33    -          - -267.13385      -  11.9 2201s
 62941 38292 -220.42276   29    4          - -267.07976      -  11.9 2219s
 63104 38420 -234.03900   25   10          - -267.05877      -  11.9 2235s
 63293 38526 -215.45717   34    5          - -267.02832      -  11.9 2248s
 63340 38542 -220.04998   29    5          - -267.02764      -  11.9 2251s
 63484 38635 -227.91197   32    7          - -267.00833      -  11.9 2263s
 63619 38721 -209.41501   27    3          - -266.95993      -  11.9 2276s
 63791 38815 -196.71489   27   11          - -266.91116      -  12.0 2296s
 63989 38956 -231.28235   25   19          - -266.83206      -  12.0 2317s
 64078 38991 -189.36897   29    9          - -266.80776      -  12.0 2323s
 64172 39041 -215.55115   28    7          - -266.78859      -  12.0 2332s
 64213 39058 -191.47985   33    -          - -266.78101      -  12.0 2338s
 64331 39139 -200.86037   31    6          - -266.71258      -  12.0 2351s
 64472 39196 -193.33760   36    3          - -266.69982      -  12.0 2363s
 64528 39254 -245.75385   29   14          - -266.68202      -  12.0 2370s
 64703 39366 -209.69958   34    3          - -266.65383      -  12.0 2383s
 64903 39477 -205.33471   34    5          - -266.61024      -  12.0 2399s
 64945 39499 -227.56459   29   10          - -266.60498      -  12.0 2405s
 65072 39567 -220.71607   28   11          - -266.57046      -  12.0 2415s
 65254 39648 -229.20751   27    -          - -266.53925      -  12.0 2423s
 65303 39705 -213.63009   36    -          - -266.53463      -  12.0 2429s
 65474 39808 -220.67676   28   18          - -266.48426      -  12.0 2447s
 65564 39841 -212.13982   33    4          - -266.46728      -  12.0 2454s
 65664 39908 -230.04282   33    6          - -266.38919      -  12.0 2461s
 65772 39952 -200.46722   33    -          - -266.35335      -  12.0 2467s
 65814 40015 -209.81919   34    2          - -266.35054      -  12.0 2470s
 66013 40108 -216.74421   28    3          - -266.26135      -  12.1 2490s
 66190 40217 -210.42676   28    3          - -266.15539      -  12.1 2504s
 66233 40233 -215.59179   36    3          - -266.15531      -  12.1 2507s
 66276 40251 -179.78203   32    5          - -266.15430      -  12.1 2513s
 66354 40325 -226.69307   29    3          - -266.10203      -  12.1 2520s
 66509 40383 -198.07450   31   52          - -266.07818      -  12.1 2553s
 66512 40385 -263.43648   33   56          - -266.07818      -  12.1 2555s
 66514 40386 -245.11356   24   56          - -266.07818      -  12.1 2561s
 66525 40399 -266.07818   25   43          - -266.07818      -  12.1 2565s
 66616 40456 -225.51542   38    3          - -266.07818      -  12.2 2570s
 66648 40468 -266.07818   27   50          - -266.07818      -  12.2 2576s
 66678 40472 -198.69994   41    -          - -266.07818      -  12.2 2580s
 66745 40524 -209.02693   40    -          - -266.07818      -  12.2 2585s
 66837 40562 -256.23532   33   35          - -266.07818      -  12.2 2591s
 66906 40586 -233.24278   49    -          - -266.07818      -  12.2 2596s
 66967 40631 -224.06724   38    6          - -266.07818      -  12.2 2601s
 67063 40693 -236.28080   31   15          - -266.07818      -  12.2 2606s
 67142 40702 -266.07818   30   46          - -266.07818      -  12.2 2610s
 67275 40772 -219.01377   49    -          - -266.07818      -  12.3 2617s
 67360 40823 -233.88250   40    6          - -266.07818      -  12.3 2620s
 67454 40872 -237.15125   37    8          - -266.07818      -  12.3 2627s
 67578 40872 -246.91626   35   16          - -266.07818      -  12.3 2632s
 67607 40912 -242.43487   41    -          - -266.07818      -  12.3 2635s
 67807 41020 -247.05161   27   32          - -264.63748      -  12.3 2643s
 67862 41011 -248.10771   33   28          - -264.63748      -  12.3 2645s
 68030 41157 -231.25888   34   17          - -263.67541      -  12.3 2653s
 68089 41153 -237.68399   50    3          - -263.40760      -  12.3 2656s
 68141 41150 -242.52566   39    5          - -262.56941      -  12.3 2660s
 68266 41264 -236.31916   37   18          - -261.65442      -  12.3 2665s
 68507 41358 -213.68358   51    -          - -259.91034      -  12.3 2679s
 68743 41452 -243.16058   31   22          - -257.92292      -  12.4 2687s
 68851 41439 -224.18819   39    8          - -257.68122      -  12.4 2690s
 68988 41562 -237.40967   38   13          - -257.33278      -  12.4 2697s
 69122 41550 -250.07692   31   34          - -256.94314      -  12.4 2703s
 69207 41549 -242.76373   35   34          - -256.49209      -  12.4 2705s
 69383 41637 -245.67505   30   35          - -256.35192      -  12.4 2712s
 69466 41766 -239.30797   36    5          - -256.28611      -  12.4 2715s
 69767 41902 -242.45223   39    8          - -255.83223      -  12.4 2726s
 69853 41896 -207.42206   44    4          - -255.69530      -  12.4 2730s
 70141 42051 -236.88102   42    -          - -255.09115      -  12.4 2743s
 70426 42035 -218.12774   43    -          - -254.75902      -  12.4 2751s
 70629 42201 -215.88258   47    7          - -254.29164      -  12.4 2757s
 70924 42347 -246.03702   36   31          - -253.61791      -  12.5 2768s
 71130 42335 -229.80389   51    -          - -253.54714      -  12.5 2776s
 71326 42525 -227.75105   36    -          - -253.23401      -  12.5 2784s
 71708 42633 -223.68313   43    3          - -252.65399      -  12.5 2803s
 71794 42630 -201.50672   49    3          - -252.62045      -  12.5 2808s
 72010 42824 -220.09247   41    -          - -252.02350      -  12.5 2816s
 72469 43004 -238.96844   39    9          - -251.24522      -  12.5 2844s
 72954 43195 -233.32563   30   23          - -250.96862      -  12.5 2865s
 73390 43385 -237.58628   41   10          - -250.65444      -  12.5 2881s
 73744 43357 -210.73496   41    5          - -250.17801      -  12.5 2893s
 73867 43520 -226.62227   51    5          - -250.16748      -  12.5 2903s
 74046 43510 -216.67832   41    4          - -250.07805      -  12.6 2915s
 74259 43766 -230.94147   35    7          - -249.82816      -  12.6 2923s
 74841 43997 -237.34477   30   28          - -249.19663      -  12.6 2949s
 75435 44256 -232.56969   43   17          - -248.63560      -  12.6 2975s
 76086 44434 -235.19081   41    -          - -248.17468      -  12.6 3007s
 76581 44702 -234.81824   38   18          - -247.77612      -  12.6 3033s
 76775 44686 -233.86898   38   10          - -247.77444      -  12.6 3042s
 77291 44941 -229.84539   42    -          - -247.27323      -  12.6 3059s
 77957 45163 -229.29985   38   15          - -246.91694      -  12.6 3082s
 78075 45152 -220.98832   43    3          - -246.87388      -  12.6 3088s
 78551 45300 -225.07961   42    9          - -246.70237      -  12.6 3112s
 78697 45307 -237.07628   34   21          - -246.66128      -  12.6 3117s
 79076 45536 -231.46812   42   11          - -246.39625      -  12.6 3129s
 79742 45779 -229.88634   38   13          - -245.98788      -  12.6 3160s
 80332 45961 -229.39558   39    7          - -245.78021      -  12.6 3184s
 80906 46121 -233.92838   39    8          - -245.60048      -  12.6 3211s
 81016 46122 -237.25998   41   14          - -245.57398      -  12.6 3217s
 81354 46296 -230.18158   37    3          - -245.40987      -  12.6 3232s
 81631 46289 -228.76808   42    2          - -245.36156      -  12.6 3247s
 81951 46516 -210.67720   43    -          - -245.13978      -  12.6 3261s
 82454 46707 -225.55780   49    -          - -244.90665      -  12.6 3290s
 82875 46695 -238.09810   39   20          - -244.72968      -  12.6 3313s
 83016 46893 -218.35571   42    -          - -244.71864      -  12.6 3318s
 83543 46984 -214.68280   44    2          - -244.51784      -  12.7 3352s
 83940 47219 -226.01927   35   11          - -244.40718      -  12.7 3374s
 84500 47404 -221.32821   41    -          - -244.15439      -  12.7 3407s
 85002 47584 -214.62100   43    2          - -244.01205      -  12.7 3433s
 85281 47579 -181.94835   47    -          - -243.99718      -  12.7 3447s
 85541 47747 -235.23214   46    3          - -243.80793      -  12.7 3460s
 85973 47756 -231.77440   34   20          - -243.66926      -  12.7 3479s
 86111 47879 -216.94750   48    -          - -243.60251      -  12.7 3485s
 86480 48054 -226.12645   35   25          - -243.54224      -  12.7 3506s
 87029 48250 -214.28265   50    -          - -243.34689      -  12.7 3534s
 87561 48406 -227.07445   36    3          - -243.18226      -  12.7 3570s
 88078 48574 -221.41820   42   11          - -243.01552      -  12.7 3602s
 88197 48574 -219.01151   48    -          - -243.01062      -  12.7 3612s
 88578 48706 -219.34523   41    5          - -242.94425      -  12.7 3630s
 88958 48833 -229.11077   38   16          - -242.83094      -  12.7 3648s
 89067 48835 -204.74319   47    -          - -242.83004      -  12.7 3657s
 89202 48830 -220.49361   52    3          - -242.82441      -  12.7 3665s
 89467 49020 -227.44527   38    7          - -242.72668      -  12.7 3680s
 89960 49161 -217.37031   35   21          - -242.47216      -  12.7 3710s
 90081 49161 -219.06479   36    9          - -242.46872      -  12.7 3718s
 90458 49306 -227.90818   37   15          - -242.31791      -  12.7 3742s
 90706 49317 -211.05669   54    3          - -242.29880      -  12.7 3757s
 90829 49317 -224.01106   35    6          - -242.19360      -  12.7 3765s
 90955 49431 -227.28758   47    5          - -242.19252      -  12.7 3774s
 91160 49441 -217.05862   44    -          - -242.13448      -  12.7 3783s
 91324 49607 -232.92026   39   12          - -242.06365      -  12.7 3792s
 91797 49754 -215.51572   47    -          - -241.90444      -  12.7 3822s
 92285 49931 -233.83399   31   31          - -241.75960      -  12.7 3850s
 92715 50012 -225.87424   33   31          - -241.66743      -  12.7 3881s
 93052 50196 -228.77965   35   22          - -241.60332      -  12.7 3900s
 93166 50193 -225.64935   38   11          - -241.59973      -  12.7 3907s
 93404 50187 -212.82836   45    -          - -241.50692      -  12.7 3922s
 93530 50305 -217.61790   46    5          - -241.49008      -  12.7 3928s
 94023 50500 -210.64496   45    -          - -241.31360      -  12.7 3968s
 94397 50490 -209.53395   44    -          - -241.14717      -  12.7 3996s
 94524 50661 -222.42159   46    3          - -241.13619      -  12.7 4006s
 94636 50655 -215.39074   50    -          - -241.12690      -  12.7 4019s
 95003 50767 -221.75894   44    7          - -241.03033      -  12.7 4048s
 95188 50768 -217.55262   43    5          - -241.00992      -  12.7 4058s
 95353 50896 -222.12636   47    -          - -240.93163      -  12.7 4071s
 95788 51066 -222.27685   37   18          - -240.84313      -  12.7 4099s
 95898 51061 -227.25716   33   25          - -240.83949      -  12.7 4106s
 96214 51204 -222.97345   41   11          - -240.76809      -  12.7 4128s
 96554 51205 -217.77554   42   13          - -240.70375      -  12.7 4150s
 96665 51380 -211.99912   38    9          - -240.68080      -  12.7 4157s
 96896 51377 -219.06289   37   21          - -240.67002      -  12.7 4181s
 97117 51464 -216.23475   44    2          - -240.59260      -  12.7 4189s
 97411 51585 -202.85319   44    3          - -240.50097      -  12.7 4210s
 97544 51592 -209.64522   45    2          - -240.49802      -  12.7 4218s
 97900 51739 -213.57878   49    -          - -240.41687      -  12.7 4244s
 98347 51869 -228.76333   37   16          - -240.29210      -  12.7 4276s
 98571 51875 -224.23555   38    6          - -240.29126      -  12.7 4294s
 98765 51983 -214.95268   46    2          - -240.23151      -  12.7 4309s
 98984 51997 -226.81084   38   25          - -240.21961      -  12.7 4328s
 99206 52116 -196.60746   46    -          - -240.15613      -  12.7 4349s
 99287 52108 -202.21871   42    6          - -240.14073      -  12.7 4354s
 99554 52267 -208.78495   45    4          - -240.06462      -  12.7 4372s
 99985 52408 -229.47722   40   15          - -239.97052      -  12.7 4401s
 100425 52532 -236.29792   37   14          - -239.89359      -  12.7 4429s
 100519 52535 -207.74128   43    6          - -239.89255      -  12.7 4435s
 100836 52668 -235.16523   38   27          - -239.81528      -  12.7 4454s
 100947 52664 -228.85160   35   26          - -239.81311      -  12.7 4464s
 101289 52788 -232.90654   35   35          - -239.72877      -  12.7 4492s
 101361 52787 -214.10603   43   15          - -239.70756      -  12.7 4498s
 101510 52779 -214.86790   48    -          - -239.66288      -  12.7 4509s
 101580 52870 -226.98352   35   27          - -239.65411      -  12.7 4517s
 102026 53018 -204.96261   36    5          - -239.58515      -  12.7 4548s
 102239 53024 -210.78535   43    7          - -239.57872      -  12.7 4568s
 102346 53028 -217.04531   45    9          - -239.49552      -  12.7 4578s
 102456 53139 -201.38934   41    8          - -239.48396      -  12.7 4587s
 102848 53247 -228.65746   38   24          - -239.40054      -  12.7 4620s
 103189 53261 -209.32119   46    -          - -239.30473      -  12.7 4644s
 103289 53381 -231.40414   41   12          - -239.30429      -  12.7 4650s
 103377 53383 -222.93277   39   18          - -239.30245      -  12.8 4659s
 103608 53498 -217.68375   36    7          - -239.26448      -  12.8 4683s
 103707 53496 -211.64876   36    -          - -239.26044      -  12.8 4691s
 103820 53495 -197.51407   46    -          - -239.25596      -  12.8 4701s
 104040 53631 -211.03643   40    9          - -239.22124      -  12.8 4718s
 104263 53636 -220.50067   37   16          - -239.20071      -  12.8 4732s
 104470 53767 -208.14576   47    4          - -239.13371      -  12.8 4754s
 104789 53766 -220.89733   46    -          - -239.04169      -  12.8 4790s
 104901 53879 -219.40792   41    -          - -239.03581      -  12.8 4801s
 105258 53972 -206.14467   42    9          - -238.96836      -  12.8 4830s
 105576 54100 -214.04536   45    8          - -238.87060      -  12.8 4863s
 105669 54102 -209.16588   50    -          - -238.86953      -  12.8 4873s
 105988 54244 -208.63622   48    5          - -238.76550      -  12.8 4895s
 106083 54247 -219.11519   43    2          - -238.75678      -  12.8 4901s
 106381 54354 -218.49889   41   17          - -238.67096      -  12.8 4928s
 106761 54477 -220.41210   39   11          - -238.60813      -  12.8 4962s
 106968 54481 -212.14476   42    7          - -238.60709      -  12.8 4982s
 107171 54566 -217.51855   46    6          - -238.55889      -  12.8 4998s
 107314 54567 -223.41361   47    8          - -238.54956      -  12.8 5011s
 107468 54678 -217.76617   41   19          - -238.47759      -  12.8 5027s
 107560 54679 -213.61372   44    -          - -238.47468      -  12.8 5033s
 107840 54809 -214.75262   36   18          - -238.43053      -  12.8 5064s
 108051 54812 -226.82214   41   15          - -238.42378      -  12.8 5080s
 108258 54972 -211.09158   47    4          - -238.36670      -  12.8 5100s
 108675 55065 -202.61706   43    -          - -238.32353      -  12.8 5139s
 109025 55161 -215.89899   40    5          - -238.26932      -  12.8 5172s
 109112 55162 -223.37233   40    4          - -238.25234      -  12.8 5179s
 109339 55287 -206.23807   47    2          - -238.23657      -  12.8 5196s
 109715 55398 -217.66628   41    4          - -238.17221      -  12.8 5228s
 110105 55518 -207.81144   35   10          - -238.09635      -  12.8 5268s
 110206 55520 -215.60909   53    -          - -238.09367      -  12.8 5280s
 110511 55662 -194.62529   50    -          - -238.02742      -  12.8 5304s
 110604 55663 -216.84325   39   21          - -238.01107      -  12.8 5311s
 110904 55750 -197.88808   47    -          - -237.94415      -  12.8 5341s
 111223 55872 -218.15944   37   11          - -237.92360      -  12.8 5364s
 111414 55869 -208.73195   48    3          - -237.91229      -  12.8 5375s
 111586 55984 -217.80385   41    5          - -237.88057      -  12.8 5391s
 111942 56108 -203.80993   38   10          - -237.82223      -  12.8 5425s
 112032 56106 -221.80258   43   12          - -237.81240      -  12.8 5434s
 112298 56247 -229.39425   37   10          - -237.77510      -  12.8 5461s
 112586 56245 -218.39377   42    6          - -237.72068      -  12.8 5497s
 112678 56318 -229.31632   44   15          - -237.71625      -  12.8 5505s
 112799 56315 -223.87502   36   21          - -237.71241      -  12.8 5519s
 112960 56436 -202.40150   46    -          - -237.68195      -  12.8 5538s
 113330 56561 -216.01657   36   11          - -237.64623      -  12.8 5566s
 113430 56557 -192.59566   45    -          - -237.64280      -  12.8 5578s
 113690 56675 -200.43117   50    -          - -237.59913      -  12.8 5601s
 113952 56675 -209.87776   45    4          - -237.54117      -  12.8 5628s
 114047 56788 -220.46203   48    -          - -237.53841      -  12.8 5633s
 114409 56894 -212.87380   50    -          - -237.47233      -  12.8 5668s
 114476 56894 -221.72027   42   12          - -237.46402      -  12.8 5673s
 114672 57016 -185.37925   43    -          - -237.43559      -  12.8 5690s
 115014 57141 -217.05029   45    4          - -237.40510      -  12.8 5721s
 115388 57238 -233.16749   33   33          - -237.35943      -  12.8 5754s
 115469 57236 -204.35921   44    -          - -237.35705      -  12.8 5763s
 115753 57341 -210.14244   48    -          - -237.32338      -  12.8 5789s
 115918 57347 -203.25424   48    6          - -237.30570      -  12.8 5802s
 116086 57438 -225.43220   45    2          - -237.26310      -  12.8 5818s
 116238 57441 -222.21836   40   18          - -237.25484      -  12.8 5833s
 116346 57545 -214.06253   38    8          - -237.22312      -  12.8 5847s
 116588 57538 -218.33408   41   21          - -237.19654      -  12.8 5863s
 116693 57655 -230.03887   36   21          - -237.17805      -  12.8 5874s
 116782 57655 -220.83925   40   17          - -237.15404      -  12.8 5883s
 116880 57654 -214.77988   37    9          - -237.15303      -  12.8 5889s
 117059 57804 -219.63307   51    5          - -237.09488      -  12.8 5907s
 117410 57897 -220.02820   47    -          - -237.05224      -  12.9 5936s
 117666 57900 -222.59623   38    4          - -236.99798      -  12.9 5964s
 117749 57978 -197.02285   42    -          - -236.99486      -  12.9 5975s
 117881 57980 -209.19061   42    5          - -236.99048      -  12.9 5989s
 118034 58079 -202.10242   38   13          - -236.97670      -  12.9 6006s
 118420 58212 -229.69732   40    8          - -236.92022      -  12.9 6046s
 118514 58211 -213.78027   47    -          - -236.91703      -  12.9 6055s
 118783 58301 -194.71706   49    -          - -236.87393      -  12.9 6095s
 119117 58369 -216.04371   49    2          - -236.81303      -  12.9 6128s
 119374 58457 -220.92300   31   27          - -236.78431      -  12.9 6162s
 119733 58571 -223.29555   35   31          - -236.72719      -  12.9 6207s
 119817 58573 -197.38006   42    8          - -236.71993      -  12.9 6225s
 120057 58661 -206.75074   50    3          - -236.67877      -  12.9 6249s
 120143 58667 -225.93880   40   13          - -236.66849      -  12.9 6263s
 120320 58672 -197.85449   46    -          - -236.63766      -  12.9 6280s
 120404 58789 -204.27616   43    5          - -236.63060      -  12.9 6289s
 120485 58789 -221.69341   47    -          - -236.63017      -  12.9 6298s
 120751 58837 -200.05359   49    -          - -236.60981      -  12.9 6323s
 121033 58953 -208.42002   41    -          - -236.57431      -  12.9 6363s
 121300 58963 -205.55426   44    4          - -236.54543      -  12.9 6384s
 121389 59070 -197.81654   38    -          - -236.54183      -  12.9 6397s
 121473 59070 -210.87420   46    -          - -236.53771      -  12.9 6405s
 121770 59197 -208.17102   46    -          - -236.50901      -  12.9 6441s
 122107 59311 -226.12374   39   12          - -236.45640      -  12.9 6474s
 122269 59307 -219.50187   39   11          - -236.44957      -  12.9 6484s
 122439 59370 -221.08350   46    7          - -236.42361      -  12.9 6502s
 122589 59372 -203.60797   46    3          - -236.42018      -  12.9 6517s
 122726 59481 -212.12553   42    8          - -236.39789      -  12.9 6535s
 122993 59493 -200.81689   42    8          - -236.36277      -  12.9 6565s
 123078 59568 -217.51621   34   13          - -236.35826      -  12.9 6574s
 123159 59567 -217.14224   51    -          - -236.35582      -  12.9 6586s
 123431 59668 -206.80154   41    6          - -236.31171      -  12.9 6623s
 123677 59678 -198.18151   41   10          - -236.28169      -  12.9 6649s
 123772 59758 -211.11995   49    -          - -236.27730      -  12.9 6664s
 124041 59873 -214.69409   46    4          - -236.24695      -  12.9 6696s
 124132 59868 -215.28944   50    4          - -236.24471      -  12.9 6704s
 124395 59965 -213.07276   38   11          - -236.20891      -  12.9 6729s
 124488 59972 -212.01964   38   14          - -236.20827      -  12.9 6740s
 124725 60095 -225.73118   41    5          - -236.17270      -  12.9 6766s
 125109 60171 -216.64355   48    9          - -236.12911      -  12.9 6817s
 125183 60170 -213.51945   52    5          - -236.12833      -  12.9 6823s
 125360 60283 -214.25160   44   11          - -236.09836      -  12.9 6847s
 125432 60283 -210.33643   38   18          - -236.09366      -  12.9 6853s
 125517 60285 -219.24692   43   10          - -236.09055      -  12.9 6861s
 125696 60377 -207.40849   47    9          - -236.07567      -  12.9 6887s
 125883 60385 -209.33328   50    5          - -236.06648      -  12.9 6907s
 126063 60502 -201.56948   42    2          - -236.02429      -  12.9 6931s
 126155 60505 -228.38570   36   26          - -236.01398      -  12.9 6940s
 126438 60604 -202.52824   36   10          - -235.99127      -  12.9 6969s
 126531 60598 -218.86662   39   18          - -235.98676      -  12.9 6973s
 126608 60603 -207.82136   44    4          - -235.98441      -  12.9 6980s
 126798 60668 -198.57193   48    -          - -235.96480      -  12.9 7005s
 126942 60675 -217.45620   48    2          - -235.96179      -  12.9 7020s
 127067 60786 -219.73494   45    -          - -235.93040      -  12.9 7044s
 127137 60787 -213.13872   42    6          - -235.92941      -  12.9 7053s
 127374 60871 -203.22094   40    5          - -235.90939      -  12.9 7075s
 127717 60997 -192.81045   45    -          - -235.86626      -  13.0 7126s
 127978 60995 -219.41757   42   13          - -235.82273      -  13.0 7156s
 128082 61064 -214.06669   44    -          - -235.81830      -  13.0 7165s
 128149 61064 -212.14604   48    -          - -235.81697      -  13.0 7172s
 128371 61171 -218.16797   38   19          - -235.78839      -  13.0 7204s
 128762 61273 -222.44574   38   11          - -235.75956      -  12.9 7239s
 128860 61273 -214.58392   41   13          - -235.75914      -  12.9 7250s
 129033 61279 -202.66627   51    -          - -235.74495      -  12.9 7275s
 129116 61392 -204.08790   47    -          - -235.73555      -  13.0 7289s
 129437 61494 -189.04361   41    5          - -235.69361      -  13.0 7321s
 129705 61502 -215.09138   41   11          - -235.66652      -  13.0 7356s
 129790 61585 -188.74618   46    5          - -235.66054      -  13.0 7367s
 129869 61590 -185.51186   45    -          - -235.65837      -  13.0 7375s
 130055 61651 -202.32946   48    -          - -235.64363      -  13.0 7400s
 130133 61655 -203.24253   44    -          - -235.63959      -  13.0 7409s
 130379 61746 -213.41986   36   24          - -235.61128      -  13.0 7437s
 130559 61751 -199.08392   47    2          - -235.60455      -  13.0 7454s
 130734 61836 -208.76783   45    9          - -235.57364      -  13.0 7484s
 130825 61840 -215.69182   43    2          - -235.57104      -  13.0 7496s
 131067 61917 -219.14450   41   13          - -235.53780      -  13.0 7527s
 131137 61922 -210.02568   41    7          - -235.53199      -  13.0 7533s
 131246 61918 -199.08373   39    6          - -235.52556      -  13.0 7546s
 131303 62021 -201.33660   51    -          - -235.51717      -  13.0 7563s
 131559 62023 -192.23305   50    -          - -235.46770      -  13.0 7586s
 131643 62153 -207.21618   40   12          - -235.46647      -  13.0 7599s
 131926 62146 -215.98155   39   11          - -235.42869      -  13.0 7634s
 132011 62236 -201.09325   42    2          - -235.42735      -  13.0 7648s
 132330 62332 -222.14483   35   17          - -235.41585      -  13.0 7671s
 132417 62339 -205.22142   51    -          - -235.41014      -  13.0 7686s
 132659 62419 -195.89552   43    6          - -235.37358      -  13.0 7729s
 132839 62424 -221.32423   38   13          - -235.34928      -  13.0 7757s
 132910 62512 -223.61943   38   16          - -235.34720      -  13.0 7764s
 132996 62510 -217.93683   38   10          - -235.34104      -  13.0 7772s
 133173 62523 -222.00710   32   34          - -235.31261      -  13.0 7801s
 133259 62619 -199.35166   45    -          - -235.30872      -  13.0 7814s
 133610 62686 -191.91573   41    -          - -235.27007      -  13.0 7859s
 133781 62702 -208.80672   37   17          - -235.25559      -  13.0 7874s
 133951 62788 -215.55286   39   12          - -235.23281      -  13.0 7886s
 134018 62789 -216.71878   39    8          - -235.23258      -  13.0 7894s
 134216 62882 -212.30770   51    2          - -235.21535      -  13.0 7920s
 134574 62978 -190.35515   45    -          - -235.19112      -  13.0 7977s
 134738 62981 -206.23482   46    -          - -235.18347      -  13.0 7987s
 134919 63078 -194.22913   44    2          - -235.16777      -  13.0 8001s
 135084 63084 -207.66497   41    7          - -235.16299      -  13.0 8022s
 135259 63196 -189.86482   44    -          - -235.12600      -  13.0 8037s
 135581 63265 -211.57092   47    5          - -235.10055      -  13.0 8088s
 135704 63272 -187.47006   43    9          - -235.08851      -  13.0 8102s
 135827 63374 -202.59762   47    3          - -235.06907      -  13.0 8119s
 136073 63378 -204.58351   38   16          - -235.04525      -  13.0 8145s
 136173 63466 -189.01230   47    2          - -235.04485      -  13.0 8155s
 136413 63474 -219.13381   44    5          - -235.01484      -  13.0 8179s
 136508 63542 -203.23627   49    -          - -235.01434      -  13.0 8201s
 136581 63545 -221.13086   37   21          - -235.01322      -  13.0 8210s
 136674 63546 -216.02401   48    -          - -235.01199      -  13.0 8222s
 136864 63675 -221.72607   41    2          - -234.96820      -  13.0 8247s
 136949 63678 -220.58993   42   20          - -234.96640      -  13.0 8258s
 137202 63733 -212.62550   46    -          - -234.93476      -  13.0 8292s
 137269 63736 -197.98438   37   11          - -234.93195      -  13.0 8300s
 137461 63820 -215.70092   50    -          - -234.90062      -  13.0 8331s
 137817 63917 -224.69081   33   23          - -234.86650      -  13.0 8372s
 137906 63918 -218.40527   43   15          - -234.86356      -  13.0 8386s
 138190 64017 -208.46136   43    5          - -234.81344      -  13.0 8425s
 138256 64019 -212.16424   44    -          - -234.80732      -  13.0 8439s
 138491 64086 -209.36912   45   11          - -234.78489      -  13.0 8470s
 138754 64164 -198.10168   44    -          - -234.74601      -  13.0 8523s
 138843 64167 -216.38571   40    7          - -234.74348      -  13.0 8540s
 139071 64272 -177.53526   47    -          - -234.70600      -  13.0 8580s
 139233 64274 -226.19039   33   24          - -234.69935      -  13.0 8599s
 139385 64383 -224.10687   47    4          - -234.66400      -  13.1 8624s
 139536 64385 -219.02025   40   22          - -234.65821      -  13.1 8646s
 139699 64451 -204.24826   45    6          - -234.60373      -  13.1 8664s
 139787 64460 -199.11120   45    5          - -234.60363      -  13.1 8680s
 140058 64552 -199.07368   43    6          - -234.55715      -  13.1 8723s
 140300 64649 -212.15697   47    5          - -234.53213      -  13.1 8759s
 140372 64652 -201.90694   46    -          - -234.52849      -  13.1 8776s
 140526 64650 -196.86733   44    -          - -234.50113      -  13.1 8796s
 140610 64722 -183.47865   45    -          - -234.50002      -  13.1 8816s
 140848 64736 -201.38556   47    2          - -234.45290      -  13.1 8854s
 140924 64820 -216.02475   38   11          - -234.45119      -  13.1 8864s
 141013 64819 -204.48036   43    2          - -234.44927      -  13.1 8874s
 141270 64890 -199.81467   48    -          - -234.42716      -  13.1 8917s
 141400 64898 -221.81619   41   13          - -234.41791      -  13.1 8937s
 141518 64990 -214.47379   47    -          - -234.39940      -  13.1 8960s
 141671 64994 -215.29119   39    4          - -234.39181      -  13.1 8978s
 141851 65075 -199.94873   44    3          - -234.36594      -  13.1 9004s
 141931 65079 -210.07124   46    2          - -234.36286      -  13.1 9024s
 142176 65166 -217.95692   35   23          - -234.33055      -  13.1 9063s
 142523 65288 -226.92344   40   20          - -234.30177      -  13.1 9126s
 142598 65284 -204.30683   44    6          - -234.29838      -  13.1 9134s
 142818 65354 -213.35111   41    2          - -234.27642      -  13.1 9165s
 143049 65458 -205.09243   44    9          - -234.25079      -  13.1 9202s
 143390 65542 -217.41412   38   16          - -234.21858      -  13.1 9258s
 143696 65627 -204.36193   42   17          - -234.20646      -  13.1 9300s
 144052 65720 -210.69957   37   28          - -234.17987      -  13.1 9346s
 144111 65725 -213.55970   38   10          - -234.17953      -  13.1 9354s
 144294 65803 -193.84461   43    -          - -234.16534      -  13.1 9384s
 144594 65876 -204.57379   51    -          - -234.14078      -  13.1 9420s
 144904 65985 -213.82445   36   16          - -234.10404      -  13.1 9470s
 145079 65994 -209.90074   45    -          - -234.10112      -  13.1 9505s
 145251 66068 -200.01060   49    -          - -234.07854      -  13.1 9522s
 145508 66135 -217.24840   43    -          - -234.06304      -  13.1 9560s
 145853 66237 -208.91898   47    -          - -234.04389      -  13.1 9601s
 146163 66342 -210.59202   47    2          - -234.00765      -  13.1 9648s
 146320 66344 -197.69211   45    -          - -234.00019      -  13.2 9674s
 146467 66422 -188.14730   43    -          - -233.97722      -  13.2 9697s
 146539 66426 -206.88676   45    6          - -233.97603      -  13.2 9706s
 146718 66433 -203.81655   46    6          - -233.95372      -  13.2 9722s
 146810 66490 -194.52732   45    -          - -233.94930      -  13.2 9740s
 146945 66495 -188.52900   48    -          - -233.94829      -  13.2 9758s
 147062 66576 -202.27043   44    -          - -233.93056      -  13.2 9785s
 147143 66579 -209.06645   47    -          - -233.92685      -  13.2 9797s
 147393 66671 -189.98192   43    -          - -233.90886      -  13.2 9841s
 147576 66684 -212.68103   49    2          - -233.90186      -  13.2 9870s
 147743 66797 -216.00938   36   22          - -233.87687      -  13.2 9891s
 148079 66844 -202.17479   38    5          - -233.84839      -  13.2 9942s
 148164 66856 -195.21662   47    2          - -233.84684      -  13.2 9950s
 148386 66928 -207.17311   42    6          - -233.82940      -  13.2 9981s
 148493 66930 -208.81739   35   16          - -233.81828      -  13.2 9996s
 148534 66930 -194.54573   41    -          - -233.80811      -  13.2 10003s
 148597 67010 -219.88576   40    -          - -233.80134      -  13.2 10019s
 148942 67105 -228.84076   35   26          - -233.77588      -  13.2 10059s
 149022 67104 -221.31198   38   11          - -233.77018      -  13.2 10067s
 149273 67205 -201.68687   46    3          - -233.74180      -  13.2 10110s
 149505 67211 -201.89884   42    4          - -233.72605      -  13.2 10137s
 149590 67301 -209.93019   38   16          - -233.72050      -  13.2 10153s
 149905 67395 -204.73784   47    3          - -233.67202      -  13.2 10206s
 150157 67404 -214.71154   53    -          - -233.62740      -  13.2 10247s
 150245 67471 -217.77298   37   11          - -233.62476      -  13.2 10260s
 150301 67472 -207.27605   42    9          - -233.62406      -  13.2 10272s
 150492 67553 -194.17522   49    -          - -233.60804      -  13.2 10313s
 150854 67651 -225.08020   37   25          - -233.58410      -  13.2 10368s
 150918 67649 -206.63085   39    9          - -233.58080      -  13.2 10372s
 151152 67717 -214.51603   42    -          - -233.57326      -  13.2 10410s
 151227 67726 -202.55577   48    5          - -233.57049      -  13.2 10419s
 151449 67810 -206.51354   49    -          - -233.53269      -  13.2 10463s
 151683 67905 -222.76527   37   23          - -233.50982      -  13.2 10487s
 151772 67903 -204.13104   47    -          - -233.50731      -  13.2 10495s
 152018 68002 -183.08767   47    -          - -233.49815      -  13.2 10531s
 152319 68071 -197.06596   45    -          - -233.48698      -  13.2 10568s
 152635 68168 -191.86772   40    4          - -233.44673      -  13.2 10630s
 152718 68174 -197.69676   44    4          - -233.44669      -  13.2 10643s
 152983 68285 -210.08561   45    5          - -233.42313      -  13.2 10684s
 153079 68285 -221.50588   37    5          - -233.42258      -  13.2 10701s
 153341 68339 -192.41945   45    8          - -233.40087      -  13.2 10745s
 153568 68472 -213.71213   44    6          - -233.38268      -  13.2 10799s
 153641 68469 -210.08659   36   17          - -233.38215      -  13.2 10812s
 153891 68534 -202.04096   43   10          - -233.36243      -  13.2 10842s
 153962 68533 -179.05972   45    -          - -233.35993      -  13.2 10852s
 154199 68637 -215.11669   45    3          - -233.34776      -  13.2 10892s
 154542 68732 -204.98358   40   10          - -233.32781      -  13.2 10952s
 154846 68801 -190.60972   51    -          - -233.30340      -  13.2 11006s
 155089 68881 -213.36226   46    2          - -233.28584      -  13.2 11053s
 155334 68895 -206.54569   44    -          - -233.25733      -  13.2 11097s
 155414 69002 -193.56750   39    5          - -233.24863      -  13.2 11112s
 155729 69068 -214.89630   39   21          - -233.22795      -  13.2 11161s
 155820 69077 -212.36824   33   16          - -233.22557      -  13.2 11165s
 155909 69081 -222.52320   36   15          - -233.22187      -  13.2 11170s
 156057 69142 -204.38619   48    3          - -233.21664      -  13.2 11189s
 156304 69223 -201.23628   45    5          - -233.18360      -  13.2 11235s
 156602 69319 -213.67883   43    2          - -233.12975      -  13.2 11302s
 156768 69320 -225.10929   44   12          - -233.12917      -  13.2 11327s
 156929 69415 -215.28233   34   16          - -233.11051      -  13.2 11367s
 157004 69417 -199.97291   39   10          - -233.10815      -  13.2 11376s
 157093 69420 -217.51191   39   12          - -233.10583      -  13.2 11384s
 157260 69520 -215.52201   41   20          - -233.08254      -  13.2 11422s
 157319 69520 -220.28366   45   10          - -233.08125      -  13.2 11434s
 157496 69570 -186.94902   46    6          - -233.06993      -  13.2 11463s
 157586 69577 -215.85358   47    -          - -233.06772      -  13.2 11474s
 157832 69677 -206.51894   46    8          - -233.05617      -  13.2 11511s
 157914 69686 -193.68174   50    -          - -233.05282      -  13.2 11520s
 157984 69687 -201.34116   49    -          - -233.05208      -  13.3 11536s
 158151 69769 -205.11525   49    8          - -233.03813      -  13.3 11560s
 158223 69768 -194.74094   41    3          - -233.03808      -  13.3 11575s
 158300 69777 -220.95604   43    3          - -233.03569      -  13.3 11591s
 158447 69846 -205.82731   44    5          - -233.01111      -  13.3 11633s
 158517 69846 -215.62409   39   17          - -233.00950      -  13.3 11649s
 158714 69950 -198.05088   42   11          - -232.99957      -  13.3 11685s
 159039 70013 -200.96252   48    -          - -232.97459      -  13.3 11753s
 159120 70017 -189.21687   48    -          - -232.97326      -  13.3 11762s
 159354 70117 -228.44878   42   19          - -232.95514      -  13.3 11812s
 159420 70114 -208.33049   43   13          - -232.95346      -  13.3 11817s
 159667 70177 -197.39884   47    6          - -232.94175      -  13.3 11855s
 159891 70249 -213.34132   39    6          - -232.92459      -  13.3 11897s
 159973 70253 -202.61032   45    2          - -232.92114      -  13.3 11906s
 160217 70350 -179.82870   45    -          - -232.89791      -  13.3 11949s
 160366 70355 -208.77065   46    -          - -232.89267      -  13.3 11979s
 160519 70433 -218.98946   36   23          - -232.87105      -  13.3 12008s
 160615 70437 -200.10984   43    5          - -232.86586      -  13.3 12021s
 160687 70438 -215.22748   48    -          - -232.86332      -  13.3 12034s
 160850 70531 -193.84925   45    -          - -232.83551      -  13.3 12071s
 161080 70540 -214.85739   43    8          - -232.80896      -  13.3 12106s
 161170 70607 -211.06973   45    5          - -232.80706      -  13.3 12116s
 161232 70611 -202.27073   46    2          - -232.80621      -  13.3 12126s
 161410 70713 -188.52029   49    -          - -232.79076      -  13.3 12166s
 161713 70799 -187.64939   37   11          - -232.76710      -  13.3 12221s
 161858 70802 -190.40284   42    -          - -232.76570      -  13.3 12251s
 162005 70886 -189.19424   42    5          - -232.74643      -  13.3 12286s
 162087 70885 -195.47690   44   10          - -232.74261      -  13.3 12298s
 162313 70950 -200.43067   42    3          - -232.70993      -  13.3 12332s
 162530 71068 -216.92076   38   18          - -232.69440      -  13.3 12359s
 162605 71064 -188.07285   47    8          - -232.69034      -  13.3 12369s
 162860 71177 -210.74371   37   14          - -232.66754      -  13.3 12406s
 162941 71175 -204.29775   41   11          - -232.66440      -  13.3 12421s
 163189 71256 -204.28245   45    -          - -232.65857      -  13.3 12462s
 163494 71322 -206.79327   48    2          - -232.64316      -  13.3 12511s
 163806 71432 -218.39835   38   17          - -232.60175      -  13.3 12565s
 164125 71515 -208.15383   40    8          - -232.58063      -  13.3 12626s
 164193 71514 -202.60475   51    -          - -232.58041      -  13.3 12640s
 164378 71609 -214.94015   42   20          - -232.56316      -  13.3 12674s
 164692 71688 -213.30491   32   27          - -232.53861      -  13.3 12725s
 164762 71690 -204.68489   46    4          - -232.53516      -  13.3 12735s
 165001 71787 -204.35954   47    -          - -232.53017      -  13.3 12769s
 165082 71792 -197.10028   41    7          - -232.52550      -  13.3 12774s
 165306 71883 -186.18704   44    3          - -232.49859      -  13.3 12806s
 165362 71888 -215.53265   40   11          - -232.49777      -  13.3 12815s
 165564 71949 -215.04532   39   18          - -232.47767      -  13.3 12845s
 165727 71959 -221.50505   41   15          - -232.47283      -  13.3 12870s
 165819 71966 -201.44734   44    5          - -232.46662      -  13.3 12883s
 165896 72059 -204.54990   48    5          - -232.45903      -  13.3 12897s
 166216 72149 -189.99164   43    -          - -232.42196      -  13.3 12970s
 166298 72153 -221.12923   37   16          - -232.42120      -  13.3 12983s
 166531 72207 -203.12801   39   15          - -232.39011      -  13.3 13034s
 166643 72214 -193.68692   44    -          - -232.38659      -  13.3 13055s
 166705 72215 -193.04047   47    -          - -232.37489      -  13.3 13074s
 166762 72327 -200.78991   49    -          - -232.37034      -  13.3 13101s
 167081 72401 -199.83822   46    6          - -232.33841      -  13.3 13155s
 167150 72404 -213.93815   35   22          - -232.33216      -  13.3 13172s
 167358 72496 -202.92911   46    4          - -232.30800      -  13.3 13206s
 167432 72500 -209.27726   38   11          - -232.30638      -  13.3 13219s
 167681 72561 -224.41708   42   13          - -232.28424      -  13.3 13259s
 167742 72559 -207.80393   43    -          - -232.28020      -  13.3 13269s
 167805 72567 -216.13511   44   11          - -232.28018      -  13.3 13286s
 167934 72643 -210.40118   43   10          - -232.26135      -  13.3 13313s
 168005 72641 -190.58680   41    6          - -232.26011      -  13.3 13319s
 168178 72652 -204.56782   49    5          - -232.25723      -  13.3 13348s
 168267 72727 -201.79273   47    -          - -232.25461      -  13.3 13373s
 168429 72744 -185.06800   47    2          - -232.24987      -  13.3 13415s
 168587 72826 -207.46324   45   10          - -232.20800      -  13.3 13445s
 168882 72871 -200.45148   41   18          - -232.18878      -  13.3 13505s
 168953 72877 -200.91391   48    -          - -232.18627      -  13.3 13518s
 169132 72956 -213.92487   36   19          - -232.17809      -  13.3 13555s
 169205 72956 -193.97203   46    4          - -232.17511      -  13.3 13565s
 169415 73054 -201.14828   43    -          - -232.14355      -  13.3 13607s
 169486 73057 -201.79125   44    5          - -232.14117      -  13.3 13617s
 169720 73131 -198.15839   46    -          - -232.11213      -  13.4 13666s
 169795 73134 -209.04538   34   17          - -232.11009      -  13.4 13681s
 169864 73136 -194.71990   47    -          - -232.10999      -  13.4 13697s
 170005 73206 -212.26403   42    5          - -232.09524      -  13.4 13729s
 170264 73303 -218.97457   34   15          - -232.07161      -  13.4 13777s
 170345 73305 -215.87706   41   11          - -232.07128      -  13.4 13784s
 170422 73311 -211.45054   41    6          - -232.07051      -  13.4 13796s
 170584 73366 -206.15473   51    -          - -232.06052      -  13.4 13821s
 170658 73371 -206.69753   45    -          - -232.05565      -  13.4 13847s
 170872 73452 -196.83126   44    -          - -232.02009      -  13.4 13906s
 170932 73455 -196.72799   44    7          - -232.01945      -  13.4 13918s
 171166 73531 -199.83660   41    9          - -231.99106      -  13.4 13965s
 171364 73633 -204.45418   36   24          - -231.97431      -  13.4 14001s
 171449 73631 -200.53549   38    4          - -231.97220      -  13.4 14011s
 171668 73704 -194.82294   47    2          - -231.96380      -  13.4 14048s
 171820 73714 -202.18219   42    6          - -231.96050      -  13.4 14077s
 171976 73785 -210.87967   40   11          - -231.93551      -  13.4 14106s
 172191 73795 -212.88373   46    2          - -231.91352      -  13.4 14138s
 172271 73880 -205.27856   44    6          - -231.91184      -  13.4 14152s
 172331 73880 -206.23203   43    5          - -231.91071      -  13.4 14160s
 172409 73886 -203.58702   42    8          - -231.90737      -  13.4 14176s
 172561 73961 -202.55027   42    9          - -231.89116      -  13.4 14197s
 172645 73962 -193.37885   49    -          - -231.89071      -  13.4 14209s
 172899 74033 -196.09743   47    -          - -231.87430      -  13.4 14272s
 172963 74036 -197.83461   46    2          - -231.87149      -  13.4 14284s
 173146 74119 -205.85273   45    -          - -231.84195      -  13.4 14330s
 173222 74124 -207.94039   38   11          - -231.83865      -  13.4 14344s
 173473 74221 -211.43402   48    -          - -231.81825      -  13.4 14386s
 173778 74313 -201.10260   46    5          - -231.79175      -  13.4 14445s
 173913 74317 -202.10463   45    5          - -231.78715      -  13.4 14473s
 174058 74369 -201.72116   46    6          - -231.77137      -  13.4 14505s
 174239 74381 -199.53939   49    2          - -231.76167      -  13.4 14527s
 174304 74444 -201.45575   48    5          - -231.75847      -  13.4 14537s
 174626 74551 -226.43062   39   26          - -231.72268      -  13.4 14591s
 174708 74552 -197.57811   45    -          - -231.71839      -  13.4 14604s
 174788 74554 -197.58147   50    -          - -231.71824      -  13.4 14617s
 174939 74634 -215.67694   48    2          - -231.70490      -  13.4 14645s
 175011 74634 -205.48565   44   10          - -231.70488      -  13.4 14658s
 175235 74701 -194.50362   47    5          - -231.68856      -  13.4 14695s
 175470 74783 -193.06392   47    2          - -231.66235      -  13.4 14754s
 175544 74781 -187.17929   48    2          - -231.66041      -  13.4 14762s
 175780 74836 -206.64364   48    5          - -231.64533      -  13.4 14799s
 175924 74847 -213.07540   42   23          - -231.64427      -  13.4 14829s
 176079 74915 -211.39004   43    8          - -231.62092      -  13.4 14874s
 176384 74996 -204.53634   46    -          - -231.59849      -  13.4 14942s
 176553 75004 -212.78282   53    -          - -231.58596      -  13.4 14968s
 176612 75074 -185.48776   46    -          - -231.57794      -  13.4 14980s
 176682 75076 -200.14966   43    2          - -231.57423      -  13.4 14993s
 176890 75182 -223.77905   47   11          - -231.55616      -  13.4 15040s
 176964 75186 -210.56622   43    6          - -231.55239      -  13.4 15059s
 177039 75189 -208.14727   43   18          - -231.55146      -  13.4 15075s
 177206 75258 -190.04744   42    3          - -231.52742      -  13.4 15106s
 177290 75259 -204.36091   40    -          - -231.52539      -  13.4 15120s
 177511 75332 -202.74162   40   10          - -231.51113      -  13.4 15175s
 177760 75409 -182.09345   45   13          - -231.48676      -  13.4 15218s
 177897 75417 -199.23269   45    -          - -231.48072      -  13.4 15249s
 178047 75499 -206.92094   39    2          - -231.45777      -  13.4 15284s
 178114 75498 -198.27669   46    4          - -231.45760      -  13.4 15299s
 178329 75555 -211.85586   49    -          - -231.43056      -  13.4 15346s
 178397 75563 -204.39168   46    4          - -231.42719      -  13.4 15368s
 178577 75637 -221.60885   43   20          - -231.41254      -  13.4 15413s
 178716 75642 -206.45456   45    6          - -231.40857      -  13.4 15447s
 178786 75645 -205.78404   37   11          - -231.39371      -  13.4 15465s
 178867 75706 -181.85821   45    5          - -231.38954      -  13.4 15479s
 179198 75816 -193.79189   38   14          - -231.37650      -  13.4 15543s
 179427 75821 -185.26666   50    4          - -231.36045      -  13.4 15584s
 179501 75871 -205.65086   41    7          - -231.35788      -  13.4 15606s
 179607 75875 -197.04623   50    -          - -231.35453      -  13.4 15638s
 179740 75975 -206.36542   43    6          - -231.34225      -  13.4 15667s
 179801 75973 -189.54326   44    -          - -231.34169      -  13.4 15676s
 180025 76062 -211.00451   44   11          - -231.33064      -  13.4 15723s
 180095 76065 -211.17266   44    8          - -231.32831      -  13.4 15756s
 180324 76143 -210.60547   43    9          - -231.29062      -  13.4 15812s
 180374 76141 -199.52793   50    5          - -231.28689      -  13.4 15820s
 180449 76145 -207.18253   40   12          - -231.28633      -  13.4 15846s
 180572 76221 -213.58184   41   18          - -231.27477      -  13.5 15879s
 180649 76226 -215.82049   32   24          - -231.27233      -  13.4 15890s
 180875 76286 -170.22237   41    -          - -231.25966      -  13.4 15927s
 181077 76294 -196.14987   43    -          - -231.24133      -  13.5 15980s
 181150 76384 -216.17609   48    6          - -231.23973      -  13.5 16009s
 181456 76454 -208.27486   42    6          - -231.22424      -  13.5 16068s
 181520 76454 -211.60009   42    8          - -231.22332      -  13.5 16079s
 181694 76543 -204.05266   40    -          - -231.21579      -  13.5 16119s
 181978 76631 -216.43944   40   24          - -231.18595      -  13.5 16176s
 182282 76699 -222.48364   37   23          - -231.17042      -  13.5 16238s
 182475 76706 -210.39502   38    8          - -231.15043      -  13.5 16283s
 182539 76792 -197.70469   45    -          - -231.14631      -  13.5 16298s
 182687 76797 -196.66602   45    2          - -231.14335      -  13.5 16326s
 182840 76870 -206.97405   46    -          - -231.13540      -  13.5 16379s
 183141 76970 -188.67055   39    9          - -231.10635      -  13.5 16464s
 183217 76971 -187.00905   49    -          - -231.10513      -  13.5 16477s
 183448 77026 -203.60300   46    -          - -231.08776      -  13.5 16525s
 183661 77100 -220.18953   47    9          - -231.06512      -  13.5 16591s
 183973 77185 -208.21063   39   20          - -231.04620      -  13.5 16660s
 184271 77284 -209.37853   35   22          - -231.02261      -  13.5 16734s
 184345 77287 -207.03843   43    9          - -231.02085      -  13.5 16752s
 184552 77344 -194.71348   48    2          - -230.99015      -  13.5 16788s
 184781 77396 -203.20142   42    5          - -230.98642      -  13.5 16835s
 184982 77409 -221.54411   48    -          - -230.96950      -  13.5 16864s
 185052 77511 -195.43942   43    -          - -230.96862      -  13.5 16889s
 185195 77508 -204.28766   45    8          - -230.96505      -  13.5 16916s
 185342 77585 -194.49876   48    -          - -230.95515      -  13.5 16946s
 185479 77593 -213.43309   37   11          - -230.95151      -  13.5 16988s
 185627 77686 -190.47920   53    -          - -230.91621      -  13.5 17021s
 185691 77686 -192.34946   46    3          - -230.91621      -  13.5 17029s
 185916 77740 -206.77750   39    5          - -230.89736      -  13.5 17075s
 186180 77811 -218.04107   46    5          - -230.88731      -  13.5 17132s
 186486 77926 -202.59649   43    2          - -230.87275      -  13.5 17202s
 186566 77929 -219.09475   37   33          - -230.87247      -  13.5 17224s
 186772 77999 -199.51356   46    5          - -230.84826      -  13.5 17269s
 187069 78031 -186.01284   43    3          - -230.83007      -  13.5 17329s
 187134 78035 -206.46215   50    3          - -230.82831      -  13.5 17347s
 187307 78132 -195.33547   47    -          - -230.80883      -  13.5 17400s
 187490 78141 -205.24420   44    5          - -230.79507      -  13.5 17445s
 187565 78211 -210.72236   41   26          - -230.79363      -  13.5 17453s
 187862 78344 -203.65941   44   13          - -230.77942      -  13.5 17508s
 188171 78526 -189.64811   52    -          - -230.76581      -  13.5 17573s
 188230 78546 -213.80147   38   19          - -230.76478      -  13.5 17579s
 188402 78673 -209.38269   49    -          - -230.75440      -  13.5 17631s
 188483 78713 -217.02217   41   16          - -230.75344      -  13.5 17640s
 188706 78852 -188.64325   42    7          - -230.74063      -  13.5 17677s
 189002 79018 -186.01904   42    6          - -230.72585      -  13.5 17735s
 189183 79088 -205.90952   42   11          - -230.71325      -  13.5 17768s
 189241 79190 -216.23761   37   27          - -230.71227      -  13.5 17791s
 189525 79353 -192.83960   43    -          - -230.69503      -  13.5 17855s
 189595 79381 -183.98800   42    -          - -230.69402      -  13.5 17867s
 189784 79513 -190.33618   47    6          - -230.67004      -  13.5 17925s
 190014 79662 -202.86415   39   11          - -230.64305      -  13.5 17979s
 190324 79849 -213.99547   39   11          - -230.62712      -  13.5 18047s
 190590 80023 -201.23090   37   14          - -230.61230      -  13.5 18094s
 190668 80056 -219.11874   39   23          - -230.60501      -  13.5 18101s
 190897 80210 -206.31081   43    9          - -230.58585      -  13.5 18143s
 191145 80371 -200.61651   47    -          - -230.56880      -  13.5 18200s
 191443 80546 -204.25838   41    8          - -230.55120      -  13.5 18266s
 191639 80620 -205.60520   46    2          - -230.53730      -  13.6 18312s
 191715 80716 -183.26293   40    8          - -230.53439      -  13.6 18332s
 191800 80750 -199.56892   47    -          - -230.53196      -  13.6 18360s
 192026 80887 -186.52251   42    2          - -230.51036      -  13.6 18419s
 192077 80909 -200.35755   44   11          - -230.50836      -  13.6 18426s
 192187 80945 -187.95588   47    -          - -230.49847      -  13.6 18451s
 192241 81025 -184.86521   44    3          - -230.49456      -  13.6 18471s
 192382 81082 -194.38594   48    7          - -230.49166      -  13.6 18505s
 192514 81197 -185.54580   37   17          - -230.47957      -  13.6 18533s
 192586 81222 -213.70731   46    -          - -230.47509      -  13.6 18557s
 192817 81402 -190.91644   41    5          - -230.45734      -  13.6 18625s
 192954 81446 -209.83179   40   19          - -230.45130      -  13.6 18654s
 193087 81529 -185.04084   47    -          - -230.43301      -  13.6 18682s
 193222 81589 -214.47579   40   17          - -230.43074      -  13.6 18717s
 193354 81700 -199.51262   51    -          - -230.41344      -  13.6 18756s
 193589 81840 -187.78238   40    3          - -230.39886      -  13.6 18806s
 193660 81872 -185.10135   46    -          - -230.39852      -  13.6 18815s
 193863 82013 -203.36168   44   10          - -230.38824      -  13.6 18852s
 194005 82066 -198.93686   49    -          - -230.38512      -  13.6 18889s
 194135 82185 -199.53715   43    5          - -230.36973      -  13.6 18926s
 194201 82210 -206.86873   49    2          - -230.36475      -  13.6 18938s
 194441 82356 -199.62478   39    5          - -230.34809      -  13.6 18991s
 194657 82502 -193.36133   43    4          - -230.33224      -  13.6 19030s
 194724 82522 -201.95826   50    -          - -230.33103      -  13.6 19044s
 194947 82659 -205.91006   44    5          - -230.31880      -  13.6 19091s
 195155 82748 -186.05462   44    -          - -230.30370      -  13.6 19148s
 195225 82824 -201.83310   47    5          - -230.30132      -  13.6 19165s
 195525 82992 -186.49887   48    -          - -230.28170      -  13.6 19259s
 195573 83013 -198.07083   45    5          - -230.27820      -  13.6 19268s
 195671 83047 -198.44570   46    -          - -230.25488      -  13.6 19308s
 195713 83102 -199.24282   48    7          - -230.25018      -  13.6 19319s
 195793 83138 -192.84814   43    -          - -230.24996      -  13.6 19345s
 196018 83300 -205.87649   45    -          - -230.21916      -  13.6 19403s
 196226 83382 -221.66826   39   22          - -230.20363      -  13.6 19457s
 196292 83472 -203.72350   50    4          - -230.19995      -  13.6 19471s
 196362 83498 -194.63356   43    4          - -230.19886      -  13.6 19490s
 196427 83523 -201.01111   42    2          - -230.19877      -  13.6 19506s
 196565 83651 -190.34174   53    -          - -230.17845      -  13.6 19543s
 196698 83693 -199.68942   46    2          - -230.17319      -  13.6 19566s
 196849 83789 -186.37711   42    -          - -230.16007      -  13.6 19621s
 196912 83815 -198.70696   46    -          - -230.15883      -  13.6 19633s
 197073 83946 -190.92094   46    4          - -230.14732      -  13.6 19684s
 197353 84130 -193.74799   53    -          - -230.12589      -  13.6 19748s
 197425 84155 -195.68483   52    -          - -230.12384      -  13.6 19764s
 197624 84291 -203.27237   41    2          - -230.10902      -  13.6 19806s
 197689 84316 -199.17056   42    -          - -230.10738      -  13.6 19823s
 197913 84470 -209.99532   40   15          - -230.08705      -  13.6 19876s
 198127 84634 -214.25402   35   21          - -230.07566      -  13.6 19923s
 198204 84657 -204.39851   47    -          - -230.07452      -  13.6 19946s
 198365 84719 -180.24764   49    -          - -230.05719      -  13.6 19984s
 198443 84806 -199.37175   44    3          - -230.05710      -  13.6 20001s
 198594 84867 -205.37274   46    5          - -230.05601      -  13.6 20040s
 198742 84969 -189.16736   49    2          - -230.03617      -  13.6 20061s
 198810 85001 -186.08265   49    2          - -230.03560      -  13.6 20083s
 199020 85125 -192.08268   48    -          - -230.01637      -  13.6 20143s
 199057 85140 -207.90403   39   10          - -230.01584      -  13.6 20152s
 199113 85164 -189.05667   47    -          - -230.01415      -  13.6 20164s
 199160 85179 -189.04538   48    -          - -230.00371      -  13.6 20174s
 199215 85263 -217.67710   35   13          - -230.00363      -  13.6 20188s
 199437 85348 -191.93623   47    4          - -229.99499      -  13.6 20248s
 199513 85414 -193.10642   40    7          - -229.99110      -  13.6 20279s
 199581 85443 -189.77495   47    3          - -229.99035      -  13.6 20291s
 199777 85587 -206.77487   41    4          - -229.96518      -  13.6 20341s
 199911 85635 -210.61196   48    3          - -229.96408      -  13.6 20366s
 199981 85662 -190.00657   41    5          - -229.95532      -  13.6 20388s
 200050 85748 -204.76290   39   16          - -229.95106      -  13.6 20414s
 200273 85838 -201.12110   46    6          - -229.93140      -  13.6 20477s
 200351 85903 -213.75716   43   18          - -229.93083      -  13.6 20499s
 200405 85925 -201.62199   45    -          - -229.93008      -  13.6 20513s
 200558 86034 -195.86428   50    6          - -229.91530      -  13.6 20565s
 200697 86089 -190.34222   41    -          - -229.91248      -  13.6 20604s
 200845 86227 -186.32246   42    4          - -229.89123      -  13.6 20647s
 200918 86254 -202.92343   44    5          - -229.88974      -  13.6 20666s
 201134 86405 -195.68748   50    2          - -229.87797      -  13.6 20716s
 201212 86430 -199.14000   46    6          - -229.87454      -  13.6 20729s
 201409 86560 -203.51218   45    -          - -229.86716      -  13.6 20785s
 201473 86588 -211.25087   39   15          - -229.86557      -  13.6 20800s
 201551 86614 -199.50537   48    -          - -229.86500      -  13.6 20824s
 201687 86716 -187.88490   51    -          - -229.85056      -  13.6 20881s
 201781 86757 -217.23786   37   23          - -229.84688      -  13.6 20910s
 201904 86862 -216.18672   42   18          - -229.83569      -  13.6 20944s
 202052 86921 -194.01662   41    8          - -229.83112      -  13.6 21001s
 202128 86953 -190.13758   44    -          - -229.81935      -  13.6 21025s
 202195 87030 -200.86335   47    5          - -229.81712      -  13.6 21045s
 202336 87087 -185.56265   51    -          - -229.81414      -  13.6 21085s
 202473 87209 -200.06152   45   10          - -229.79438      -  13.6 21114s
 202604 87261 -197.19487   40    6          - -229.78994      -  13.6 21156s
 202753 87370 -205.23703   48    3          - -229.77256      -  13.6 21192s
 202802 87389 -207.75659   41   19          - -229.76997      -  13.6 21199s
 202915 87430 -192.03888   46    9          - -229.76572      -  13.6 21220s
 202979 87500 -209.30469   37    6          - -229.76555      -  13.7 21244s
 203037 87525 -194.84623   50    -          - -229.75873      -  13.7 21270s
 203248 87681 -182.27908   46    -          - -229.73663      -  13.7 21337s
 203302 87699 -197.52926   49    -          - -229.73653      -  13.7 21349s
 203513 87831 -200.52785   35   24          - -229.71891      -  13.7 21399s
 203557 87842 -193.50789   45    4          - -229.71860      -  13.7 21406s
 203671 87886 -189.42441   44   10          - -229.71450      -  13.7 21441s
 203733 87977 -184.74573   43    -          - -229.71311      -  13.7 21467s
 203804 88000 -205.90165   42    -          - -229.71093      -  13.7 21485s
 203995 88154 -178.87458   50    -          - -229.69057      -  13.7 21527s
 204054 88174 -218.17529   43   14          - -229.68730      -  13.7 21537s
 204200 88227 -213.09486   37   15          - -229.67179      -  13.7 21569s
 204266 88296 -176.60279   46    -          - -229.66350      -  13.7 21583s
 204313 88316 -182.70777   43    -          - -229.66284      -  13.7 21598s
 204387 88348 -197.83188   51    -          - -229.66275      -  13.7 21615s
 204499 88447 -204.02572   37    2          - -229.64618      -  13.7 21649s
 204560 88467 -184.22522   44    -          - -229.64597      -  13.7 21672s
 204766 88601 -179.36691   41    -          - -229.63576      -  13.7 21731s
 204898 88650 -193.93201   48    3          - -229.62997      -  13.7 21771s
 205043 88774 -206.46074   40   11          - -229.61669      -  13.7 21817s
 205246 88853 -185.24691   47    -          - -229.60480      -  13.7 21880s
 205314 88939 -213.23070   47   16          - -229.60176      -  13.7 21902s
 205511 89078 -203.31342   41   13          - -229.59432      -  13.7 21947s
 205579 89103 -207.42226   38   21          - -229.59275      -  13.7 21967s
 205656 89130 -186.17241   49    -          - -229.58790      -  13.7 21990s
 205815 89255 -188.14594   38    4          - -229.57878      -  13.7 22035s
 205886 89283 -187.24799   44    4          - -229.57859      -  13.7 22045s
 205951 89311 -193.07223   36    5          - -229.57683      -  13.7 22067s
 206093 89408 -192.27298   45    2          - -229.57345      -  13.7 22099s
 206379 89573 -199.50768   49    -          - -229.54953      -  13.7 22197s
 206448 89604 -209.26379   46    3          - -229.54878      -  13.7 22217s
 206587 89660 -196.57607   42    5          - -229.53050      -  13.7 22261s
 206655 89724 -205.19754   36   24          - -229.52681      -  13.7 22293s
 206705 89741 -205.78861   44   13          - -229.52525      -  13.7 22306s
 206868 89898 -185.90432   48    -          - -229.51321      -  13.7 22366s
 206931 89920 -204.37965   50    -          - -229.50812      -  13.7 22387s
 207139 90057 -201.69437   50    -          - -229.49164      -  13.7 22458s
 207407 90208 -205.76242   43   11          - -229.47665      -  13.7 22530s
 207513 90246 -209.25134   46    5          - -229.47211      -  13.7 22556s
 207629 90351 -191.35421   45    2          - -229.45750      -  13.7 22582s
 207690 90377 -194.10408   49    -          - -229.45750      -  13.7 22600s
 207840 90438 -201.99860   41    6          - -229.44127      -  13.7 22628s
 207906 90536 -206.70826   36   20          - -229.44122      -  13.7 22648s
 207957 90552 -208.68578   48    8          - -229.44056      -  13.7 22656s
 208167 90682 -195.67825   37   19          - -229.42692      -  13.7 22720s
 208375 90764 -216.26077   43   15          - -229.42146      -  13.7 22758s
 208450 90851 -208.61648   41   27          - -229.41937      -  13.7 22779s
 208611 90908 -209.33230   37   19          - -229.40640      -  13.7 22811s
 208669 90991 -208.94752   42   12          - -229.40340      -  13.7 22838s
 208744 91021 -211.17019   47    -          - -229.40300      -  13.7 22867s
 208810 91048 -190.39852   43   11          - -229.40300      -  13.7 22882s
 208951 91162 -197.29120   46    6          - -229.39362      -  13.7 22920s
 209222 91327 -211.75897   38   18          - -229.37437      -  13.7 23003s
 209279 91344 -213.30612   50    7          - -229.37423      -  13.7 23024s
 209444 91415 -197.44785   42    2          - -229.35769      -  13.7 23070s
 209512 91527 -195.80987   46    9          - -229.35740      -  13.7 23090s
 209571 91548 -184.66216   46    6          - -229.35513      -  13.7 23100s
 209715 91601 -211.85549   37   15          - -229.34943      -  13.7 23141s
 209793 91651 -190.74814   47    4          - -229.34634      -  13.7 23156s
 209997 91825 -213.84215   40   19          - -229.33203      -  13.7 23208s
 210294 92000 -199.48319   33   32          - -229.32424      -  13.7 23267s
 210360 92017 -200.79364   43   16          - -229.32010      -  13.7 23278s
 210486 92072 -213.15404   47    6          - -229.31071      -  13.7 23327s
 210565 92135 -195.16177   43    6          - -229.30764      -  13.7 23345s
 210639 92165 -189.90447   40    8          - -229.30620      -  13.7 23366s
 210853 92332 -190.50986   49    7          - -229.28739      -  13.7 23426s
 210993 92384 -201.53526   42    7          - -229.28449      -  13.7 23458s
 211131 92515 -197.88399   40   10          - -229.27440      -  13.7 23502s
 211201 92537 -196.19849   44    -          - -229.26984      -  13.7 23518s
 211390 92643 -213.71916   45    -          - -229.25561      -  13.7 23567s
 211618 92802 -194.14167   45    -          - -229.23967      -  13.7 23660s
 211749 92848 -194.43797   42   14          - -229.23834      -  13.7 23693s
 211902 92968 -188.72465   40    2          - -229.22979      -  13.7 23724s
 211971 92993 -198.77738   45    2          - -229.22839      -  13.7 23742s
 212174 93145 -188.50443   44    -          - -229.21144      -  13.7 23802s
 212379 93223 -200.22414   50    -          - -229.19382      -  13.7 23847s
 212453 93312 -200.62747   44   12          - -229.18987      -  13.7 23865s
 212681 93452 -207.75698   39    9          - -229.18378      -  13.7 23914s
 212968 93627 -212.59893   42   10          - -229.15953      -  13.7 23985s
 213174 93710 -197.96644   53    2          - -229.14443      -  13.7 24038s
 213249 93794 -201.36371   41   18          - -229.14379      -  13.7 24069s
 213311 93816 -191.80200   49    7          - -229.14285      -  13.7 24079s
 213380 93841 -203.30333   39   16          - -229.14063      -  13.7 24095s
 213478 93904 -196.10497   45    -          - -229.13213      -  13.7 24135s
 213552 93937 -215.11493   47    2          - -229.13107      -  13.7 24149s
 213710 94006 -197.05565   46    -          - -229.11858      -  13.7 24200s
 213776 94113 -172.65779   50    -          - -229.11614      -  13.7 24239s
 214022 94231 -182.58615   41    -          - -229.09990      -  13.7 24308s
 214275 94402 -212.64518   42   14          - -229.09086      -  13.7 24390s
 214418 94453 -198.22417   40   16          - -229.07590      -  13.7 24433s
 214477 94544 -188.45838   42    -          - -229.07538      -  13.7 24448s
 214607 94589 -215.99585   46    7          - -229.07263      -  13.8 24499s
 214742 94701 -218.86992   39   23          - -229.06009      -  13.8 24549s
 214879 94748 -175.88098   43    -          - -229.05753      -  13.8 24586s
 214938 94771 -186.81120   40    -          - -229.04919      -  13.8 24610s
 214996 94847 -199.34089   41    5          - -229.04889      -  13.8 24644s
 215150 94918 -191.25717   36   11          - -229.04747      -  13.8 24664s
 215299 95018 -172.02420   45    -          - -229.03008      -  13.8 24704s
 215465 95084 -200.16510   48    -          - -229.02419      -  13.8 24734s
 215515 95169 -202.65766   45    2          - -229.02236      -  13.8 24744s
 215792 95329 -197.26700   41   11          - -229.01087      -  13.8 24813s
 215854 95350 -206.67107   42    3          - -229.00842      -  13.8 24824s
 216066 95517 -196.81015   36    9          - -228.99087      -  13.8 24868s
 216138 95541 -194.11153   49    5          - -228.99053      -  13.8 24887s
 216205 95567 -181.46033   48    -          - -228.98601      -  13.8 24906s
 216326 95681 -177.97902   43    2          - -228.96743      -  13.8 24932s
 216382 95699 -191.98325   40    6          - -228.96579      -  13.8 24943s
 216531 95750 -203.68369   49    -          - -228.96312      -  13.8 24975s
 216603 95805 -198.91247   43    -          - -228.95566      -  13.8 25005s
 216651 95830 -205.07980   43   10          - -228.95474      -  13.8 25011s
 216808 95937 -196.93423   49    -          - -228.94888      -  13.8 25056s
 217064 96105 -209.94993   39    8          - -228.93003      -  13.8 25129s
 217123 96126 -204.63331   47    5          - -228.92260      -  13.8 25148s
 217182 96146 -192.94241   46    -          - -228.92043      -  13.8 25170s
 217251 96175 -192.63935   38    9          - -228.90800      -  13.8 25200s
 217323 96271 -206.69932   43    -          - -228.90664      -  13.8 25230s
 217620 96452 -205.63487   37   24          - -228.89105      -  13.8 25313s
 217686 96474 -195.40937   48    -          - -228.88955      -  13.8 25335s
 217752 96499 -193.37835   45    2          - -228.88934      -  13.8 25352s
 217886 96599 -163.86067   41   13          - -228.87297      -  13.8 25384s
 217933 96616 -181.06732   45    2          - -228.87250      -  13.8 25395s
 218091 96761 -211.67482   42    5          - -228.86595      -  13.8 25434s
 218164 96788 -217.73503   38   15          - -228.86187      -  13.8 25453s
 218374 96921 -195.76693   50    -          - -228.85053      -  13.8 25516s
 218661 97072 -175.10188   47    -          - -228.82943      -  13.8 25614s
 218793 97129 -183.74349   47    -          - -228.82909      -  13.8 25653s
 218919 97247 -194.27771   43   14          - -228.81083      -  13.8 25703s
 219133 97331 -204.87352   41   14          - -228.79272      -  13.8 25765s
 219206 97428 -206.44122   41    7          - -228.79196      -  13.8 25794s
 219388 97560 -198.76617   46    2          - -228.78033      -  13.8 25840s
 219597 97638 -191.04685   42    -          - -228.77132      -  13.8 25885s
 219662 97727 -204.13283   43    3          - -228.77120      -  13.8 25903s
 219933 97856 -212.23875   35   26          - -228.75173      -  13.8 25963s
 219987 97878 -201.50430   40   12          - -228.75173      -  13.8 25969s
 220223 98034 -186.60973   45    -          - -228.74997      -  13.8 26007s
 220510 98195 -200.10909   41    -          - -228.73444      -  13.8 26104s
 220745 98329 -215.69906   36   24          - -228.72561      -  13.8 26181s
 220940 98412 -185.82709   44    -          - -228.71582      -  13.8 26242s
 221002 98492 -204.21351   50    3          - -228.71581      -  13.8 26263s
 221257 98645 -213.38872   46   11          - -228.69295      -  13.8 26352s
 221300 98654 -196.50845   48    -          - -228.68945      -  13.8 26361s
 221447 98777 -199.79544   47    9          - -228.68176      -  13.8 26402s
 221706 98931 -197.25668   39   11          - -228.66438      -  13.8 26491s
 221771 98958 -205.33001   41   21          - -228.66302      -  13.8 26524s
 221914 99009 -217.56828   42   18          - -228.65068      -  13.8 26549s
 221980 99094 -185.58973   49    -          - -228.64550      -  13.8 26574s
 222113 99146 -187.89260   48    -          - -228.64382      -  13.8 26610s
 222242 99266 -189.36160   45    -          - -228.63118      -  13.8 26662s
 222528 99406 -208.97474   49    -          - -228.61143      -  13.8 26771s
 222717 99544 -190.73431   46    2          - -228.60120      -  13.8 26827s
 222857 99598 -173.40412   50    -          - -228.59885      -  13.8 26863s
 222990 99698 -199.31117   47    4          - -228.58022      -  13.8 26902s
 223054 99722 -203.99925   43    5          - -228.57908      -  13.8 26916s
 223259 99882 -200.90174   53    -          - -228.56458      -  13.8 26999s
 223463 99953 -185.96709   46    3          - -228.54547      -  13.8 27044s
 223526 100019 -195.55120   45    8          - -228.53333      -  13.8 27057s
 223585 100042 -204.56638   51    -          - -228.53296      -  13.8 27086s
 223703 100087 -182.69934   44    3          - -228.52074      -  13.8 27120s
 223774 100172 -187.70916   36   17          - -228.51921      -  13.8 27153s
 223963 100308 -220.24240   42   10          - -228.50977      -  13.8 27198s
 224164 100384 -208.33648   44   19          - -228.49480      -  13.8 27267s
 224225 100447 -205.11376   46    -          - -228.49192      -  13.8 27301s
 224303 100485 -190.31647   48    -          - -228.49185      -  13.8 27318s
 224507 100614 -171.45214   44    -          - -228.48540      -  13.8 27391s
 224784 100787 -194.60252   43    2          - -228.45936      -  13.8 27464s
 224927 100843 -200.49241   42   10          - -228.45840      -  13.8 27501s
 225060 100939 -202.30324   44    2          - -228.45232      -  13.8 27546s
 225281 101071 -202.89830   45    2          - -228.44426      -  13.8 27602s
 225347 101097 -192.58883   51    -          - -228.44365      -  13.8 27617s
 225554 101262 -201.41386   46    6          - -228.43390      -  13.8 27683s
 225759 101334 -196.08627   49    -          - -228.42326      -  13.8 27743s
 225824 101408 -187.23498   45    -          - -228.42013      -  13.8 27765s
 225989 101471 -188.89653   49    3          - -228.41033      -  13.8 27805s
 226042 101565 -203.10120   51    -          - -228.41017      -  13.8 27833s
 226301 101715 -179.26009   41   15          - -228.39204      -  13.8 27925s
 226353 101732 -183.28791   40    -          - -228.39187      -  13.9 27943s
 226556 101862 -206.78805   43    6          - -228.38039      -  13.9 28003s
 226798 102003 -210.78824   46    7          - -228.36684      -  13.9 28069s
 227067 102177 -220.27305   30   31          - -228.35222      -  13.9 28154s
 227138 102200 -204.72574   46    7          - -228.35137      -  13.9 28169s
 227212 102231 -193.42647   47    4          - -228.34775      -  13.9 28189s
 227349 102325 -204.36013   37   30          - -228.33876      -  13.9 28232s
 227398 102342 -195.15215   44    5          - -228.33833      -  13.9 28246s
 227569 102496 -192.80329   46    6          - -228.32473      -  13.9 28295s
 227696 102537 -207.85465   43   15          - -228.31914      -  13.9 28327s
 227799 102639 -199.82635   48    3          - -228.31181      -  13.9 28352s
 227867 102664 -212.73273   42   11          - -228.31167      -  13.9 28372s
 228078 102811 -198.66053   52    -          - -228.29711      -  13.9 28420s
 228281 102937 -180.05754   51    5          - -228.27897      -  13.9 28477s
 228342 102960 -200.53449   42    3          - -228.27762      -  13.9 28498s
 228462 103006 -198.01718   46    3          - -228.26731      -  13.9 28530s
 228525 103052 -209.88420   48    4          - -228.26654      -  13.9 28561s
 228595 103091 -198.92487   44    6          - -228.26615      -  13.9 28579s
 228807 103232 -165.13583   50    -          - -228.25083      -  13.9 28653s
 228866 103259 -199.37326   40   14          - -228.24912      -  13.9 28674s
 229049 103410 -190.97675   45    7          - -228.23770      -  13.9 28726s
 229299 103529 -176.91317   48    -          - -228.22960      -  13.9 28800s
 229363 103558 -195.70720   47    -          - -228.22932      -  13.9 28835s
 229564 103683 -206.54013   46    8          - -228.21967      -  13.9 28872s
 229713 103736 -198.64645   45    6          - -228.21147      -  13.9 28904s
 229751 103815 -204.00676   50    3          - -228.21087      -  13.9 28921s
 229945 103888 -206.97858   36   20          - -228.19158      -  13.9 28987s
 230013 103958 -186.22358   46    4          - -228.19145      -  13.9 29012s
 230074 103986 -181.40445   41    4          - -228.19043      -  13.9 29030s
 230288 104135 -199.87853   44    -          - -228.18060      -  13.9 29073s
 230353 104162 -194.83983   48    2          - -228.17968      -  13.9 29096s
 230558 104303 -203.14862   38   12          - -228.16173      -  13.9 29157s
 230625 104326 -201.95180   43   10          - -228.16164      -  13.9 29184s
 230837 104457 -208.28830   32   31          - -228.14813      -  13.9 29247s
 230991 104514 -189.72548   42    6          - -228.13814      -  13.9 29279s
 231051 104595 -194.54155   42    2          - -228.13212      -  13.9 29322s
 231115 104623 -200.50756   35   14          - -228.13061      -  13.9 29334s
 231314 104762 -215.19785   37   29          - -228.12375      -  13.9 29389s
 231575 104896 -195.13078   39   13          - -228.10389      -  13.9 29451s
 231802 105050 -202.06663   45   10          - -228.09526      -  13.9 29503s
 231942 105103 -189.02179   52    -          - -228.08750      -  13.9 29541s
 232068 105199 -191.83559   40    5          - -228.07476      -  13.9 29599s
 232335 105365 -211.16970   40    9          - -228.05418      -  13.9 29678s
 232388 105381 -209.08696   48    -          - -228.05323      -  13.9 29702s
 232497 105421 -197.89114   43    9          - -228.04179      -  13.9 29725s
 232544 105517 -201.15893   44    -          - -228.03943      -  13.9 29745s
 232686 105571 -195.69955   43    4          - -228.03432      -  13.9 29781s
 232837 105661 -182.58660   51    -          - -228.02703      -  13.9 29810s
 233090 105818 -197.01966   42   17          - -228.01590      -  13.9 29923s
 233302 105948 -180.12210   41    -          - -228.00011      -  13.9 30006s
 233367 105976 -208.88244   42   10          - -227.99951      -  13.9 30022s
 233578 106113 -211.57901   38    7          - -227.99004      -  13.9 30094s
 233644 106137 -211.59937   42   17          - -227.98753      -  13.9 30112s
 233718 106168 -206.75648   43    7          - -227.98610      -  13.9 30157s
 233851 106278 -180.56618   44    -          - -227.97497      -  13.9 30223s
 233953 106318 -175.69719   44   11          - -227.96803      -  13.9 30268s
 234063 106419 -196.89743   38   16          - -227.95845      -  13.9 30307s
 234127 106445 -216.03892   41    8          - -227.95821      -  13.9 30328s
 234323 106551 -206.14338   41   14          - -227.94799      -  13.9 30382s
 234602 106752 -197.12245   34   22          - -227.93344      -  13.9 30472s
 234661 106771 -158.83990   50    3          - -227.93163      -  13.9 30484s
 234834 106875 -191.40792   49    -          - -227.92462      -  13.9 30538s
 234906 106907 -179.27899   46    -          - -227.92443      -  13.9 30559s
 235110 107035 -196.30550   50    6          - -227.90698      -  13.9 30615s
 235227 107082 -201.60445   35   13          - -227.90432      -  13.9 30656s
 235296 107111 -193.86069   50    -          - -227.90095      -  13.9 30671s
 235357 107163 -183.69432   45    -          - -227.89990      -  13.9 30694s
 235523 107229 -210.98946   44    -          - -227.88451      -  13.9 30767s
 235575 107286 -204.62814   41    7          - -227.88359      -  13.9 30805s
 235771 107375 -213.46148   37   23          - -227.86784      -  13.9 30882s
 235827 107474 -212.11718   33   19          - -227.86474      -  13.9 30903s
 235887 107496 -203.72451   48    8          - -227.86288      -  13.9 30933s
 236113 107635 -205.04558   45    6          - -227.84755      -  13.9 30999s
 236170 107657 -196.40329   49    -          - -227.84585      -  13.9 31029s
 236302 107706 -193.95968   50    2          - -227.82928      -  13.9 31075s
 236373 107783 -217.62622   43   20          - -227.82513      -  13.9 31109s
 236424 107804 -197.42073   40    4          - -227.82330      -  13.9 31133s
 236573 107905 -201.66482   48    5          - -227.81695      -  13.9 31193s
 236637 107932 -193.72117   47    -          - -227.81449      -  13.9 31226s
 236765 107983 -207.51742   45    9          - -227.79960      -  13.9 31259s
 236836 108070 -193.73890   42    3          - -227.79611      -  13.9 31289s
 236965 108118 -198.49287   49    -          - -227.78926      -  13.9 31320s
 237088 108233 -185.81402   47    -          - -227.77977      -  13.9 31379s
 237349 108384 -201.20460   41   11          - -227.76379      -  13.9 31466s
 237407 108405 -196.61870   41    -          - -227.76151      -  13.9 31488s
 237588 108514 -212.12068   43   11          - -227.74708      -  13.9 31555s
 237721 108570 -173.32803   43    -          - -227.74549      -  13.9 31611s
 237792 108604 -198.56337   47    2          - -227.72921      -  13.9 31645s
 237859 108665 -201.42522   49    3          - -227.72479      -  14.0 31670s
 238139 108834 -202.20320   34   16          - -227.70351      -  14.0 31770s
 238315 108906 -201.10165   42   11          - -227.69736      -  14.0 31834s
 238359 108971 -201.03061   46    6          - -227.69697      -  14.0 31849s
 238616 109127 -206.87222   39   19          - -227.67547      -  14.0 31962s
 238808 109200 -199.44168   42   13          - -227.66088      -  14.0 32027s
 238871 109260 -182.02329   43    -          - -227.65688      -  14.0 32054s
 238930 109287 -206.33719   47   11          - -227.65343      -  14.0 32073s
 238986 109310 -181.52560   43    -          - -227.65207      -  14.0 32092s
 239100 109404 -188.64587   47    3          - -227.63920      -  14.0 32132s
 239166 109431 -183.90559   44    2          - -227.63794      -  14.0 32164s
 239368 109593 -208.59312   46    2          - -227.63234      -  14.0 32209s
 239652 109751 -210.97578   40   15          - -227.61890      -  14.0 32279s
 239759 109791 -208.77099   46   17          - -227.61622      -  14.0 32297s
 239875 109857 -199.05668   43   12          - -227.60789      -  14.0 32334s
 239942 109886 -191.57534   50    -          - -227.60702      -  14.0 32366s
 240012 109921 -203.99268   35   24          - -227.60701      -  14.0 32393s
 240151 110005 -178.69801   48    -          - -227.59809      -  14.0 32421s
 240339 110089 -184.05438   48    -          - -227.57901      -  14.0 32504s
 240412 110168 -195.03138   48    -          - -227.57753      -  14.0 32553s
 240631 110279 -174.68159   45    -          - -227.56723      -  14.0 32642s
 240910 110459 -183.85586   45    3          - -227.54343      -  14.0 32754s
 241176 110607 -199.75000   46    6          - -227.53252      -  14.0 32829s
 241318 110664 -192.97151   44    2          - -227.52878      -  14.0 32867s
 241441 110764 -188.94162   47    5          - -227.52061      -  14.0 32910s
 241626 110912 -210.40467   46   10          - -227.50983      -  14.0 33001s
 241689 110931 -202.68126   46    4          - -227.50483      -  14.0 33020s
 241821 110980 -177.63474   48    -          - -227.49956      -  14.0 33075s
 241888 111059 -197.08861   42    3          - -227.49473      -  14.0 33106s
 241958 111088 -187.74683   49    -          - -227.49329      -  14.0 33138s
 242135 111229 -201.22266   45    -          - -227.48375      -  14.0 33223s
 242405 111405 -201.82697   37   16          - -227.47007      -  14.0 33298s
 242480 111433 -199.24741   40   25          - -227.46916      -  14.0 33305s
 242552 111457 -190.17320   44    -          - -227.46840      -  14.0 33327s
 242687 111553 -196.55687   47    -          - -227.46665      -  14.0 33381s
 242887 111705 -206.92910   35   28          - -227.45587      -  14.0 33431s
 243007 111748 -197.01323   52    -          - -227.45124      -  14.0 33472s
 243134 111851 -209.20880   39   23          - -227.44241      -  14.0 33528s
 243205 111875 -187.44020   40   13          - -227.44165      -  14.0 33550s
 243403 111998 -214.63516   39   17          - -227.42699      -  14.0 33620s
 243532 112052 -187.40651   45    4          - -227.42343      -  14.0 33670s
 243664 112201 -198.59717   40    8          - -227.40383      -  14.0 33729s
 243722 112220 -202.93571   34   20          - -227.40333      -  14.0 33745s
 243785 112236 -183.88367   47    -          - -227.40272      -  14.0 33765s
 243920 112318 -173.93214   45    8          - -227.39648      -  14.0 33806s
 244102 112445 -212.07308   41   10          - -227.38748      -  14.0 33862s
 244168 112469 -181.92754   48    -          - -227.38436      -  14.0 33897s
 244366 112617 -198.23392   46    5          - -227.36968      -  14.0 33951s
 244431 112646 -206.41730   33   15          - -227.36745      -  14.0 33982s
 244637 112787 -193.51836   39   15          - -227.35781      -  14.0 34056s
 244693 112804 -190.59196   46    -          - -227.35655      -  14.0 34069s
 244820 112854 -188.46945   44    4          - -227.35489      -  14.0 34107s
 244883 112902 -181.27636   50    -          - -227.34969      -  14.0 34122s
 245145 113061 -190.60435   47    8          - -227.31786      -  14.0 34225s
 245377 113216 -209.55403   37   11          - -227.30436      -  14.0 34317s
 245446 113243 -202.16509   37   14          - -227.30234      -  14.0 34337s
 245647 113375 -207.23139   46    4          - -227.29471      -  14.0 34410s
 245701 113400 -189.96310   49    -          - -227.29444      -  14.0 34439s
 245901 113509 -197.98473   38    5          - -227.28228      -  14.0 34500s
 245951 113528 -190.78719   43    5          - -227.28143      -  14.0 34513s
 246123 113649 -212.25999   42   15          - -227.27251      -  14.0 34560s
 246189 113674 -205.92758   40   12          - -227.27065      -  14.0 34573s
 246359 113808 -194.85577   52    -          - -227.26265      -  14.0 34650s
 246626 113953 -211.82893   40    -          - -227.24762      -  14.0 34734s
 246684 113977 -191.09658   43    6          - -227.24672      -  14.0 34753s
 246836 114040 -192.79939   44    -          - -227.23663      -  14.0 34805s
 246906 114097 -203.43146   41   12          - -227.23459      -  14.0 34830s
 247055 114159 -202.30682   46    3          - -227.22820      -  14.0 34866s
 247094 114235 -184.74375   43    -          - -227.22676      -  14.0 34910s
 247220 114285 -194.74326   45    2          - -227.22611      -  14.0 34969s
 247345 114385 -208.75499   45   12          - -227.21181      -  14.0 35004s
 247532 114462 -194.48911   45    -          - -227.20170      -  14.0 35098s
 247608 114542 -184.21659   49    -          - -227.19547      -  14.0 35117s
 247880 114702 -213.33159   36   27          - -227.18352      -  14.0 35209s
 247945 114725 -191.26149   48    -          - -227.18276      -  14.0 35238s
 248134 114832 -196.15082   43   13          - -227.17171      -  14.0 35302s
 248186 114855 -192.28319   43   10          - -227.17036      -  14.0 35326s
 248369 114979 -203.60221   44    5          - -227.15103      -  14.0 35412s
 248622 115141 -213.79918   34   29          - -227.13789      -  14.0 35510s
 248683 115163 -181.43437   43    -          - -227.13725      -  14.0 35536s
 248868 115266 -177.47194   42    -          - -227.12764      -  14.0 35641s
 249047 115379 -195.83025   45    2          - -227.11739      -  14.0 35702s
 249119 115407 -182.85185   47    -          - -227.11584      -  14.0 35736s
 249296 115520 -188.05400   44    4          - -227.10054      -  14.0 35820s
 249345 115543 -207.49049   44    5          - -227.09919      -  14.0 35833s
 249422 115579 -163.66442   46    -          - -227.09744      -  14.0 35870s
 249554 115676 -190.14950   48    -          - -227.09425      -  14.0 35912s
 249628 115707 -173.37157   36   16          - -227.09149      -  14.0 35948s
 249756 115760 -192.29003   47    4          - -227.07852      -  14.0 35991s
 249823 115832 -195.81548   49    2          - -227.07547      -  14.0 36006s
 250091 116001 -223.09294   43    -          - -227.06492      -  14.0 36105s
 250148 116023 -189.25850   41    9          - -227.06439      -  14.0 36119s
 250328 116137 -213.80401   44    2          - -227.05632      -  14.0 36161s
 250399 116167 -192.93869   46    6          - -227.05616      -  14.0 36184s
 250452 116188 -204.80748   49    -          - -227.05398      -  14.0 36218s
 250515 116210 -191.91364   49    -          - -227.03980      -  14.0 36244s
 250568 116291 -195.27293   45    6          - -227.03518      -  14.0 36263s
 250629 116309 -196.02157   48    -          - -227.03451      -  14.0 36283s
 250825 116435 -192.93359   43    -          - -227.02228      -  14.0 36378s
 250867 116454 -199.19033   33   29          - -227.02067      -  14.0 36388s
 251003 116539 -191.62282   42    -          - -227.01661      -  14.0 36424s
 251058 116560 -198.01853   40   10          - -227.01521      -  14.0 36448s
 251258 116712 -198.39420   41    3          - -227.00471      -  14.0 36551s
 251494 116870 -211.06560   36   24          - -226.98834      -  14.0 36634s
 251756 117004 -198.54104   45   11          - -226.98148      -  14.1 36728s
 252019 117164 -199.53460   38   29          - -226.96286      -  14.1 36854s
 252067 117181 -206.58079   45    2          - -226.96089      -  14.1 36865s
 252174 117225 -174.92859   44    -          - -226.95771      -  14.1 36906s
 252227 117282 -201.06635   38    8          - -226.95560      -  14.1 36929s
 252279 117303 -183.98952   48    -          - -226.95401      -  14.1 36949s
 252465 117413 -217.84474   38   24          - -226.94171      -  14.1 37032s
 252524 117439 -184.64384   43    5          - -226.93871      -  14.1 37049s
 252710 117594 -185.47960   39   18          - -226.92155      -  14.1 37133s
 252771 117615 -205.75383   41   22          - -226.91959      -  14.1 37147s
 252896 117659 -212.35455   38   19          - -226.91167      -  14.1 37188s
 252968 117711 -194.05791   48    5          - -226.91053      -  14.1 37210s
 253199 117862 -170.94449   47    -          - -226.90431      -  14.1 37253s
 253463 118038 -205.02705   42   19          - -226.89314      -  14.1 37350s
 253674 118119 -211.20629   32   20          - -226.88660      -  14.1 37395s
 253743 118190 -197.59281   38    8          - -226.88515      -  14.1 37415s
 253804 118214 -195.06610   48    6          - -226.88424      -  14.1 37428s
 253987 118320 -179.72067   50    -          - -226.87755      -  14.1 37512s
 254242 118487 -210.28602   43   11          - -226.85674      -  14.1 37651s
 254427 118557 -197.71583   52    -          - -226.84116      -  14.1 37716s
 254489 118634 -203.81685   43   13          - -226.84096      -  14.1 37746s
 254540 118650 -202.97763   43    -          - -226.83917      -  14.1 37756s
 254601 118681 -184.83841   44    5          - -226.83761      -  14.1 37770s
 254708 118770 -196.89095   46    2          - -226.83318      -  14.1 37810s
 254821 118816 -204.06811   43   13          - -226.82822      -  14.1 37869s
 254952 118916 -184.68358   41   10          - -226.81185      -  14.1 37912s
 255074 118964 -182.57566   48    4          - -226.80914      -  14.1 37947s
 255192 119081 -199.35332   39   14          - -226.80130      -  14.1 37984s
 255257 119108 -189.83881   41   11          - -226.80054      -  14.1 38001s
 255322 119136 -203.11542   41   10          - -226.79909      -  14.1 38028s
 255460 119192 -179.70610   45    -          - -226.79237      -  14.1 38071s
 255656 119327 -204.14540   43    6          - -226.78663      -  14.1 38142s
 255728 119359 -211.82217   33   28          - -226.78632      -  14.1 38163s
 255800 119386 -177.78825   50    -          - -226.78606      -  14.1 38177s
 255931 119496 -199.94131   46    3          - -226.77953      -  14.1 38231s
 256192 119648 -193.08620   46    -          - -226.76015      -  14.1 38333s
 256328 119704 -184.79465   45    -          - -226.75896      -  14.1 38384s
 256472 119814 -204.05356   48    -          - -226.75344      -  14.1 38418s
 256532 119841 -188.31887   48    6          - -226.75281      -  14.1 38439s
 256688 119940 -210.63160   42    3          - -226.74387      -  14.1 38492s
 256825 120002 -185.29357   42    -          - -226.74099      -  14.1 38530s
 256952 120107 -177.10064   45    5          - -226.73302      -  14.1 38574s
 257016 120128 -176.98101   49    -          - -226.72857      -  14.1 38591s
 257203 120239 -195.81998   50    -          - -226.72044      -  14.1 38649s
 257402 120370 -208.64742   39    4          - -226.70333      -  14.1 38731s
 257538 120428 -196.86158   37   29          - -226.70105      -  14.1 38779s
 257678 120528 -171.66967   46    -          - -226.69511      -  14.1 38833s
 257929 120685 -204.08497   42    5          - -226.68256      -  14.1 38926s
 258201 120840 -208.91051   39    9          - -226.66328      -  14.1 39033s
 258245 120853 -212.77009   41   23          - -226.66328      -  14.1 39050s
 258344 120893 -185.68482   50    2          - -226.65914      -  14.1 39078s
 258403 120966 -200.80619   42    9          - -226.65738      -  14.1 39101s
 258469 120990 -188.91989   52    -          - -226.65684      -  14.1 39112s
 258657 121117 -211.65877   39    6          - -226.64142      -  14.1 39176s
 258921 121287 -208.92585   38   13          - -226.63457      -  14.1 39251s
 258980 121302 -179.55850   46    3          - -226.63244      -  14.1 39269s
 259036 121323 -193.51709   50    -          - -226.63125      -  14.1 39301s
 259159 121392 -190.22008   46    3          - -226.62371      -  14.1 39359s
 259228 121424 -211.54153   39   11          - -226.62008      -  14.1 39377s
 259396 121571 -197.89671   43    4          - -226.61193      -  14.1 39466s
 259651 121721 -208.35181   36   13          - -226.59876      -  14.1 39553s
 259774 121768 -202.64413   37   11          - -226.59272      -  14.1 39606s
 259841 121793 -193.18335   49    2          - -226.58461      -  14.1 39642s
 259910 121854 -196.88009   45    6          - -226.58188      -  14.1 39672s
 260023 121907 -207.82436   47    -          - -226.57792      -  14.1 39731s
 260138 121981 -175.98900   48    -          - -226.56466      -  14.1 39789s
 260324 122064 -185.65515   43    4          - -226.54482      -  14.1 39853s
 260385 122153 -203.57997   40    3          - -226.54470      -  14.1 39876s
 260644 122285 -183.18801   43    2          - -226.53053      -  14.1 39956s
 260692 122303 -176.30493   43    6          - -226.53001      -  14.1 39967s
 260753 122329 -160.77727   39    -          - -226.52933      -  14.1 39992s
 260868 122433 -196.11153   45    -          - -226.52629      -  14.1 40019s
 260925 122457 -182.37281   39    5          - -226.52609      -  14.1 40029s
 261099 122548 -181.25451   46    5          - -226.51983      -  14.1 40088s
 261160 122573 -190.81760   44    8          - -226.51797      -  14.1 40109s
 261363 122739 -179.32120   49    -          - -226.50663      -  14.1 40189s
 261416 122759 -217.86357   43   24          - -226.50575      -  14.1 40210s
 261615 122875 -196.21111   42   11          - -226.49115      -  14.1 40273s
 261712 122914 -193.50430   42   11          - -226.48832      -  14.1 40315s
 261811 122973 -178.73596   41    -          - -226.48009      -  14.1 40359s
 262014 123063 -209.33902   45    4          - -226.46162      -  14.1 40448s
 262084 123155 -216.85464   37   11          - -226.45979      -  14.1 40476s
 262145 123179 -163.36521   50    -          - -226.45773      -  14.1 40508s
 262319 123280 -188.01684   44    -          - -226.44352      -  14.1 40580s
 262561 123431 -208.95990   35   11          - -226.43413      -  14.1 40675s
 262659 123467 -187.40081   48    2          - -226.43236      -  14.1 40700s
 262759 123578 -177.87107   49    -          - -226.42723      -  14.1 40741s
 263017 123726 -199.77139   39    5          - -226.41219      -  14.1 40819s
 263086 123754 -187.57046   50    3          - -226.41207      -  14.1 40836s
 263153 123782 -199.04430   46    4          - -226.41039      -  14.1 40861s
 263228 123812 -174.83603   49    -          - -226.40426      -  14.1 40900s
 263298 123869 -189.81136   42    5          - -226.39738      -  14.1 40931s
 263342 123883 -170.17862   43    5          - -226.39642      -  14.1 40949s
 263400 123908 -172.48441   43    -          - -226.39456      -  14.1 40996s
 263512 124005 -189.13975   37    5          - -226.38794      -  14.1 41046s
 263644 124060 -178.64406   45    -          - -226.38303      -  14.1 41070s
 263784 124173 -193.15117   45    -          - -226.38073      -  14.1 41148s
 264034 124330 -208.23294   41   15          - -226.36453      -  14.1 41251s
 264247 124451 -173.84381   43   11          - -226.35625      -  14.1 41332s
 264475 124599 -202.69735   40   19          - -226.34159      -  14.1 41413s
 264550 124628 -201.04949   39   14          - -226.34083      -  14.1 41441s
 264682 124680 -187.66151   49    -          - -226.33287      -  14.1 41514s
 264748 124741 -180.29191   49    -          - -226.32969      -  14.1 41547s
 264798 124761 -187.29589   48    2          - -226.32725      -  14.1 41557s
 264968 124886 -187.13288   48    6          - -226.32092      -  14.1 41627s
 265201 125029 -202.47686   49    -          - -226.30893      -  14.1 41739s
 265452 125178 -193.32245   46    -          - -226.28766      -  14.1 41842s
 265520 125203 -192.64309   46    3          - -226.28553      -  14.1 41856s
 265704 125286 -198.52894   49    -          - -226.27206      -  14.1 41967s
 265854 125359 -197.17492   42    -          - -226.26053      -  14.1 42071s
 265912 125425 -204.56852   42    3          - -226.25971      -  14.1 42088s
 265974 125454 -186.35805   39   14          - -226.25812      -  14.2 42138s
 266169 125567 -202.28168   47    -          - -226.24431      -  14.2 42232s
 266297 125620 -185.72679   49    -          - -226.24375      -  14.2 42293s
 266408 125705 -204.21197   38   25          - -226.23705      -  14.2 42339s
 266619 125794 -202.71597   50    4          - -226.21685      -  14.2 42419s
 266693 125880 -191.80975   47    -          - -226.21605      -  14.2 42455s
 266790 125921 -187.62444   42    -          - -226.21320      -  14.2 42504s
 266879 125986 -183.64697   47    4          - -226.20408      -  14.2 42537s
 267006 126042 -188.16531   43    7          - -226.20049      -  14.2 42599s
 267147 126158 -181.60569   47    6          - -226.18900      -  14.2 42677s
 267412 126321 -191.61909   44    -          - -226.17581      -  14.2 42761s
 267474 126345 -194.30513   45    6          - -226.17550      -  14.2 42775s
 267539 126363 -203.25466   39   11          - -226.17540      -  14.2 42787s
 267611 126399 -203.39166   41   11          - -226.16899      -  14.2 42824s
 267667 126458 -184.91159   52    -          - -226.16803      -  14.2 42878s
 267734 126485 -206.91933   45   14          - -226.16733      -  14.2 42900s
 267910 126620 -201.24509   41    8          - -226.15801      -  14.2 42985s
 267955 126636 -206.79420   37   21          - -226.15734      -  14.2 43000s
 268053 126669 -194.34043   41   19          - -226.15081      -  14.2 43037s
 268106 126733 -183.95472   43    6          - -226.14857      -  14.2 43057s
 268361 126905 -204.78095   39    8          - -226.12725      -  14.2 43155s
 268416 126918 -207.16595   40   16          - -226.12721      -  14.2 43163s
 268610 127035 -185.02252   44    -          - -226.12481      -  14.2 43234s
 268679 127069 -215.92673   36   28          - -226.12306      -  14.2 43245s
 268803 127117 -174.25779   43    -          - -226.12084      -  14.2 43296s
 268871 127216 -171.24151   40    -          - -226.11735      -  14.2 43325s
 268979 127255 -194.91979   50    2          - -226.11347      -  14.2 43362s
 269105 127342 -179.23240   45    -          - -226.10137      -  14.2 43405s
 269295 127458 -192.27171   40    9          - -226.09519      -  14.2 43495s
 269357 127481 -185.39398   43    -          - -226.09434      -  14.2 43514s
 269562 127602 -172.20428   46    -          - -226.08254      -  14.2 43634s
 269677 127649 -189.84603   46    -          - -226.07833      -  14.2 43697s
 269792 127767 -186.41609   42    5          - -226.06225      -  14.2 43762s
 269922 127813 -183.53983   44    2          - -226.06068      -  14.2 43795s
 270054 127911 -197.48769   48    5          - -226.05025      -  14.2 43849s
 270121 127935 -189.35503   45    -          - -226.04932      -  14.2 43882s
 270182 127964 -184.39979   40   10          - -226.04883      -  14.2 43898s
 270290 128058 -205.47338   34   30          - -226.03564      -  14.2 43955s
 270333 128070 -173.72836   48    8          - -226.03541      -  14.2 43963s
 270472 128162 -191.69355   48    -          - -226.02851      -  14.2 44010s
 270712 128301 -197.97189   41    3          - -226.02130      -  14.2 44122s
 270775 128329 -178.92728   49    4          - -226.01968      -  14.2 44174s
 270837 128355 -188.13890   52    5          - -226.01588      -  14.2 44211s
 270969 128464 -208.18257   36   11          - -225.99726      -  14.2 44261s
 271040 128486 -210.68245   48    5          - -225.99435      -  14.2 44276s
 271100 128515 -167.62551   45    2          - -225.99303      -  14.2 44313s
 271223 128586 -189.04213   46    -          - -225.98422      -  14.2 44357s
 271491 128747 -195.30594   33   19          - -225.97496      -  14.2 44434s
 271529 128757 -186.59114   49    -          - -225.97365      -  14.2 44446s
 271685 128873 -192.57709   45    4          - -225.96866      -  14.2 44530s
 271953 129031 -204.25985   36   13          - -225.95317      -  14.2 44666s
 272021 129060 -187.37557   51    -          - -225.94987      -  14.2 44692s
 272076 129077 -174.96691   52    -          - -225.94718      -  14.2 44715s
 272197 129164 -190.29930   49    -          - -225.93255      -  14.2 44777s
 272268 129194 -208.38644   42   12          - -225.93140      -  14.2 44803s
 272458 129330 -201.84079   41   14          - -225.91755      -  14.2 44906s
 272514 129350 -205.53532   35   18          - -225.91120      -  14.2 44917s
 272677 129455 -196.73544   41    -          - -225.90022      -  14.2 44979s
 272935 129601 -165.91773   42    4          - -225.88575      -  14.2 45100s
 272993 129623 -189.31977   46   10          - -225.88558      -  14.2 45126s
 273187 129742 -196.15176   42    4          - -225.86956      -  14.2 45226s
 273278 129774 -203.74327   42   13          - -225.86748      -  14.2 45252s
 273367 129889 -182.96973   44    7          - -225.86337      -  14.2 45284s
 273439 129920 -189.44150   47    2          - -225.86077      -  14.2 45303s
 273651 130025 -178.59536   50    -          - -225.85401      -  14.2 45372s
 273906 130212 -200.16440   45    5          - -225.83884      -  14.2 45529s
 273961 130234 -200.17920   40    9          - -225.83638      -  14.2 45558s
 274155 130309 -168.95813   53    -          - -225.82686      -  14.2 45630s
 274267 130361 -178.67559   46    -          - -225.82606      -  14.2 45711s
 274373 130456 -188.65565   44    3          - -225.81484      -  14.2 45767s
 274441 130486 -173.28065   42    -          - -225.81213      -  14.2 45823s
 274626 130614 -188.65016   53    -          - -225.79930      -  14.2 45891s
 274692 130641 -175.28485   40    -          - -225.79926      -  14.2 45910s
 274889 130745 -180.66516   47    4          - -225.78844      -  14.2 45977s
 274943 130769 -191.60678   37   15          - -225.78796      -  14.2 46000s
 275056 130820 -178.31424   45    -          - -225.78435      -  14.2 46038s
 275101 130891 -195.96412   43   13          - -225.78126      -  14.2 46052s
 275350 131032 -173.58327   47    -          - -225.77189      -  14.2 46146s
 275458 131079 -192.19395   41   13          - -225.77087      -  14.2 46202s
 275581 131165 -159.11007   47    -          - -225.76134      -  14.2 46247s
 275637 131186 -187.00523   47    -          - -225.76069      -  14.2 46289s
 275836 131322 -190.50835   44    -          - -225.75181      -  14.2 46418s
 275885 131338 -191.48484   50    -          - -225.75125      -  14.2 46441s
 276057 131488 -189.40079   46    -          - -225.73418      -  14.2 46556s
 276307 131609 -167.37725   46    -          - -225.72561      -  14.2 46664s
 276435 131666 -188.19115   50    -          - -225.72424      -  14.2 46735s
 276565 131761 -194.39603   50    -          - -225.71395      -  14.2 46799s
 276607 131774 -194.80779   46    -          - -225.71289      -  14.2 46814s
 276762 131887 -190.36093   45    -          - -225.70292      -  14.2 46866s
 276821 131910 -179.39724   40    4          - -225.70199      -  14.2 46878s
 277002 132030 -195.07468   38    7          - -225.69235      -  14.2 46968s
 277062 132060 -208.39951   36   19          - -225.69233      -  14.2 46995s
 277253 132199 -180.07430   52    -          - -225.68449      -  14.2 47060s
 277381 132251 -198.61117   45    5          - -225.68186      -  14.2 47126s
 277494 132318 -181.67100   43    -          - -225.67403      -  14.2 47163s
 277551 132345 -188.21872   48    -          - -225.67256      -  14.2 47191s
 277702 132430 -201.86672   46    2          - -225.66002      -  14.2 47272s
 277764 132456 -198.01643   43    4          - -225.65956      -  14.2 47288s
 277966 132606 -207.28432   34   21          - -225.64983      -  14.2 47390s
 278029 132626 -191.59820   50    -          - -225.64834      -  14.2 47431s
 278214 132741 -188.43879   51    -          - -225.63660      -  14.2 47509s
 278421 132876 -194.10896   49    -          - -225.62311      -  14.2 47627s
 278656 132998 -200.70964   45   13          - -225.61502      -  14.2 47710s
 278714 133017 -184.76000   48    5          - -225.61410      -  14.2 47731s
 278888 133133 -204.49200   46    -          - -225.60223      -  14.2 47863s
 279079 133215 -168.92552   48    -          - -225.58466      -  14.2 47965s
 279138 133294 -201.68427   47    4          - -225.58320      -  14.2 48001s
 279195 133315 -197.35946   50    4          - -225.58279      -  14.2 48017s
 279375 133417 -173.37898   44    2          - -225.57309      -  14.2 48099s
 279434 133442 -202.76324   50    3          - -225.57075      -  14.2 48132s
 279612 133548 -208.81849   48    8          - -225.55888      -  14.2 48239s
 279667 133573 -204.10467   45   11          - -225.55620      -  14.2 48260s
 279722 133594 -191.05993   46    -          - -225.55432      -  14.3 48301s
 279826 133689 -210.63946   45    3          - -225.54073      -  14.3 48360s
 280078 133799 -197.24258   43    6          - -225.53140      -  14.3 48495s
 280265 133884 -208.80693   47    -          - -225.51146      -  14.3 48627s
 280328 133944 -177.85993   42    3          - -225.50623      -  14.3 48692s
 280375 133964 -180.63326   45    3          - -225.50571      -  14.3 48725s
 280518 134088 -209.26365   46    -          - -225.49571      -  14.3 48803s
 280589 134117 -191.45412   45    -          - -225.49383      -  14.3 48844s
 280647 134136 -216.39072   39   22          - -225.49379      -  14.3 48873s
 280783 134233 -178.09148   48    2          - -225.48486      -  14.3 48938s
 280831 134252 -181.90350   36    3          - -225.48133      -  14.3 48963s
 280948 134304 -178.07364   52    -          - -225.47769      -  14.3 49016s
 281008 134371 -190.72461   50    -          - -225.47749      -  14.3 49044s
 281205 134451 -173.29988   44    -          - -225.46372      -  14.3 49126s
 281273 134498 -186.37608   43    8          - -225.46347      -  14.3 49150s
 281333 134526 -194.67066   45    6          - -225.46214      -  14.3 49166s
 281513 134654 -181.68614   50    -          - -225.44868      -  14.3 49261s
 281613 134685 -202.38792   45    -          - -225.44602      -  14.3 49307s
 281668 134713 -193.42833   47    6          - -225.43721      -  14.3 49353s
 281728 134798 -183.40908   47    2          - -225.43653      -  14.3 49372s
 281786 134813 -194.21209   47    6          - -225.43399      -  14.3 49393s
 281977 134952 -208.45925   48    -          - -225.42639      -  14.3 49496s
 282240 135075 -204.72916   41   11          - -225.41563      -  14.3 49614s
 282499 135237 -186.58580   44   10          - -225.39770      -  14.3 49770s
 282546 135261 -185.68522   43    5          - -225.39600      -  14.3 49791s
 282592 135278 -192.84152   38   12          - -225.39549      -  14.3 49811s
 282688 135378 -220.29915   41   18          - -225.38683      -  14.3 49852s
 282750 135398 -201.76194   39   11          - -225.38664      -  14.3 49873s
 282881 135443 -190.77258   53    -          - -225.38231      -  14.3 49933s
 282950 135510 -187.75725   46    -          - -225.38104      -  14.3 49987s
 283073 135558 -191.51335   49    -          - -225.37695      -  14.3 50032s
 283185 135652 -199.86137   42    -          - -225.36155      -  14.3 50141s
 283235 135668 -186.05336   45    3          - -225.36011      -  14.3 50153s
 283415 135780 -187.95805   40    4          - -225.35116      -  14.3 50284s
 283613 135902 -190.42239   39    2          - -225.33766      -  14.3 50375s
 283673 135925 -196.16479   40    5          - -225.33743      -  14.3 50396s
 283793 135975 -204.46914   47    -          - -225.33328      -  14.3 50463s
 283857 136070 -193.07547   45    4          - -225.33146      -  14.3 50501s
 284113 136224 -180.47961   41    2          - -225.31843      -  14.3 50593s
 284241 136270 -206.13560   43   14          - -225.31801      -  14.3 50660s
 284358 136364 -181.44541   43    6          - -225.30502      -  14.3 50718s
 284409 136379 -206.39651   45    2          - -225.30341      -  14.3 50731s
 284512 136416 -181.27828   45    -          - -225.29676      -  14.3 50774s
 284563 136476 -195.52589   50    -          - -225.29497      -  14.3 50811s
 284619 136501 -200.15403   44    2          - -225.29417      -  14.3 50833s
 284798 136609 -209.03334   42   12          - -225.28275      -  14.3 50952s
 285052 136750 -191.82483   41   15          - -225.26827      -  14.3 51089s
 285094 136767 -200.61441   48    -          - -225.26801      -  14.3 51115s
 285145 136791 -197.17594   40   14          - -225.26710      -  14.3 51128s
 285195 136808 -204.43181   38   10          - -225.26270      -  14.3 51154s
 285246 136884 -182.47638   38    8          - -225.26243      -  14.3 51178s
 285307 136908 -181.86866   42    -          - -225.26008      -  14.3 51213s
 285500 137038 -198.74394   39   25          - -225.25142      -  14.3 51295s
 285670 137099 -164.62929   46    5          - -225.24485      -  14.3 51367s
 285737 137159 -186.28560   38    8          - -225.24288      -  14.3 51414s
 285797 137183 -178.80823   42    8          - -225.23990      -  14.3 51445s
 285977 137322 -190.21357   49    -          - -225.23321      -  14.3 51537s
 286102 137371 -196.56272   44    -          - -225.23280      -  14.3 51589s
 286220 137450 -184.88326   42   13          - -225.22074      -  14.3 51640s
 286355 137501 -202.24669   39   10          - -225.21063      -  14.3 51727s
 286398 137579 -194.35301   41    9          - -225.20831      -  14.3 51745s
 286642 137729 -199.46080   43    9          - -225.19648      -  14.3 51855s
 286883 137892 -189.74674   35    6          - -225.18492      -  14.3 51942s
 286941 137907 -203.96954   37   29          - -225.18285      -  14.3 51959s
 286999 137930 -203.70138   50    -          - -225.18244      -  14.3 51991s
 287063 137957 -211.28373   38   14          - -225.18149      -  14.3 52051s
 287118 138020 -197.25976   48    2          - -225.17911      -  14.3 52070s
 287169 138038 -194.14226   45    6          - -225.17819      -  14.3 52103s
 287274 138083 -207.28763   45   15          - -225.16551      -  14.3 52172s
 287332 138143 -171.77962   38    5          - -225.16125      -  14.3 52203s
 287452 138193 -193.92516   47    5          - -225.15842      -  14.3 52280s
 287577 138297 -185.64817   40   10          - -225.14636      -  14.3 52351s
 287645 138320 -204.41045   48    3          - -225.14526      -  14.3 52379s
 287719 138351 -199.13973   47    3          - -225.14488      -  14.3 52403s
 287782 138377 -191.33735   47    2          - -225.14217      -  14.3 52427s
 287846 138455 -195.76381   43    5          - -225.13771      -  14.3 52463s
 287895 138468 -186.48543   53    -          - -225.12950      -  14.3 52480s
 288019 138551 -180.48041   43    -          - -225.12580      -  14.3 52525s
 288285 138706 -205.27586   37   15          - -225.11190      -  14.3 52663s
 288338 138724 -195.19755   48    3          - -225.11173      -  14.3 52679s
 288406 138758 -199.13620   43    7          - -225.11092      -  14.3 52725s
 288526 138850 -169.17250   43    -          - -225.09642      -  14.3 52793s
 288763 138977 -177.50107   39    4          - -225.08807      -  14.3 52906s
 288806 138992 -192.12486   46    8          - -225.08634      -  14.3 52922s
 288895 139027 -194.13272   44    -          - -225.07294      -  14.3 52967s
 288947 139109 -190.95386   41   15          - -225.07215      -  14.3 52986s
 289013 139138 -197.94917   43   11          - -225.07171      -  14.3 53015s
 289206 139274 -196.38996   41    8          - -225.06485      -  14.3 53123s
 289457 139398 -178.66109   49    -          - -225.05237      -  14.3 53239s
 289509 139418 -210.63764   43    9          - -225.05121      -  14.3 53272s
 289564 139444 -187.46482   48    -          - -225.04930      -  14.3 53293s
 289693 139546 -206.57748   45    6          - -225.04260      -  14.3 53336s
 289751 139564 -207.00042   42    6          - -225.04116      -  14.3 53353s
 289946 139677 -197.35083   43   10          - -225.03526      -  14.3 53480s
 290173 139827 -185.06320   45    4          - -225.01791      -  14.3 53640s
 290215 139841 -176.17909   47    -          - -225.01682      -  14.3 53657s
 290357 139929 -188.32148   43    -          - -225.01071      -  14.3 53717s
 290412 139950 -193.44714   49    -          - -225.01066      -  14.3 53746s
 290606 140087 -180.95259   46    2          - -225.00238      -  14.3 53806s
 290674 140115 -205.67579   42   20          - -224.99780      -  14.3 53831s
 290731 140138 -198.18236   42    4          - -224.99779      -  14.3 53848s
 290843 140232 -176.84201   50    -          - -224.98812      -  14.3 53896s
 290892 140250 -191.75562   47    -          - -224.98673      -  14.3 53937s
 291071 140368 -187.48198   43    5          - -224.97143      -  14.3 54022s
 291121 140385 -196.01001   41    2          - -224.97092      -  14.3 54048s
 291269 140474 -188.48792   43    -          - -224.96511      -  14.3 54125s
 291425 140546 -205.11630   39   23          - -224.95635      -  14.3 54179s
 291484 140620 -191.17772   41    5          - -224.95569      -  14.3 54203s
 291538 140640 -182.40313   40    6          - -224.95481      -  14.3 54232s
 291669 140690 -214.34241   44    7          - -224.94469      -  14.3 54269s
 291728 140739 -192.50741   47    3          - -224.94318      -  14.4 54305s
 291818 140783 -208.02901   41   20          - -224.94158      -  14.4 54347s
 291866 140801 -184.57387   45    3          - -224.93591      -  14.4 54360s
 291910 140864 -163.72854   41    -          - -224.93568      -  14.4 54384s
 292150 140998 -213.79313   41   22          - -224.91989      -  14.4 54496s
 292219 141026 -193.31854   38    -          - -224.91952      -  14.4 54517s
 292279 141053 -162.00718   48    -          - -224.91922      -  14.4 54562s
 292347 141080 -177.00275   47    -          - -224.91585      -  14.4 54583s
 292411 141172 -203.92470   46    -          - -224.91217      -  14.4 54619s
 292533 141219 -189.59112   40    9          - -224.90878      -  14.4 54674s
 292665 141324 -210.79622   47    5          - -224.90154      -  14.4 54731s
 292724 141347 -195.44490   52    -          - -224.90069      -  14.4 54765s
 292877 141469 -188.59892   48    -          - -224.89661      -  14.4 54827s
 293124 141587 -196.93741   44    -          - -224.88164      -  14.4 54943s
 293325 141728 -204.30394   40    6          - -224.87233      -  14.4 55030s
 293446 141775 -187.95366   38    6          - -224.87070      -  14.4 55081s
 293513 141802 -183.33870   50    -          - -224.85981      -  14.4 55102s
 293572 141863 -192.87027   50    2          - -224.85793      -  14.4 55125s
 293636 141888 -176.94384   40    -          - -224.85770      -  14.4 55155s
 293817 141987 -190.25953   51    -          - -224.85168      -  14.4 55247s
 293920 142028 -183.17192   45    -          - -224.84927      -  14.4 55305s
 294002 142118 -200.16687   48    6          - -224.83663      -  14.4 55353s
 294056 142142 -181.49937   43    -          - -224.83491      -  14.4 55387s
 294237 142231 -205.67964   41   10          - -224.81344      -  14.4 55453s
 294425 142315 -167.59292   47    -          - -224.80608      -  14.4 55573s
 294491 142394 -171.03392   44    -          - -224.80304      -  14.4 55609s
 294558 142421 -207.95944   44   10          - -224.80039      -  14.4 55635s
 294691 142477 -198.64446   42   15          - -224.79238      -  14.4 55689s
 294757 142518 -213.63280   38   23          - -224.79217      -  14.4 55717s
 294977 142664 -192.01582   36   16          - -224.78668      -  14.4 55835s
 295160 142741 -173.47457   43    -          - -224.77324      -  14.4 55934s
 295230 142815 -208.68527   37   16          - -224.77301      -  14.4 55954s
 295339 142858 -196.30391   49    -          - -224.77150      -  14.4 55988s
 295456 142959 -180.89745   46    4          - -224.76652      -  14.4 56029s
 295706 143099 -173.47996   44    9          - -224.75293      -  14.4 56159s
 295753 143116 -187.90856   46    -          - -224.75287      -  14.4 56180s
 295934 143221 -190.05510   51    -          - -224.74600      -  14.4 56255s
 296123 143357 -191.94617   46    4          - -224.73429      -  14.4 56339s
 296363 143497 -184.78879   46    3          - -224.72051      -  14.4 56452s
 296609 143632 -195.04498   54    -          - -224.71127      -  14.4 56593s
 296734 143687 -205.05388   44   11          - -224.70828      -  14.4 56644s
 296847 143775 -186.24873   48    4          - -224.69613      -  14.4 56702s
 297037 143920 -192.42436   48    5          - -224.68163      -  14.4 56791s
 297094 143938 -193.61031   52    -          - -224.68074      -  14.4 56808s
 297302 144042 -197.19498   45    -          - -224.67568      -  14.4 56888s
 297408 144086 -186.36205   53    -          - -224.67241      -  14.4 56930s
 297515 144159 -209.19491   38    7          - -224.66374      -  14.4 57002s
 297587 144189 -182.25934   48    -          - -224.66302      -  14.4 57024s
 297778 144334 -207.53004   42    9          - -224.64794      -  14.4 57131s
 297945 144397 -208.35752   48    -          - -224.64585      -  14.4 57195s
 297991 144463 -192.59361   43   16          - -224.64159      -  14.4 57215s
 298044 144485 -215.00183   44    8          - -224.64136      -  14.4 57241s
 298229 144596 -196.08549   49    4          - -224.63006      -  14.4 57334s
 298279 144617 -179.85048   48    3          - -224.62851      -  14.4 57356s
 298428 144748 -184.46667   46    -          - -224.61678      -  14.4 57423s
 298706 144885 -178.76552   43    -          - -224.60790      -  14.4 57530s
 298924 145012 -193.54005   46    9          - -224.60248      -  14.4 57623s
 299179 145169 -215.20726   38   17          - -224.59243      -  14.4 57754s
 299226 145185 -168.44822   45    -          - -224.58806      -  14.4 57784s
 299352 145240 -206.51072   38   22          - -224.58032      -  14.4 57840s
 299404 145299 -214.09210   43   13          - -224.58017      -  14.4 57865s
 299563 145362 -202.08380   41    8          - -224.57511      -  14.4 57953s
 299605 145408 -194.69540   48    2          - -224.57258      -  14.4 57982s
 299845 145545 -174.06898   40    7          - -224.54740      -  14.4 58145s
 299897 145567 -195.58800   45    4          - -224.54482      -  14.4 58171s
 300075 145685 -200.01717   46    2          - -224.53774      -  14.4 58281s
 300255 145757 -176.39962   46    4          - -224.52361      -  14.4 58397s
 300320 145821 -184.72456   45    6          - -224.52000      -  14.4 58445s
 300520 145942 -204.20769   34   17          - -224.50974      -  14.4 58600s
 300579 145965 -189.14130   46    5          - -224.50347      -  14.4 58612s
 300766 146090 -174.97879   45    4          - -224.49945      -  14.4 58681s
 300827 146114 -184.62410   44    -          - -224.49733      -  14.4 58711s
 301014 146223 -195.35763   42    5          - -224.48770      -  14.4 58805s
 301203 146322 -209.05399   47   11          - -224.48130      -  14.4 58913s
 301267 146348 -203.82528   41    2          - -224.47918      -  14.4 58956s
 301404 146417 -181.03481   52    -          - -224.47516      -  14.4 59022s
 301473 146485 -206.63540   45    7          - -224.47336      -  14.4 59059s
 301703 146635 -199.40822   40   18          - -224.46412      -  14.4 59164s
 301762 146657 -190.13861   50    2          - -224.46367      -  14.4 59195s
 301957 146782 -176.60454   46    -          - -224.45164      -  14.4 59280s
 302001 146798 -212.64168   38   25          - -224.45027      -  14.4 59294s
 302125 146844 -187.66055   49    -          - -224.44297      -  14.4 59376s
 302178 146917 -203.15314   41    6          - -224.43894      -  14.4 59410s
 302239 146939 -184.68948   49    -          - -224.43852      -  14.4 59436s
 302434 147059 -181.14599   46    -          - -224.42853      -  14.4 59537s
 302587 147122 -183.64157   49    -          - -224.42089      -  14.4 59660s
 302640 147184 -176.95957   44    -          - -224.41688      -  14.4 59682s
 302822 147262 -192.67278   35   25          - -224.41077      -  14.4 59802s
 302881 147312 -177.97095   43    3          - -224.40570      -  14.4 59832s
 303087 147447 -192.07960   43    6          - -224.39221      -  14.4 59940s
 303262 147513 -184.35356   44   14          - -224.38385      -  14.4 60025s
 303321 147590 -188.20325   37   15          - -224.38316      -  14.4 60068s
 303385 147613 -190.76635   50    3          - -224.38205      -  14.4 60081s
 303446 147639 -193.48517   44    -          - -224.38115      -  14.4 60112s
 303550 147709 -184.83276   50    -          - -224.37522      -  14.4 60142s
 303740 147794 -185.56168   43    8          - -224.36263      -  14.4 60275s
 303793 147830 -197.08989   48    5          - -224.36163      -  14.4 60301s
 303849 147856 -203.65836   48    -          - -224.35915      -  14.4 60367s
 303908 147883 -171.95727   45    -          - -224.35776      -  14.4 60394s
 304023 147988 -200.05822   44    2          - -224.34844      -  14.4 60476s
 304251 148111 -168.59830   48    -          - -224.34308      -  14.4 60599s
 304483 148253 -189.05870   51    -          - -224.32990      -  14.4 60758s
 304550 148284 -191.17557   47    5          - -224.32684      -  14.4 60794s
 304724 148377 -179.17057   45    4          - -224.31788      -  14.4 60872s
 304778 148402 -190.58823   49    -          - -224.31752      -  14.4 60934s
 304911 148511 -184.05791   43    3          - -224.30805      -  14.4 60973s
 305152 148651 -185.14205   49    4          - -224.29560      -  14.4 61124s
 305207 148675 -203.14410   43    8          - -224.29432      -  14.4 61142s
 305366 148766 -182.24912   47    5          - -224.28885      -  14.4 61203s
 305433 148795 -185.30411   41   10          - -224.28786      -  14.4 61230s
 305554 148848 -190.08536   42    -          - -224.28055      -  14.4 61311s
 305613 148929 -176.22180   51    -          - -224.27687      -  14.4 61359s
 305874 149064 -197.29668   40   16          - -224.26574      -  14.5 61497s
 306046 149179 -178.61147   40    7          - -224.25858      -  14.5 61611s
 306163 149225 -176.01526   41    2          - -224.25529      -  14.5 61679s
 306287 149310 -171.14612   42    2          - -224.25183      -  14.5 61762s
 306345 149335 -180.53390   39    4          - -224.25163      -  14.5 61798s
 306510 149437 -184.66083   41    4          - -224.24145      -  14.5 61846s
 306574 149465 -170.85885   45    2          - -224.24012      -  14.5 61887s
 306695 149516 -195.74975   41    6          - -224.22707      -  14.5 61959s
 306758 149591 -175.99961   50    -          - -224.22433      -  14.5 62030s
 306820 149614 -192.12203   43    -          - -224.22385      -  14.5 62065s
 306873 149634 -196.86435   47    5          - -224.22335      -  14.5 62088s
 307004 149739 -183.54994   44    4          - -224.21022      -  14.5 62180s
 307200 149861 -203.58322   40   21          - -224.20346      -  14.5 62269s
 307266 149888 -192.57498   49    -          - -224.20334      -  14.5 62311s
 307333 149914 -183.81344   50    4          - -224.20325      -  14.5 62333s
 307476 150012 -173.41994   46    -          - -224.19678      -  14.5 62427s
 307725 150167 -172.22808   48    -          - -224.18843      -  14.5 62613s
 307780 150191 -186.01058   48    -          - -224.18666      -  14.5 62654s
 307962 150301 -174.42979   46    -          - -224.17803      -  14.5 62748s
 308158 150405 -199.60487   47    4          - -224.16966      -  14.5 62832s
 308388 150544 -202.71869   42   10          - -224.15659      -  14.5 63000s
 308592 150631 -189.83847   46    5          - -224.14492      -  14.5 63097s
 308661 150689 -206.57773   41   10          - -224.14290      -  14.5 63141s
 308783 150742 -167.09510   48    3          - -224.14146      -  14.5 63228s
 308916 150853 -202.65349   40    7          - -224.12771      -  14.5 63313s
 308968 150869 -186.23157   52    5          - -224.12697      -  14.5 63336s
 309122 150973 -184.48262   42    5          - -224.12177      -  14.5 63453s
 309294 151040 -190.21595   49    -          - -224.11474      -  14.5 63530s
 309352 151113 -193.42166   46    6          - -224.11289      -  14.5 63588s
 309413 151134 -192.37782   38    9          - -224.11206      -  14.5 63611s
 309526 151183 -190.92373   45    9          - -224.10505      -  14.5 63666s
 309590 151232 -188.02678   44    3          - -224.10067      -  14.5 63725s
 309757 151353 -185.05338   44    6          - -224.09623      -  14.5 63801s
 309823 151378 -182.99836   41    3          - -224.09495      -  14.5 63843s
 310000 151511 -180.42126   48    5          - -224.08463      -  14.5 63946s
 310052 151526 -191.82948   51    5          - -224.08367      -  14.5 63960s
 310237 151658 -207.42643   39   10          - -224.07486      -  14.5 64045s
 310302 151680 -196.76364   46    9          - -224.07392      -  14.5 64072s
 310484 151784 -190.50587   51    -          - -224.06042      -  14.5 64158s
 310672 151888 -166.77861   45    -          - -224.05080      -  14.5 64298s
 310788 151937 -202.12723   40   14          - -224.03311      -  14.5 64345s
 310894 152028 -177.82343   44    6          - -224.02234      -  14.5 64454s
 311140 152169 -175.15355   48    5          - -224.01082      -  14.5 64572s
 311391 152320 -199.94328   46    -          - -223.99600      -  14.5 64751s
 311448 152343 -193.57471   46    -          - -223.99552      -  14.5 64793s
 311632 152455 -198.15704   42   10          - -223.98965      -  14.5 64875s
 311811 152569 -179.91694   41    4          - -223.98429      -  14.5 64981s
 311927 152615 -201.18293   48    -          - -223.98268      -  14.5 65036s
 311989 152643 -179.61978   45    3          - -223.97507      -  14.5 65051s
 312050 152695 -184.82911   49    -          - -223.97475      -  14.5 65082s
 312254 152789 -201.95592   41   16          - -223.96886      -  14.5 65162s
 312328 152854 -190.12411   48    -          - -223.96843      -  14.5 65203s
 312381 152877 -179.13452   48    -          - -223.96769      -  14.5 65221s
 312447 152909 -164.80579   48    -          - -223.96748      -  14.5 65245s
 312556 152981 -199.79195   37   15          - -223.96235      -  14.5 65318s
 312760 153093 -193.73727   42    8          - -223.95126      -  14.5 65436s
 312815 153115 -160.05062   42    6          - -223.94998      -  14.5 65474s
 312997 153225 -198.83693   48    3          - -223.93666      -  14.5 65588s
 313058 153251 -193.02026   42    -          - -223.93598      -  14.5 65616s
 313166 153300 -175.48149   50    -          - -223.92354      -  14.5 65704s
 313209 153364 -188.94299   47    -          - -223.92143      -  14.5 65744s
 313388 153433 -173.36914   46    -          - -223.90775      -  14.5 65851s
 313451 153495 -194.56083   49    2          - -223.90577      -  14.5 65887s
 313497 153515 -182.79034   48    -          - -223.90486      -  14.5 65905s
 313650 153633 -207.08215   37   19          - -223.89566      -  14.5 65992s
 313701 153651 -188.37576   43    -          - -223.89517      -  14.5 66021s
 313766 153679 -188.37101   44    -          - -223.89404      -  14.5 66035s
 313895 153768 -203.86939   45    2          - -223.88950      -  14.5 66108s
 314091 153927 -199.58059   39   11          - -223.87559      -  14.5 66216s
 314217 153967 -197.79000   47    3          - -223.87491      -  14.5 66258s
 314288 153991 -180.98640   40    -          - -223.87089      -  14.5 66296s
 314346 154051 -187.73610   42    6          - -223.87049      -  14.5 66322s
 314583 154178 -193.73339   42    5          - -223.85191      -  14.5 66499s
 314688 154222 -183.68604   41    7          - -223.85186      -  14.5 66531s
 314796 154307 -185.00293   44    -          - -223.84729      -  14.5 66591s
 314981 154387 -183.13064   44    2          - -223.84123      -  14.5 66675s
 315052 154451 -182.13883   42    7          - -223.83920      -  14.5 66720s
 315116 154482 -195.64608   39    6          - -223.83894      -  14.5 66757s
 315292 154599 -173.80067   48    -          - -223.82687      -  14.5 66875s
 315342 154618 -189.44414   41   14          - -223.82594      -  14.5 66893s
 315511 154732 -187.89666   51    6          - -223.82066      -  14.5 66977s
 315627 154778 -205.14115   45   10          - -223.81976      -  14.5 67029s
 315751 154868 -183.93208   46    6          - -223.80633      -  14.5 67098s
 315811 154891 -177.45401   49    5          - -223.80396      -  14.5 67137s
 315970 154982 -199.96090   43    2          - -223.79238      -  14.5 67219s
 316178 155108 -157.10600   49    -          - -223.78665      -  14.5 67359s
 316233 155134 -189.57501   46   12          - -223.78651      -  14.5 67393s
 316406 155244 -198.53509   43   21          - -223.77570      -  14.5 67495s
 316468 155269 -168.11582   41    4          - -223.77412      -  14.5 67529s
 316657 155406 -197.32373   44   11          - -223.76703      -  14.5 67622s
 316792 155458 -200.65333   41    5          - -223.76587      -  14.5 67719s
 316925 155535 -201.71494   46    9          - -223.75558      -  14.5 67798s
 316980 155559 -184.02623   52    -          - -223.75547      -  14.5 67846s
 317146 155673 -195.90129   41   17          - -223.74626      -  14.5 67972s
 317389 155808 -186.97011   43    5          - -223.73488      -  14.5 68127s
 317438 155829 -207.76320   39   20          - -223.73246      -  14.5 68165s
 317626 155947 -195.39050   40   14          - -223.71893      -  14.5 68259s
 317807 156048 -197.88937   44   11          - -223.71224      -  14.5 68385s
 317928 156095 -165.24173   46    -          - -223.70993      -  14.5 68438s
 318058 156185 -197.54189   41   17          - -223.70498      -  14.5 68532s
 318112 156210 -196.07330   51    -          - -223.70132      -  14.5 68594s
 318296 156340 -191.36702   45   10          - -223.68715      -  14.5 68718s
 318412 156383 -207.18841   43    2          - -223.68533      -  14.5 68781s
 318525 156458 -189.50983   50    2          - -223.67541      -  14.5 68836s
 318578 156484 -180.27984   44    -          - -223.67500      -  14.5 68875s
 318698 156533 -197.47285   40   18          - -223.67280      -  14.5 68904s
 318753 156611 -180.77328   47    -          - -223.67043      -  14.5 68931s
 318984 156727 -167.62320   44    5          - -223.66217      -  14.5 69020s
 319225 156878 -197.35967   44    5          - -223.64719      -  14.5 69174s
 319272 156899 -182.94162   49    6          - -223.64681      -  14.5 69208s
 319322 156918 -194.03492   41    5          - -223.64551      -  14.5 69237s
 319437 156981 -178.59812   47    6          - -223.63392      -  14.5 69317s
 319662 157124 -195.60520   47    8          - -223.62517      -  14.5 69421s
 319880 157254 -195.82189   50    -          - -223.61193      -  14.5 69545s
 319997 157295 -194.36093   42    5          - -223.60991      -  14.5 69607s
 320123 157394 -193.03805   47   10          - -223.60223      -  14.5 69687s
 320300 157462 -183.91946   48    4          - -223.59501      -  14.6 69780s
 320358 157508 -187.74592   44    -          - -223.59493      -  14.6 69875s
 320403 157524 -187.76482   44    -          - -223.59425      -  14.6 69899s
 320536 157643 -182.42297   47    -          - -223.58103      -  14.6 70025s
 320790 157790 -183.16199   48    -          - -223.56881      -  14.6 70172s
 320903 157834 -182.69485   36   16          - -223.56832      -  14.6 70239s
 321017 157943 -198.07056   37   17          - -223.56219      -  14.6 70305s
 321079 157962 -191.04595   42    9          - -223.56126      -  14.6 70319s
 321201 158010 -198.22199   48    4          - -223.55670      -  14.6 70407s
 321260 158069 -194.67055   45    -          - -223.55549      -  14.6 70439s
 321490 158188 -197.54446   38   24          - -223.53925      -  14.6 70594s
 321546 158213 -171.84334   46    5          - -223.53761      -  14.6 70613s
 321718 158349 -160.00064   44    -          - -223.53027      -  14.6 70725s
 321895 158411 -210.97067   49    -          - -223.52178      -  14.6 70847s
 321962 158467 -197.19868   45    4          - -223.51470      -  14.6 70889s
 322075 158518 -181.07675   46    5          - -223.51155      -  14.6 70987s
 322195 158593 -187.33851   39   19          - -223.49963      -  14.6 71053s
 322252 158620 -168.95373   47    2          - -223.49818      -  14.6 71083s
 322308 158647 -197.20328   43    2          - -223.49817      -  14.6 71133s
 322429 158746 -200.89382   37   12          - -223.49516      -  14.6 71224s
 322468 158757 -191.32489   43    2          - -223.49408      -  14.6 71233s
 322588 158806 -184.46827   47    -          - -223.49047      -  14.6 71312s
 322646 158885 -205.79186   41   13          - -223.48750      -  14.6 71350s
 322703 158905 -195.35936   45   10          - -223.48633      -  14.6 71379s
 322820 158948 -195.67836   38   18          - -223.48150      -  14.6 71438s
 322876 159006 -196.08903   49    -          - -223.47932      -  14.6 71481s
 322993 159061 -196.56612   41    5          - -223.47932      -  14.6 71515s
 323081 159114 -213.64152   46    3          - -223.47379      -  14.6 71568s
 323315 159272 -202.00937   38   25          - -223.45501      -  14.6 71754s
 323434 159319 -186.13588   47    -          - -223.45377      -  14.6 71833s
 323561 159412 -188.87104   37    6          - -223.44783      -  14.6 71891s
 323626 159438 -183.42650   47    -          - -223.44672      -  14.6 71930s
 323808 159558 -182.68524   48    -          - -223.43832      -  14.6 72062s
 323851 159574 -206.94061   39   17          - -223.43814      -  14.6 72076s
 323978 159676 -208.69157   39    -          - -223.42945      -  14.6 72139s
 324201 159780 -195.70733   42    3          - -223.42421      -  14.6 72269s
 324263 159810 -190.53417   49    5          - -223.42240      -  14.6 72314s
 324331 159837 -208.31158   43    3          - -223.41756      -  14.6 72344s
 324459 159934 -180.82696   45    -          - -223.41089      -  14.6 72451s
 324696 160064 -201.25407   33   32          - -223.39492      -  14.6 72572s
 324846 160126 -198.26991   47    -          - -223.38539      -  14.6 72671s
 324897 160175 -199.94458   49    7          - -223.38498      -  14.6 72693s
 325067 160245 -195.07380   41   13          - -223.37756      -  14.6 72768s
 325124 160335 -200.13955   42    4          - -223.37726      -  14.6 72845s
 325173 160350 -190.73163   51    -          - -223.37433      -  14.6 72870s
 325232 160373 -188.79327   43   15          - -223.37385      -  14.6 72894s
 325351 160447 -187.03431   41   15          - -223.36901      -  14.6 72962s
 325413 160476 -194.42021   46    -          - -223.36565      -  14.6 72997s
 325601 160578 -194.80073   50    -          - -223.35103      -  14.6 73104s
 325825 160722 -192.61402   44   10          - -223.34492      -  14.6 73257s
 326061 160840 -192.41485   47    -          - -223.33238      -  14.6 73435s
 326298 160971 -202.64924   40   15          - -223.32151      -  14.6 73592s
 326340 160988 -201.52560   38   12          - -223.32107      -  14.6 73622s
 326481 161117 -201.51913   40   15          - -223.31133      -  14.6 73718s
 326544 161139 -208.12369   43   25          - -223.31011      -  14.6 73732s
 326737 161242 -188.48377   49    -          - -223.30254      -  14.6 73846s
 326863 161294 -181.52725   43    5          - -223.30143      -  14.6 73897s
 326977 161387 -201.58872   45    -          - -223.29354      -  14.6 73981s
 327203 161528 -213.57389   39   12          - -223.28180      -  14.6 74125s
 327371 161624 -186.68206   39   13          - -223.27390      -  14.6 74194s
 327475 161667 -182.69863   47    5          - -223.27235      -  14.6 74278s
 327607 161771 -193.98654   43   13          - -223.26599      -  14.6 74397s
 327785 161840 -190.98140   44    5          - -223.25869      -  14.6 74512s
 327846 161923 -193.75640   37   22          - -223.25861      -  14.6 74561s
 327951 161958 -194.99298   48    2          - -223.25384      -  14.6 74616s
 328056 162052 -198.54345   50    6          - -223.24445      -  14.6 74665s
 328243 162123 -191.28704   42    -          - -223.23794      -  14.6 74764s
 328304 162175 -196.95920   42   15          - -223.23571      -  14.6 74793s
 328349 162193 -195.71337   43    3          - -223.23520      -  14.6 74818s
 328493 162299 -194.85734   44    5          - -223.23194      -  14.6 74887s
 328721 162432 -205.99316   36   17          - -223.21795      -  14.6 75033s
 328778 162457 -202.21435   41   19          - -223.21669      -  14.6 75083s
 328841 162482 -166.89548   49    -          - -223.21659      -  14.6 75133s
 328963 162581 -181.05987   44    5          - -223.20656      -  14.6 75207s
 329035 162613 -188.36084   42    -          - -223.20610      -  14.6 75268s
 329167 162667 -206.86854   51    -          - -223.20009      -  14.6 75324s
 329225 162713 -193.62939   51    -          - -223.19941      -  14.6 75379s
 329276 162739 -193.22335   42   17          - -223.19941      -  14.6 75405s
 329331 162758 -198.24240   50    2          - -223.19783      -  14.6 75425s
 329424 162837 -190.01059   45    -          - -223.19203      -  14.6 75536s
 329541 162887 -193.72224   43    2          - -223.18792      -  14.6 75586s
 329663 162984 -185.93849   47    -          - -223.17714      -  14.6 75691s
 329907 163132 -201.79930   38   11          - -223.16267      -  14.6 75815s
 329966 163153 -183.94496   49    4          - -223.16151      -  14.6 75846s
 330136 163285 -177.90097   39   14          - -223.15487      -  14.6 75981s
 330204 163311 -197.50346   36   10          - -223.15271      -  14.6 76012s
 330322 163352 -191.38301   41    4          - -223.14818      -  14.6 76062s
 330373 163424 -189.53407   43    8          - -223.14749      -  14.6 76087s
 330583 163547 -194.54586   40   18          - -223.14002      -  14.6 76248s
 330706 163598 -178.55131   50    -          - -223.13876      -  14.6 76315s
 330763 163621 -177.96497   44    -          - -223.13489      -  14.6 76372s
 330819 163682 -160.96977   49    -          - -223.13409      -  14.6 76422s
 330864 163698 -197.66494   37   10          - -223.13212      -  14.6 76448s
 330998 163804 -187.33695   42    -          - -223.11935      -  14.6 76539s
 331206 163920 -202.84604   38   23          - -223.10835      -  14.6 76636s
 331456 164050 -201.96114   41   11          - -223.10038      -  14.6 76751s
 331668 164184 -191.59243   44   12          - -223.09261      -  14.6 76878s
 331906 164315 -193.11070   47    -          - -223.08308      -  14.6 77016s
 332142 164446 -198.63689   42    6          - -223.06912      -  14.6 77163s
 332365 164586 -196.89824   38   19          - -223.06134      -  14.6 77294s
 332477 164627 -169.02511   48    -          - -223.05999      -  14.6 77371s
 332586 164720 -201.23797   40    9          - -223.05027      -  14.6 77476s
 332644 164744 -194.45612   43    -          - -223.04961      -  14.6 77553s
 332793 164840 -187.25710   45    5          - -223.03851      -  14.6 77639s
 332842 164860 -199.30628   39   16          - -223.03782      -  14.6 77659s
 333004 164975 -196.50704   34   17          - -223.03114      -  14.6 77756s
 333063 164996 -199.26125   38   22          - -223.02903      -  14.6 77776s
 333211 165093 -206.51831   50    2          - -223.01614      -  14.6 77862s
 333336 165143 -180.93011   49    4          - -223.01153      -  14.6 77924s
 333466 165230 -212.43798   41   22          - -223.00583      -  14.6 78031s
 333612 165288 -145.75005   41    -          - -223.00290      -  14.6 78139s
 333661 165341 -195.11352   37   26          - -222.99972      -  14.6 78173s
 333779 165393 -187.25659   51    4          - -222.99398      -  14.6 78235s
 333901 165475 -181.01317   47    -          - -222.98085      -  14.6 78321s
 334010 165520 -201.23086   42   19          - -222.97959      -  14.6 78384s
 334121 165616 -186.69154   48    4          - -222.97116      -  14.6 78501s
 334234 165665 -205.21325   40   18          - -222.96726      -  14.6 78578s
 334358 165731 -189.85668   41   13          - -222.95942      -  14.6 78628s
 334412 165755 -168.65414   48    -          - -222.95925      -  14.6 78660s
 334561 165881 -207.74151   34   13          - -222.95282      -  14.7 78762s
 334613 165900 -207.16120   39    8          - -222.95169      -  14.7 78783s
 334675 165924 -208.04917   44    6          - -222.95165      -  14.7 78825s
 334797 166002 -185.23985   48    -          - -222.94536      -  14.7 78880s
 335027 166127 -172.02286   42    5          - -222.93535      -  14.7 79018s
 335086 166148 -177.01155   46    7          - -222.93520      -  14.7 79034s
 335215 166210 -201.07855   48    -          - -222.93232      -  14.7 79122s
 335279 166263 -192.67609   49    -          - -222.93060      -  14.7 79162s
 335329 166288 -172.81848   50    2          - -222.92737      -  14.7 79183s
 335482 166392 -207.58018   44    4          - -222.92215      -  14.7 79300s
 335531 166411 -180.23840   40    7          - -222.92154      -  14.7 79316s
 335711 166520 -202.06652   41   20          - -222.90808      -  14.7 79433s
 335820 166561 -183.51524   44    -          - -222.90601      -  14.7 79490s
 335912 166642 -197.87294   44    -          - -222.89955      -  14.7 79551s
 335971 166666 -189.13185   45    2          - -222.89886      -  14.7 79577s
 336168 166773 -192.71413   44    3          - -222.88708      -  14.7 79742s
 336225 166802 -202.81651   48    -          - -222.88654      -  14.7 79763s
 336392 166922 -169.04507   39    2          - -222.88246      -  14.7 79918s
 336495 166956 -197.90441   46    -          - -222.87784      -  14.7 79997s
 336604 167048 -182.22031   49    4          - -222.87131      -  14.7 80099s
 336665 167073 -176.46230   44    4          - -222.87068      -  14.7 80136s
 336828 167157 -176.43381   49    -          - -222.86088      -  14.7 80207s
 337063 167273 -207.96596   47    9          - -222.85168      -  14.7 80432s
 337176 167327 -181.79989   47    -          - -222.85133      -  14.7 80489s
 337291 167426 -198.45183   36   17          - -222.84716      -  14.7 80599s
 337346 167447 -205.23761   49    7          - -222.84504      -  14.7 80621s
 337530 167561 -178.48205   39    -          - -222.83456      -  14.7 80746s
 337741 167684 -207.36993   40   15          - -222.82380      -  14.7 80924s
 337800 167708 -183.84964   45    2          - -222.82327      -  14.7 80960s
 337971 167821 -194.24815   39    2          - -222.81179      -  14.7 81086s
 338217 167943 -184.47553   45    7          - -222.80062      -  14.7 81275s
 338264 167963 -183.12768   50    -          - -222.79938      -  14.7 81328s
 338402 168077 -193.85761   48    -          - -222.79146      -  14.7 81438s
 338462 168103 -204.78921   52    -          - -222.79139      -  14.7 81465s
 338645 168198 -191.84173   50    -          - -222.78547      -  14.7 81573s
 338702 168222 -168.20769   45    -          - -222.78420      -  14.7 81627s
 338874 168349 -186.24436   51    -          - -222.77584      -  14.7 81737s
 338980 168383 -202.77527   40    8          - -222.77527      -  14.7 81774s
 339042 168409 -173.28989   50    -          - -222.76351      -  14.7 81827s
 339101 168472 -178.68137   48    -          - -222.76349      -  14.7 81900s
 339150 168493 -188.44444   49    2          - -222.76101      -  14.7 81943s
 339254 168535 -191.18814   49    -          - -222.75587      -  14.7 82016s
 339303 168593 -191.60103   49    -          - -222.74649      -  14.7 82052s
 339353 168611 -186.83276   47    5          - -222.74389      -  14.7 82067s
 339532 168702 -188.65779   49    -          - -222.73634      -  14.7 82250s
 339797 168873 -176.46371   40   11          - -222.72769      -  14.7 82420s
 339907 168910 -198.40312   48    -          - -222.72686      -  14.7 82473s
 340018 168995 -188.97288   45   11          - -222.71963      -  14.7 82557s
 340079 169020 -180.68776   49    -          - -222.71893      -  14.7 82600s
 340260 169125 -186.90771   47   10          - -222.70793      -  14.7 82753s
 340371 169170 -193.12278   50    -          - -222.70654      -  14.7 82807s
 340422 169194 -183.60895   46    3          - -222.69848      -  14.7 82845s
 340478 169253 -171.88889   46    -          - -222.69835      -  14.7 82907s
 340647 169326 -210.49828   42   10          - -222.68657      -  14.7 83018s
 340706 169367 -176.91751   43    3          - -222.68252      -  14.7 83059s
 340764 169396 -207.60882   44   12          - -222.68116      -  14.7 83097s
 340871 169448 -197.19090   32   25          - -222.67614      -  14.7 83193s
 340928 169509 -179.14668   43    -          - -222.67534      -  14.7 83233s
 340988 169532 -197.39306   42   16          - -222.67480      -  14.7 83271s
 341112 169583 -192.22566   39    5          - -222.66763      -  14.7 83373s
 341173 169672 -192.08518   40    4          - -222.66384      -  14.7 83409s
 341405 169781 -198.03991   49    7          - -222.65581      -  14.7 83590s
 341457 169802 -202.54514   44   10          - -222.65575      -  14.7 83622s
 341510 169824 -200.40618   52    -          - -222.65406      -  14.7 83663s
 341552 169845 -184.59290   47    -          - -222.64740      -  14.7 83712s
 341600 169920 -197.11631   40    7          - -222.64625      -  14.7 83742s
 341665 169946 -185.53109   41    9          - -222.64597      -  14.7 83770s
 341734 169975 -206.10731   37   14          - -222.64349      -  14.7 83791s
 341857 170062 -186.57310   41    3          - -222.63840      -  14.7 83849s
 341914 170085 -188.68615   49    -          - -222.63806      -  14.7 83903s
 342095 170175 -182.49023   40   13          - -222.62836      -  14.7 84071s
 342164 170202 -191.79391   45   10          - -222.62756      -  14.7 84087s
 342225 170230 -177.38528   46    -          - -222.62732      -  14.7 84168s
 342347 170310 -175.09677   44    -          - -222.62385      -  14.7 84311s
 342393 170327 -192.30856   41    -          - -222.62153      -  14.7 84332s
 342437 170348 -160.24017   44    -          - -222.61959      -  14.7 84381s
 342488 170370 -163.17462   43    -          - -222.61114      -  14.7 84407s
 342537 170426 -189.05395   50    -          - -222.60712      -  14.7 84467s
 342784 170569 -198.50268   41   10          - -222.59293      -  14.7 84647s
 342909 170622 -166.41793   49    -          - -222.58998      -  14.7 84716s
 343026 170698 -174.82526   42    9          - -222.58283      -  14.7 84801s
 343091 170723 -171.81559   47    -          - -222.58269      -  14.7 84839s
 343144 170744 -184.43374   41   10          - -222.58035      -  14.7 84871s
 343265 170836 -185.64272   44    5          - -222.57264      -  14.7 84966s
 343442 170958 -197.26543   47    5          - -222.56324      -  14.7 85137s
 343689 171103 -208.35676   38   11          - -222.54960      -  14.7 85272s
 343919 171246 -207.82153   42    5          - -222.53751      -  14.7 85417s
 344162 171386 -187.96605   38    5          - -222.52649      -  14.7 85567s
 344228 171414 -207.03579   41   14          - -222.52478      -  14.7 85589s
 344277 171428 -188.59367   41    8          - -222.52431      -  14.7 85616s
 344370 171510 -202.45880   37   12          - -222.52183      -  14.7 85690s
 344620 171638 -191.71746   48    -          - -222.51457      -  14.7 85836s
 344836 171773 -198.81390   44    5          - -222.50629      -  14.7 86007s
 344892 171795 -180.59186   42    5          - -222.50455      -  14.7 86051s
 344953 171819 -161.75275   45    -          - -222.50374      -  14.7 86078s
 345063 171918 -199.80672   36    6          - -222.49337      -  14.7 86157s
 345109 171934 -190.98046   48    -          - -222.49113      -  14.7 86191s
 345210 171967 -166.33658   44    -          - -222.47918      -  14.7 86245s
 345266 172011 -187.54955   45    6          - -222.47881      -  14.7 86276s
 345336 172047 -189.20613   44    6          - -222.47764      -  14.7 86324s
 345406 172078 -178.84929   45    -          - -222.47736      -  14.7 86379s
 345523 172167 -178.11603   51    -          - -222.46806      -  14.7 86475s
 345662 172224 -203.20268   44   10          - -222.46150      -  14.7 86579s
 345713 172296 -182.28537   39   19          - -222.46136      -  14.7 86621s
 345973 172471 -201.87958   34   25          - -222.45265      -  14.7 86772s
 346026 172487 -196.80843   38   28          - -222.45160      -  14.7 86795s
 346083 172503 -202.14297   38   18          - -222.45150      -  14.7 86817s
 346145 172528 -198.83630   45    -          - -222.44764      -  14.7 86872s
 346200 172565 -189.56269   51    4          - -222.44499      -  14.7 86893s
 346302 172615 -193.91658   43    6          - -222.44415      -  14.7 86958s
 346396 172677 -192.78904   48    -          - -222.44297      -  14.7 87056s
 346638 172830 -189.25928   40   14          - -222.43230      -  14.7 87218s
 346759 172882 -194.09889   52    -          - -222.43078      -  14.7 87303s
 346863 172963 -184.89172   42    3          - -222.41985      -  14.7 87361s
 346972 173006 -188.07204   40   14          - -222.41612      -  14.7 87460s
 347036 173031 -201.59126   38    6          - -222.41132      -  14.7 87482s
 347099 173079 -192.55502   46    -          - -222.41038      -  14.7 87541s
 347259 173150 -196.43319   44    5          - -222.40347      -  14.7 87663s
 347320 173198 -168.49393   49    -          - -222.40006      -  14.7 87717s
 347372 173215 -190.04652   47    4          - -222.39997      -  14.7 87738s
 347552 173342 -193.25567   42    4          - -222.39041      -  14.7 87924s
 347605 173363 -191.33549   37    6          - -222.38817      -  14.7 87946s
 347654 173384 -173.73109   44    7          - -222.38808      -  14.7 87962s
 347743 173458 -187.13732   46    -          - -222.38325      -  14.7 88016s
 347795 173477 -181.66500   47    -          - -222.38237      -  14.7 88038s
 347852 173501 -190.02542   46    5          - -222.38017      -  14.7 88077s
 347964 173580 -188.40631   50    -          - -222.37030      -  14.7 88198s
 348157 173664 -188.63853   49    -          - -222.35971      -  14.7 88336s
 348215 173755 -200.45727   47    2          - -222.35934      -  14.7 88378s
 348255 173766 -174.66147   46    -          - -222.35901      -  14.8 88406s
 348313 173789 -170.48018   34    7          - -222.35682      -  14.7 88433s
 348412 173856 -189.65358   42    5          - -222.35168      -  14.8 88475s
 348637 173985 -199.36054   35   22          - -222.34379      -  14.8 88621s
 348701 174009 -199.69063   48    -          - -222.34376      -  14.8 88649s
 348762 174030 -187.91265   46    -          - -222.34376      -  14.8 88682s
 348853 174125 -181.37711   49    9          - -222.33645      -  14.8 88763s
 348917 174151 -196.62232   45   10          - -222.33562      -  14.8 88819s
 349029 174191 -191.81839   40   11          - -222.32486      -  14.8 88880s
 349094 174240 -175.79890   45    3          - -222.32455      -  14.8 88928s
 349139 174260 -193.93677   43   10          - -222.32329      -  14.8 88984s
 349297 174369 -192.13162   46    -          - -222.30668      -  14.8 89119s
 349528 174502 -176.48288   38   21          - -222.29628      -  14.8 89317s
 349588 174524 -195.99766   43    -          - -222.29596      -  14.8 89346s
 349758 174635 -175.21093   47    4          - -222.28534      -  14.8 89471s
 349980 174747 -168.22937   48    -          - -222.26955      -  14.8 89642s
 350094 174797 -175.67722   47    2          - -222.26928      -  14.8 89715s
 350207 174858 -192.23090   43    9          - -222.26041      -  14.8 89819s
 350454 175001 -161.14184   42    7          - -222.24801      -  14.8 90048s
 350590 175060 -178.92975   52    -          - -222.23656      -  14.8 90143s
 350637 175106 -197.38724   49    6          - -222.23620      -  14.8 90185s
 350687 175126 -184.94109   50    5          - -222.23520      -  14.8 90202s
 350796 175176 -188.46746   46    -          - -222.23339      -  14.8 90308s
 350846 175230 -192.30885   49    -          - -222.23026      -  14.8 90335s
 351027 175315 -199.41831   33   25          - -222.22032      -  14.8 90459s
 351092 175365 -189.26249   42    3          - -222.21803      -  14.8 90479s
 351155 175392 -199.78856   45    4          - -222.21720      -  14.8 90507s
 351328 175511 -170.18232   43    3          - -222.20781      -  14.8 90651s
 351378 175532 -203.10537   37   24          - -222.20622      -  14.8 90673s
 351435 175555 -180.11857   42    6          - -222.20587      -  14.8 90717s
 351530 175621 -193.53892   45    5          - -222.19652      -  14.8 90788s
 351707 175699 -195.65465   46    9          - -222.19021      -  14.8 90913s
 351768 175771 -183.74452   38   13          - -222.18866      -  14.8 90945s
 351827 175794 -187.62430   45    -          - -222.18785      -  14.8 90979s
 352012 175880 -209.18693   50    -          - -222.17970      -  14.8 91102s
 352208 176011 -204.06886   47    3          - -222.17526      -  14.8 91266s
 352275 176043 -173.73642   43    5          - -222.17292      -  14.8 91320s
 352448 176165 -196.46176   47    5          - -222.15991      -  14.8 91451s
 352511 176188 -187.48234   46    2          - -222.15951      -  14.8 91474s
 352687 176297 -195.82067   44    9          - -222.14924      -  14.8 91584s
 352865 176369 -213.94361   39    7          - -222.14505      -  14.8 91698s
 352926 176411 -180.84004   48    4          - -222.14415      -  14.8 91753s
 353104 176524 -182.33891   37   15          - -222.13750      -  14.8 91914s
 353164 176549 -192.53328   43    6          - -222.13541      -  14.8 91964s
 353345 176682 -186.33036   37   16          - -222.12706      -  14.8 92081s
 353596 176820 -182.40940   42    6          - -222.11877      -  14.8 92261s
 353731 176879 -184.15428   49    -          - -222.11740      -  14.8 92318s
 353839 176962 -169.72350   49    -          - -222.11409      -  14.8 92378s
 354031 177052 -174.46100   49    5          - -222.10437      -  14.8 92495s
 354275 177184 -203.85540   39   21          - -222.08851      -  14.8 92697s
 354530 177354 -182.57332   45    -          - -222.08069      -  14.8 92865s
 354777 177491 -196.74115   43    3          - -222.06659      -  14.8 93050s
 354827 177510 -185.91697   42    9          - -222.06506      -  14.8 93078s
 354978 177607 -175.74665   48    -          - -222.05999      -  14.8 93179s
 355101 177656 -188.95849   47    2          - -222.05921      -  14.8 93293s
 355239 177778 -178.98710   43    4          - -222.04855      -  14.8 93384s
 355285 177792 -195.51859   42    5          - -222.04854      -  14.8 93407s
 355453 177906 -183.64418   46    -          - -222.04310      -  14.8 93508s
 355512 177931 -191.61764   40   13          - -222.04143      -  14.8 93542s
 355679 178023 -204.35872   38   20          - -222.03143      -  14.8 93675s
 355763 178051 -176.71314   46    -          - -222.02867      -  14.8 93704s
 355802 178071 -173.37633   45    -          - -222.02732      -  14.8 93738s
 355848 178136 -190.10190   48    2          - -222.02432      -  14.8 93765s
 355918 178168 -173.20263   38   11          - -222.02365      -  14.8 93793s
 356083 178275 -192.33519   44    2          - -222.00450      -  14.8 93944s
 356203 178324 -201.13036   41   14          - -222.00116      -  14.8 94019s
 356259 178346 -180.41274   46    3          - -221.99656      -  14.8 94064s
 356318 178396 -161.87555   50    -          - -221.99188      -  14.8 94097s
 356383 178424 -182.73519   48    -          - -221.99121      -  14.8 94142s
 356447 178452 -182.78571   46    6          - -221.99094      -  14.8 94170s
 356567 178539 -189.35598   46    5          - -221.98330      -  14.8 94259s
 356616 178556 -181.21029   46    5          - -221.98228      -  14.8 94288s
 356794 178682 -175.33056   45    -          - -221.97698      -  14.8 94382s
 356968 178749 -182.31545   40    6          - -221.96986      -  14.8 94491s
 357025 178836 -187.74698   46    -          - -221.96826      -  14.8 94535s
 357148 178881 -177.86411   46    2          - -221.96762      -  14.8 94576s
 357212 178909 -194.96313   41   10          - -221.96578      -  14.8 94610s
 357278 178949 -192.54159   44    5          - -221.96332      -  14.8 94637s
 357328 178970 -170.30817   45    -          - -221.96323      -  14.8 94666s
 357420 179014 -189.03260   44    4          - -221.95852      -  14.8 94775s
 357465 179073 -171.02318   47    7          - -221.95520      -  14.8 94802s
 357519 179093 -201.38360   47    9          - -221.95345      -  14.8 94831s
 357686 179221 -181.11857   49    5          - -221.94149      -  14.8 94957s
 357804 179264 -194.54512   35   25          - -221.94037      -  14.8 95072s
 357925 179343 -189.73063   44    -          - -221.93016      -  14.8 95149s
 358028 179384 -173.67442   45    -          - -221.92689      -  14.8 95213s
 358136 179449 -208.26153   46    2          - -221.92382      -  14.8 95274s
 358197 179479 -176.84359   45    -          - -221.92003      -  14.8 95320s
 358358 179592 -194.48278   41   11          - -221.91354      -  14.8 95468s
 358564 179707 -188.81691   48    -          - -221.90254      -  14.8 95655s
 358810 179847 -202.37610   44   19          - -221.89280      -  14.8 95818s
 358865 179869 -188.10776   50   10          - -221.88970      -  14.8 95852s
 358960 179908 -169.90777   50    2          - -221.88584      -  14.8 95899s
 359013 179978 -195.21816   48    -          - -221.88415      -  14.8 95938s
 359117 180015 -211.56125   37   17          - -221.88246      -  14.8 96007s
 359237 180090 -203.24830   44    6          - -221.87580      -  14.8 96144s
 359289 180108 -139.19361   46    -          - -221.87552      -  14.8 96179s
 359339 180131 -175.55893   48    -          - -221.87243      -  14.8 96203s
 359432 180192 -197.32980   50    -          - -221.86343      -  14.8 96299s
 359674 180319 -186.84283   46    -          - -221.85521      -  14.8 96479s
 359896 180448 -187.91693   48    -          - -221.83912      -  14.8 96686s
 360075 180549 -192.47167   44    5          - -221.83599      -  14.8 96794s
 360317 180702 -198.46209   47    2          - -221.82758      -  14.8 96971s
 360365 180722 -180.65002   47    3          - -221.82701      -  14.8 97000s
 360427 180746 -186.02945   50    -          - -221.82635      -  14.8 97040s
 360531 180833 -203.25226   42    9          - -221.81678      -  14.8 97129s
 360588 180856 -180.94935   45    -          - -221.81632      -  14.8 97170s
 360767 180961 -199.21444   44   11          - -221.80197      -  14.8 97308s
 360883 181013 -207.75116   38   25          - -221.80148      -  14.8 97378s
 360982 181100 -200.26345   42   16          - -221.79592      -  14.8 97433s
 361046 181125 -191.49335   42    5          - -221.79546      -  14.8 97473s
 361232 181223 -191.94747   43   15          - -221.78545      -  14.8 97623s
 361292 181254 -194.82781   44   13          - -221.78535      -  14.8 97687s
 361458 181357 -202.03438   44    8          - -221.77357      -  14.8 97859s
 361670 181505 -182.98356   45   13          - -221.76843      -  14.8 98036s
 361898 181620 -168.28180   44    5          - -221.76172      -  14.8 98198s
 362130 181764 -178.38802   49    -          - -221.75445      -  14.8 98377s
 362166 181776 -190.17420   45    -          - -221.75293      -  14.8 98401s
 362317 181866 -184.47940   44    -          - -221.74102      -  14.8 98528s
 362382 181896 -195.56343   41   18          - -221.74096      -  14.8 98569s
 362573 182005 -188.50784   42    -          - -221.73116      -  14.8 98708s
 362670 182041 -193.60707   40    5          - -221.73038      -  14.8 98743s
 362769 182131 -194.27683   45    9          - -221.72849      -  14.8 98845s
 362833 182162 -185.32890   43    8          - -221.72475      -  14.8 98852s
 363012 182251 -191.61490   42    7          - -221.71941      -  14.8 98931s
 363178 182324 -177.50039   49    4          - -221.70965      -  14.8 99055s
 363233 182378 -203.98651   48    -          - -221.70858      -  14.8 99089s
 363287 182394 -173.88708   46    -          - -221.70782      -  14.8 99101s
 363459 182510 -192.39360   41   14          - -221.70266      -  14.8 99280s
 363518 182534 -208.02983   44    5          - -221.70016      -  14.8 99310s
 363581 182561 -184.51096   46    2          - -221.69809      -  14.8 99333s
 363683 182643 -164.58626   47    -          - -221.69009      -  14.8 99431s
 363804 182695 -178.48043   49    -          - -221.68854      -  14.8 99555s
 363916 182782 -184.98968   44    6          - -221.67624      -  14.8 99648s
 364119 182879 -198.97121   40   13          - -221.66372      -  14.8 99793s
 364242 182936 -178.44437   50    -          - -221.66092      -  14.9 99900s
 364308 182967 -179.44689   44    8          - -221.64935      -  14.9 99953s
 364371 183027 -184.23678   44    5          - -221.64861      -  14.9 100016s
 364617 183168 -193.42058   41    7          - -221.64249      -  14.9 100211s
 364818 183312 -189.05005   51    2          - -221.63340      -  14.9 100347s
 365046 183429 -194.80361   42   10          - -221.62495      -  14.9 100511s
 365161 183475 -178.21415   44    -          - -221.62342      -  14.9 100581s
 365273 183569 -204.46139   41    9          - -221.61344      -  14.9 100674s
 365327 183590 -184.20489   45    -          - -221.61303      -  14.9 100715s
 365476 183673 -159.13071   45    3          - -221.60517      -  14.9 100849s
 365586 183716 -189.75670   44    6          - -221.60407      -  14.9 100932s
 365712 183816 -181.74594   44    -          - -221.59396      -  14.9 101001s
 365767 183839 -188.31697   41    -          - -221.59261      -  14.9 101060s
 365943 183944 -182.76626   40   18          - -221.58729      -  14.9 101159s
 365997 183963 -185.71670   47    3          - -221.58470      -  14.9 101171s
 366176 184075 -187.81608   40    4          - -221.57963      -  14.9 101340s
 366398 184205 -183.65655   48    -          - -221.56768      -  14.9 101576s
 366455 184231 -198.66875   41    5          - -221.56457      -  14.9 101606s
 366632 184355 -185.81132   44    2          - -221.55779      -  14.9 101753s
 366828 184447 -200.56443   42   17          - -221.55104      -  14.9 101852s
 366892 184477 -196.03091   46    6          - -221.55080      -  14.9 101864s
 367019 184533 -198.26147   49    3          - -221.54657      -  14.9 101905s
 367075 184584 -193.14448   46    -          - -221.54573      -  14.9 102007s
 367193 184638 -183.72079   45   11          - -221.54497      -  14.9 102066s
 367312 184719 -192.91407   46    5          - -221.53312      -  14.9 102184s
 367361 184741 -184.16865   47   10          - -221.53245      -  14.9 102232s
 367459 184786 -201.32408   43   12          - -221.52825      -  14.9 102323s
 367514 184836 -162.79244   45   10          - -221.52648      -  14.9 102352s
 367571 184859 -184.90032   49    2          - -221.52617      -  14.9 102405s
 367693 184908 -193.14657   48    -          - -221.52083      -  14.9 102532s
 367756 184973 -172.23561   47    -          - -221.51807      -  14.9 102649s
 367812 184998 -206.99069   46    2          - -221.51620      -  14.9 102686s
 367991 185088 -195.80184   46    4          - -221.50592      -  14.9 102793s
 368166 185198 -187.18471   40   14          - -221.49368      -  14.9 102966s
 368290 185253 -194.27934   47    -          - -221.49221      -  14.9 103037s
 368415 185367 -200.67150   46    -          - -221.48124      -  14.9 103094s
 368461 185383 -209.70478   45    -          - -221.48013      -  14.9 103137s
 368638 185493 -180.21954   42   11          - -221.47078      -  14.9 103255s
 368876 185610 -178.21973   47    2          - -221.45788      -  14.9 103439s
 369080 185733 -169.74258   43    4          - -221.45482      -  14.9 103569s
 369202 185787 -189.16173   47    5          - -221.45327      -  14.9 103666s
 369261 185808 -188.63124   49    6          - -221.44614      -  14.9 103702s
 369317 185860 -198.07648   48    -          - -221.44574      -  14.9 103802s
 369377 185887 -181.99942   45    -          - -221.44278      -  14.9 103849s
 369433 185911 -189.62742   41    5          - -221.44103      -  14.9 103885s
 369493 185939 -189.86361   44    8          - -221.42500      -  14.9 103915s
 369553 186005 -215.53030   41   17          - -221.42340      -  14.9 103962s
 369607 186028 -197.20339   47    -          - -221.42326      -  14.9 104010s
 369766 186142 -165.56390   48    6          - -221.41837      -  14.9 104122s
 369815 186159 -203.05620   48    5          - -221.41562      -  14.9 104159s
 369865 186181 -192.07867   49    6          - -221.41461      -  14.9 104212s
 369978 186238 -198.61419   52    5          - -221.40853      -  14.9 104259s
 370225 186388 -195.86932   39   17          - -221.39188      -  14.9 104469s
 370273 186403 -192.23114   43    5          - -221.39166      -  14.9 104488s
 370325 186425 -187.49297   46    -          - -221.39153      -  14.9 104536s
 370425 186499 -167.07594   38    -          - -221.38555      -  14.9 104625s
 370653 186606 -179.79466   45    2          - -221.37432      -  14.9 104780s
 370838 186692 -157.63442   44    -          - -221.36451      -  14.9 104901s
 370890 186755 -178.32670   45    -          - -221.36420      -  14.9 104983s
 370945 186775 -192.35565   43    3          - -221.36363      -  14.9 104989s
 370991 186794 -207.33068   47    -          - -221.36338      -  14.9 105037s
 371096 186875 -193.26386   48    -          - -221.35965      -  14.9 105108s
 371304 187021 -194.26588   41   16          - -221.34755      -  14.9 105271s
 371360 187040 -198.24317   40    8          - -221.34698      -  14.9 105283s
 371559 187159 -184.95876   45    6          - -221.34212      -  14.9 105425s
 371614 187181 -179.27904   41    4          - -221.34198      -  14.9 105455s
 371775 187281 -198.45667   48    -          - -221.33757      -  14.9 105589s
 371832 187305 -176.40140   39    4          - -221.33556      -  14.9 105625s
 371989 187401 -197.25912   41    9          - -221.32821      -  14.9 105764s
 372217 187535 -194.13768   44   12          - -221.31752      -  14.9 105915s
 372281 187558 -163.66304   50    5          - -221.31648      -  14.9 105945s
 372346 187590 -179.22819   46    -          - -221.31640      -  14.9 106037s
 372456 187682 -188.68718   43    9          - -221.30603      -  14.9 106108s
 372507 187700 -179.77138   42    2          - -221.30505      -  14.9 106140s
 372677 187800 -188.17365   46    -          - -221.29913      -  14.9 106290s
 372875 187929 -195.37682   47    -          - -221.28861      -  14.9 106464s
 372985 187974 -182.47456   49    -          - -221.28681      -  14.9 106549s
 373102 188057 -179.15729   44    -          - -221.27739      -  14.9 106619s
 373343 188194 -201.49318   48    2          - -221.27057      -  14.9 106763s
 373402 188218 -183.35792   47    -          - -221.26986      -  14.9 106818s
 373520 188266 -189.02405   50    -          - -221.26042      -  14.9 106890s
 373576 188323 -200.96575   42    4          - -221.25685      -  14.9 106936s
 373674 188360 -179.95228   46    -          - -221.25231      -  14.9 107042s
 373766 188455 -197.55713   39   11          - -221.24688      -  14.9 107126s
 373824 188474 -202.81066   39   16          - -221.24607      -  14.9 107168s
 373939 188523 -183.57113   46    -          - -221.23351      -  14.9 107253s
 373998 188569 -185.98126   46    -          - -221.23157      -  14.9 107294s
 374226 188704 -192.54065   46    5          - -221.22374      -  14.9 107525s
 374435 188825 -195.10617   39   16          - -221.21702      -  14.9 107701s
 374692 188979 -184.24105   43   13          - -221.20948      -  14.9 107866s
 374870 189077 -196.83134   38   25          - -221.20318      -  14.9 107963s
 374929 189103 -183.68655   44    5          - -221.20246      -  14.9 108001s
 375112 189223 -172.28489   44    6          - -221.19673      -  14.9 108158s
 375356 189362 -161.68375   40    -          - -221.18587      -  14.9 108328s
 375583 189487 -194.20205   45    3          - -221.17701      -  14.9 108488s
 375747 189552 -184.64767   50    -          - -221.17212      -  14.9 108562s
 375803 189643 -173.62468   49    5          - -221.17176      -  14.9 108597s
 375981 189710 -201.96180   38   17          - -221.16662      -  14.9 108738s
 376044 189745 -184.77322   45    4          - -221.16558      -  14.9 108779s
 376292 189898 -213.69614   39   14          - -221.15377      -  14.9 109010s
 376333 189912 -190.97572   39    2          - -221.15198      -  14.9 109029s
 376424 189950 -180.22700   49    -          - -221.15020      -  14.9 109103s
 376475 189998 -163.41803   42    4          - -221.14900      -  14.9 109151s
 376593 190050 -172.80207   48    3          - -221.14846      -  14.9 109224s
 376705 190151 -199.21765   37   15          - -221.14323      -  14.9 109295s
 376763 190172 -191.05563   47    8          - -221.14247      -  14.9 109320s
 376938 190264 -207.90467   36   18          - -221.13323      -  14.9 109485s
 377125 190346 -177.91296   38    9          - -221.12379      -  14.9 109645s
 377180 190422 -200.97005   46    3          - -221.12334      -  14.9 109755s
 377312 190464 -186.47347   49    -          - -221.11637      -  14.9 109854s
 377360 190527 -199.00561   45   16          - -221.11586      -  14.9 109895s
 377411 190545 -189.99165   44    -          - -221.11557      -  14.9 109926s
 377593 190670 -194.83343   43    9          - -221.10606      -  14.9 110073s
 377783 190741 -199.77980   48    -          - -221.10083      -  14.9 110204s
 377835 190781 -180.41079   44    6          - -221.09871      -  14.9 110283s
 377926 190825 -184.86846   50    -          - -221.09622      -  14.9 110375s
 378022 190926 -184.90478   41   20          - -221.07779      -  14.9 110473s
 378086 190949 -195.03519   51    5          - -221.07609      -  14.9 110504s
 378143 190970 -212.88696   35   18          - -221.07568      -  14.9 110541s
 378200 190991 -160.81213   46    5          - -221.07207      -  14.9 110572s
 378262 191047 -176.70499   45    3          - -221.07043      -  14.9 110614s
 378307 191065 -188.72673   47    -          - -221.06956      -  14.9 110658s
 378366 191086 -193.43339   42   13          - -221.06883      -  14.9 110677s
 378476 191167 -187.61305   45    5          - -221.06500      -  14.9 110793s
 378708 191303 -195.95402   49    6          - -221.04952      -  14.9 110990s
 378753 191320 -167.51657   39   13          - -221.04855      -  14.9 111020s
 378884 191402 -151.17639   43    -          - -221.04229      -  14.9 111137s
 379067 191481 -199.54974   43    4          - -221.03271      -  14.9 111307s
 379120 191527 -180.46822   47    -          - -221.03265      -  14.9 111361s
 379176 191550 -186.79896   44    2          - -221.03189      -  14.9 111386s
 379359 191670 -190.32185   42   17          - -221.02405      -  14.9 111534s
 379567 191801 -198.69029   36   18          - -221.01066      -  14.9 111665s
 379635 191829 -203.12474   45    9          - -221.00842      -  14.9 111696s
 379807 191935 -181.50850   37   17          - -221.00067      -  14.9 111775s
 379914 191977 -188.53423   50    9          - -220.99949      -  14.9 111881s
 380015 192047 -188.38615   50    -          - -220.99361      -  14.9 111973s
 380061 192064 -195.22421   43    8          - -220.99265      -  14.9 112016s
 380255 192192 -190.65835   39   15          - -220.98282      -  14.9 112183s
 380366 192238 -206.44589   38   18          - -220.98182      -  14.9 112283s
 380430 192267 -184.55048   44    -          - -220.97326      -  14.9 112341s
 380491 192337 -173.35423   46    -          - -220.97107      -  14.9 112395s
 380600 192380 -187.49487   47    -          - -220.96921      -  14.9 112463s
 380697 192453 -194.85954   45    5          - -220.96626      -  14.9 112531s
 380919 192584 -168.80043   41    7          - -220.95241      -  14.9 112743s
 381139 192718 -199.78112   50    -          - -220.94057      -  14.9 112897s
 381378 192834 -165.40726   44    5          - -220.93064      -  14.9 113104s
 381472 192878 -183.76592   48    -          - -220.93064      -  14.9 113160s
 381574 192957 -188.32479   47    2          - -220.92882      -  14.9 113227s
 381786 193102 -196.44939   40   22          - -220.91027      -  15.0 113425s
 382038 193226 -197.30641   47    -          - -220.90625      -  15.0 113575s
 382250 193355 -183.35584   45    6          - -220.89852      -  15.0 113744s
 382488 193482 -173.41812   47    -          - -220.88649      -  15.0 113970s
 382545 193508 -197.66361   39    3          - -220.88635      -  15.0 114019s
 382595 193530 -181.26158   43    -          - -220.88587      -  15.0 114102s
 382704 193613 -174.25324   45    4          - -220.88093      -  15.0 114177s
 382954 193730 -188.44263   52    -          - -220.86974      -  15.0 114403s
 383172 193880 -184.40046   40   20          - -220.86454      -  15.0 114629s
 383225 193898 -180.20212   41    9          - -220.86425      -  15.0 114674s
 383390 193994 -182.18761   48    -          - -220.85795      -  15.0 114817s
 383430 194010 -180.38777   42    9          - -220.85781      -  15.0 114856s
 383576 194108 -192.02587   40    2          - -220.84943      -  15.0 114986s
 383634 194130 -200.80385   39   24          - -220.84490      -  15.0 115011s
 383795 194222 -184.52629   51    -          - -220.83092      -  15.0 115129s
 383852 194247 -189.61366   38   13          - -220.82989      -  15.0 115167s
 383919 194277 -183.37830   47    -          - -220.82888      -  15.0 115230s
 384041 194358 -173.65384   46    4          - -220.81912      -  15.0 115343s
 384088 194380 -201.74237   45   12          - -220.81907      -  15.0 115426s
 384261 194481 -200.51003   40    6          - -220.81142      -  15.0 115563s
 384319 194503 -165.85984   47    -          - -220.81074      -  15.0 115589s
 384520 194627 -189.90715   42    2          - -220.80655      -  15.0 115696s
 384633 194679 -204.81544   35   29          - -220.80565      -  15.0 115812s
 384729 194750 -190.09611   49    -          - -220.79677      -  15.0 115918s
 384789 194774 -195.27868   49    5          - -220.79650      -  15.0 115942s
 384976 194889 -147.23161   45    -          - -220.79454      -  15.0 116130s
 385033 194917 -165.54475   47    -          - -220.79239      -  15.0 116156s
 385084 194937 -166.37797   47    -          - -220.78814      -  15.0 116220s
 385142 194959 -185.97353   49    4          - -220.78351      -  15.0 116258s
 385204 195018 -168.62778   42    7          - -220.78176      -  15.0 116314s
 385401 195150 -208.81960   45    2          - -220.77755      -  15.0 116448s
 385456 195170 -180.76287   48    -          - -220.77550      -  15.0 116474s
 385636 195274 -191.25434   41   16          - -220.76941      -  15.0 116599s
 385683 195295 -197.09173   45    2          - -220.76863      -  15.0 116618s
 385865 195390 -182.15440   45   12          - -220.76406      -  15.0 116776s
 386096 195536 -190.18215   42    8          - -220.75134      -  15.0 116962s
 386149 195558 -181.42096   37    4          - -220.75128      -  15.0 117008s
 386303 195661 -195.20471   49   10          - -220.74268      -  15.0 117159s
 386353 195681 -166.35066   45    6          - -220.74263      -  15.0 117198s
 386494 195779 -186.71444   48    3          - -220.73670      -  15.0 117317s
 386749 195927 -198.58795   39   16          - -220.72568      -  15.0 117520s
 386796 195942 -183.02107   47    8          - -220.72353      -  15.0 117557s
 386855 195973 -210.79000   40   19          - -220.72319      -  15.0 117608s
 386963 196053 -199.77429   47    3          - -220.71980      -  15.0 117670s
 387061 196087 -183.13986   43    3          - -220.71715      -  15.0 117734s
 387116 196113 -186.58561   42    9          - -220.70649      -  15.0 117766s
 387159 196159 -189.06529   39    5          - -220.70536      -  15.0 117829s
 387222 196186 -196.90346   44   10          - -220.70509      -  15.0 117906s
 387413 196290 -196.82823   40   16          - -220.69596      -  15.0 118103s
 387476 196324 -178.71279   47    3          - -220.69553      -  15.0 118116s
 387536 196353 -198.27578   47    -          - -220.69448      -  15.0 118180s
 387649 196417 -177.20769   41    -          - -220.69123      -  15.0 118269s
 387843 196547 -186.19897   44    2          - -220.67952      -  15.0 118390s
 388095 196699 -198.42763   39   23          - -220.67360      -  15.0 118556s
 388165 196724 -196.02330   43    5          - -220.67147      -  15.0 118608s
 388336 196825 -177.47015   46    4          - -220.65850      -  15.0 118754s
 388493 196890 -197.75497   45    7          - -220.64842      -  15.0 118936s
 388545 196939 -187.78150   39    8          - -220.64617      -  15.0 118993s
 388596 196962 -176.61085   45    6          - -220.64567      -  15.0 119051s
 388715 197018 -185.62481   39   17          - -220.64130      -  15.0 119141s
 388777 197065 -181.61442   48    -          - -220.64078      -  15.0 119206s
 388825 197085 -175.80572   50    -          - -220.64066      -  15.0 119245s
 388880 197107 -192.46324   44    4          - -220.64039      -  15.0 119302s
 388993 197215 -178.97438   49    -          - -220.63014      -  15.0 119443s
 389232 197349 -192.01148   42    6          - -220.62144      -  15.0 119656s
 389352 197395 -197.43561   46    4          - -220.61851      -  15.0 119746s
 389459 197467 -166.02898   47    2          - -220.61249      -  15.0 119849s
 389513 197490 -183.96145   53    -          - -220.61157      -  15.0 119906s
 389696 197602 -190.81862   46    -          - -220.60463      -  15.0 120068s
 389901 197716 -204.14431   46    6          - -220.60040      -  15.0 120194s
 390131 197853 -200.43715   38   18          - -220.59064      -  15.0 120400s
 390257 197904 -189.19332   42    6          - -220.58970      -  15.0 120503s
 390378 197992 -202.65847   42   15          - -220.58056      -  15.0 120573s
 390563 198122 -195.67521   45    4          - -220.57004      -  15.0 120713s
 390686 198173 -193.05303   37   15          - -220.56805      -  15.0 120792s
 390823 198249 -191.46292   51    -          - -220.56278      -  15.0 120862s
 390922 198297 -178.23872   46    2          - -220.56192      -  15.0 120927s
 391035 198377 -191.40132   48   10          - -220.55623      -  15.0 121015s
 391253 198493 -199.22260   41   12          - -220.54229      -  15.0 121203s
 391358 198539 -190.04321   40   20          - -220.54045      -  15.0 121255s
 391457 198607 -191.69767   46    5          - -220.53449      -  15.0 121364s
 391517 198632 -181.21398   39    7          - -220.53447      -  15.0 121411s
 391697 198744 -196.03354   43    4          - -220.52800      -  15.0 121609s
 391922 198877 -195.67634   38    8          - -220.51780      -  15.0 121847s
 391975 198897 -187.83378   50    9          - -220.51673      -  15.0 121886s
 392079 198940 -174.10626   50    -          - -220.51165      -  15.0 122015s
 392131 199006 -173.82869   46    -          - -220.50873      -  15.0 122046s
 392373 199135 -183.76076   45    5          - -220.50137      -  15.0 122233s
 392608 199259 -175.65450   44    3          - -220.49291      -  15.0 122455s
 392648 199277 -199.27488   39   16          - -220.49267      -  15.0 122495s
 392779 199365 -170.59780   47    3          - -220.48207      -  15.0 122664s
 392877 199403 -195.00919   48    -          - -220.47990      -  15.0 122730s
 392996 199490 -180.84131   48    4          - -220.47546      -  15.0 122866s
 393208 199618 -181.70891   40   15          - -220.46943      -  15.0 123008s
 393270 199641 -173.44730   44    -          - -220.46901      -  15.0 123040s
 393444 199755 -193.89110   42    7          - -220.45929      -  15.0 123195s
 393499 199780 -183.42054   44   11          - -220.45811      -  15.0 123281s
 393654 199850 -179.12401   46    -          - -220.45279      -  15.0 123411s
 393835 199934 -176.01910   45    -          - -220.43907      -  15.0 123542s
 393889 199988 -161.81158   40    -          - -220.43713      -  15.0 123632s
 394080 200096 -182.97912   48    5          - -220.43313      -  15.0 123790s
 394131 200118 -183.31194   48    2          - -220.42790      -  15.0 123835s
 394192 200145 -197.82300   45    -          - -220.42725      -  15.0 123908s
 394317 200233 -174.14063   45    4          - -220.42429      -  15.0 123971s
 394378 200258 -189.30477   45    6          - -220.42257      -  15.0 124004s
 394536 200364 -172.20457   45    4          - -220.41659      -  15.0 124154s
 394595 200391 -189.90699   46    -          - -220.41360      -  15.0 124194s
 394759 200490 -188.29126   45    3          - -220.40911      -  15.0 124324s
 394805 200508 -200.53455   41   13          - -220.40904      -  15.0 124370s
 394851 200531 -205.75700   48    4          - -220.40774      -  15.0 124417s
 394954 200608 -182.15086   49    -          - -220.40265      -  15.0 124514s
 395173 200743 -197.58941   32   27          - -220.39242      -  15.0 124711s
 395224 200764 -174.21535   47    2          - -220.39062      -  15.0 124757s
 395403 200876 -177.10621   49    -          - -220.38293      -  15.0 124893s
 395453 200896 -168.71103   48    3          - -220.38273      -  15.0 124906s
 395605 200997 -190.85871   43    9          - -220.37749      -  15.0 125030s
 395665 201028 -186.75518   40   11          - -220.37663      -  15.0 125057s
 395845 201122 -192.88880   47    5          - -220.37056      -  15.0 125173s
 396068 201248 -193.07470   43    5          - -220.36320      -  15.0 125362s
 396123 201274 -182.79288   38   10          - -220.36095      -  15.0 125421s
 396293 201388 -191.59240   44   17          - -220.35223      -  15.0 125538s
 396422 201441 -194.97787   45   10          - -220.35003      -  15.0 125617s
 396548 201505 -173.40503   49    -          - -220.34817      -  15.0 125710s
 396756 201629 -187.89135   48    -          - -220.33777      -  15.0 125929s
 396817 201656 -183.33999   38    5          - -220.33650      -  15.0 125942s
 397011 201753 -164.57008   42    6          - -220.33047      -  15.0 126142s
 397063 201775 -191.41905   44    3          - -220.33012      -  15.0 126182s
 397179 201831 -195.78761   49    2          - -220.31866      -  15.0 126282s
 397226 201908 -202.66255   43    2          - -220.31777      -  15.0 126366s
 397289 201932 -190.66741   43   12          - -220.31580      -  15.0 126399s
 397351 201957 -175.35362   44    5          - -220.31568      -  15.0 126439s
 397474 202022 -174.78135   43    2          - -220.31024      -  15.0 126556s
 397520 202043 -181.25813   43    -          - -220.30849      -  15.0 126590s
 397650 202153 -190.47455   39    5          - -220.30208      -  15.0 126694s
 397893 202274 -206.22834   42   14          - -220.29426      -  15.0 126865s
 397952 202299 -191.83739   43    7          - -220.29426      -  15.0 126885s
 398142 202431 -194.95334   48    -          - -220.29355      -  15.0 127156s
 398196 202451 -168.32822   42    -          - -220.29098      -  15.0 127176s
 398357 202551 -174.34089   46    5          - -220.28304      -  15.0 127300s
 398560 202691 -191.21050   39   15          - -220.27756      -  15.0 127477s
 398618 202713 -186.12232   45    6          - -220.27734      -  15.0 127530s
 398786 202822 -175.48912   52    -          - -220.26545      -  15.0 127641s
 399018 202943 -186.62734   42    2          - -220.25413      -  15.0 127866s
 399060 202958 -175.38889   51    -          - -220.25347      -  15.0 127913s
 399113 202980 -187.12660   39   11          - -220.25321      -  15.0 127933s
 399216 203031 -183.37090   44   11          - -220.24775      -  15.0 128038s
 399287 203066 -186.47126   48    -          - -220.24584      -  15.0 128065s
 399343 203099 -193.32476   39   14          - -220.24581      -  15.0 128132s
 399465 203196 -181.92537   50    -          - -220.24248      -  15.0 128243s
 399521 203215 -189.78410   34   23          - -220.24081      -  15.0 128270s
 399573 203234 -168.88081   43    5          - -220.23440      -  15.0 128317s
 399628 203261 -175.54869   49    5          - -220.22881      -  15.0 128391s
 399693 203328 -191.61920   48    -          - -220.22698      -  15.0 128457s
 399752 203347 -187.42201   41   11          - -220.22569      -  15.0 128470s
 399806 203365 -179.31035   52    5          - -220.22444      -  15.0 128504s
 399859 203390 -178.32416   44    -          - -220.21749      -  15.0 128598s
 399916 203440 -194.11167   46    -          - -220.21447      -  15.0 128662s
 400160 203587 -175.64603   44    4          - -220.20650      -  15.0 128837s
 400339 203695 -183.76574   48    4          - -220.20038      -  15.1 129010s
 400583 203831 -171.33924   49    -          - -220.19416      -  15.1 129276s
 400827 203969 -206.56564   40    9          - -220.18822      -  15.1 129502s
 400885 203994 -162.52537   41    -          - -220.18692      -  15.1 129542s
 401061 204085 -194.62342   38   18          - -220.17488      -  15.1 129729s
 401152 204126 -182.61850   50    -          - -220.17395      -  15.1 129850s
 401238 204206 -191.70512   40   22          - -220.16740      -  15.1 129950s
 401300 204231 -190.62510   53    -          - -220.16603      -  15.1 130004s
 401466 204316 -191.96416   48    -          - -220.15720      -  15.1 130123s
 401522 204335 -156.72901   44    -          - -220.15651      -  15.1 130157s
 401700 204460 -184.15696   44    3          - -220.14809      -  15.1 130366s
 401751 204480 -204.98159   45   15          - -220.14603      -  15.1 130401s
 401858 204521 -178.62555   45    7          - -220.14125      -  15.1 130504s
 401917 204594 -187.65241   47    -          - -220.13947      -  15.1 130582s
 401959 204610 -163.03288   43    5          - -220.13933      -  15.1 130617s
 402094 204690 -184.14027   50    -          - -220.13168      -  15.1 130738s
 402323 204857 -195.92009   37   11          - -220.12449      -  15.1 130898s
 402566 204965 -180.04555   39   14          - -220.11412      -  15.1 131073s
 402629 204990 -159.12237   48    -          - -220.11335      -  15.1 131161s
 402808 205093 -193.37664   45    -          - -220.10376      -  15.1 131396s
 402856 205117 -166.65237   46    -          - -220.10328      -  15.1 131451s
 402912 205143 -172.32155   47    2          - -220.10305      -  15.1 131490s
 402989 205193 -179.96860   45    -          - -220.09463      -  15.1 131584s
 403239 205325 -190.04172   44    4          - -220.08803      -  15.1 131845s
 403289 205346 -179.87006   45    6          - -220.08636      -  15.1 131899s
 403398 205398 -208.91684   46    7          - -220.07924      -  15.1 132031s
 403459 205460 -184.23900   49    -          - -220.07709      -  15.1 132134s
 403510 205480 -172.84112   46    -          - -220.07548      -  15.1 132161s
 403620 205526 -192.04263   39    7          - -220.07119      -  15.1 132236s
 403668 205574 -173.00416   41    -          - -220.06789      -  15.1 132295s
 403777 205619 -180.09221   51    -          - -220.06612      -  15.1 132370s
 403890 205707 -212.54472   44    7          - -220.06216      -  15.1 132464s
 404101 205817 -185.14223   45    5          - -220.04982      -  15.1 132655s
 404156 205840 -178.02281   45    7          - -220.04877      -  15.1 132703s
 404220 205870 -202.75872   44    7          - -220.04820      -  15.1 132779s
 404327 205952 -199.97493   42   23          - -220.04047      -  15.1 132880s
 404379 205968 -191.97584   46    2          - -220.04045      -  15.1 132914s
 404546 206070 -189.47558   47    5          - -220.03180      -  15.1 133064s
 404594 206092 -204.28736   46    5          - -220.02799      -  15.1 133125s
 404695 206135 -176.07006   43   10          - -220.02108      -  15.1 133207s
 404751 206181 -180.11536   42    5          - -220.01893      -  15.1 133239s
 404956 206317 -188.00075   48    2          - -220.01168      -  15.1 133457s
 405056 206355 -190.88705   46    4          - -220.01093      -  15.1 133505s
 405174 206440 -175.95091   45    4          - -220.00782      -  15.1 133626s
 405398 206591 -191.26850   40   17          - -219.99421      -  15.1 133848s
 405615 206718 -177.56461   42    2          - -219.99151      -  15.1 134018s
 405744 206773 -184.02800   46    2          - -219.99097      -  15.1 134122s
 405870 206855 -176.89556   53    -          - -219.98303      -  15.1 134217s
 405978 206904 -195.99676   43   16          - -219.98107      -  15.1 134321s
 406097 206988 -152.49694   46    -          - -219.97637      -  15.1 134458s
 406230 207038 -193.44828   46    -          - -219.97169      -  15.1 134555s
 406275 207084 -169.44271   42    6          - -219.97162      -  15.1 134596s
 406332 207110 -201.99619   39    8          - -219.97131      -  15.1 134652s
 406516 207234 -193.20681   45   13          - -219.96502      -  15.1 134857s
 406580 207262 -178.75676   43    5          - -219.96465      -  15.1 134886s
 406640 207289 -171.64103   49    -          - -219.96407      -  15.1 134954s
 406756 207365 -175.56906   47    -          - -219.95851      -  15.1 135048s
 406924 207436 -197.27664   45   11          - -219.94910      -  15.1 135214s
 406984 207512 -201.14637   46    -          - -219.94746      -  15.1 135280s
 407229 207631 -187.26209   47    -          - -219.93805      -  15.1 135465s
 407263 207644 -179.06341   49    5          - -219.93738      -  15.1 135485s
 407305 207663 -161.13375   51    -          - -219.93648      -  15.1 135527s
 407392 207725 -179.56743   47    5          - -219.93215      -  15.1 135602s
 407639 207866 -210.03376   42   12          - -219.92047      -  15.1 135785s
 407700 207889 -176.10212   47    -          - -219.91906      -  15.1 135834s
 407866 207992 -183.13300   42    7          - -219.91329      -  15.1 135977s
 408097 208125 -207.07303   42    5          - -219.90142      -  15.1 136202s
 408154 208147 -198.15455   51    -          - -219.89950      -  15.1 136251s
 408294 208217 -194.10369   47    -          - -219.89471      -  15.1 136456s
 408517 208364 -175.07472   48    -          - -219.88254      -  15.1 136691s
 408573 208385 -205.55173   41   19          - -219.88122      -  15.1 136733s
 408702 208438 -171.58796   48    -          - -219.87480      -  15.1 136863s
 408761 208503 -192.98232   45    -          - -219.87418      -  15.1 136918s
 408816 208528 -189.85439   48    2          - -219.87398      -  15.1 137007s
 408980 208631 -188.95792   48    -          - -219.86701      -  15.1 137185s
 409089 208678 -162.84310   43   10          - -219.86350      -  15.1 137255s
 409211 208738 -205.39883   42   10          - -219.85606      -  15.1 137331s
 409415 208885 -204.24539   38   22          - -219.84799      -  15.1 137652s
 409479 208911 -200.33554   45   12          - -219.84677      -  15.1 137725s
 409671 209030 -163.06505   47    -          - -219.83781      -  15.1 137903s
 409732 209055 -198.19392   43   10          - -219.83651      -  15.1 137938s
 409787 209076 -173.94040   47    -          - -219.83483      -  15.1 138015s
 409845 209103 -167.71460   48    -          - -219.83153      -  15.1 138084s
 409905 209151 -178.36278   45    2          - -219.83149      -  15.1 138132s
 410118 209291 -159.88299   44    5          - -219.82016      -  15.1 138338s
 410179 209315 -185.88011   50    -          - -219.81964      -  15.1 138380s
 410344 209397 -198.51378   45    5          - -219.81215      -  15.1 138573s
 410391 209418 -179.80471   43    -          - -219.81154      -  15.1 138608s
 410444 209440 -205.49666   46    -          - -219.81105      -  15.1 138636s
 410552 209522 -185.31788   48    -          - -219.80593      -  15.1 138774s
 410724 209596 -176.36927   43   14          - -219.80291      -  15.1 138887s
 410775 209653 -196.53799   43   13          - -219.79793      -  15.1 138964s
 410828 209673 -176.62264   45    -          - -219.79510      -  15.1 139014s
 410931 209715 -181.01718   46    -          - -219.78248      -  15.1 139135s
 410991 209775 -192.39756   43    9          - -219.78149      -  15.1 139210s
 411172 209853 -174.96569   39   10          - -219.77591      -  15.1 139404s
 411230 209881 -176.43663   40    7          - -219.77551      -  15.1 139481s
 411277 209904 -176.92240   46    -          - -219.77431      -  15.1 139566s
 411401 209997 -187.02659   47    -          - -219.76183      -  15.1 139720s
 411524 210050 -187.19815   50    -          - -219.75881      -  15.1 139840s
 411638 210140 -175.87764   42    4          - -219.74865      -  15.1 139938s
 411748 210183 -193.81607   46    -          - -219.74847      -  15.1 140043s
 411858 210270 -162.39028   46    -          - -219.74352      -  15.1 140134s
 412084 210408 -205.42144   38   27          - -219.73653      -  15.1 140335s
 412133 210425 -185.43239   43    -          - -219.73525      -  15.1 140399s
 412195 210450 -184.39210   38   24          - -219.73494      -  15.1 140442s
 412323 210515 -200.29370   40   13          - -219.72592      -  15.1 140539s
 412371 210536 -171.76295   45    4          - -219.72470      -  15.1 140604s
 412455 210575 -190.54402   38   21          - -219.72158      -  15.1 140695s
 412503 210647 -177.23515   35   17          - -219.72000      -  15.1 140821s
 412738 210762 -199.64197   41   17          - -219.71481      -  15.1 141008s
 412797 210785 -197.85480   40   18          - -219.71354      -  15.1 141066s
 412973 210899 -196.36105   42   16          - -219.70401      -  15.1 141247s
 413029 210920 -170.82377   43    8          - -219.70387      -  15.1 141269s
 413205 211031 -198.95747   44    4          - -219.69767      -  15.1 141472s
 413251 211054 -151.02896   47    -          - -219.69629      -  15.1 141507s
 413410 211178 -191.39793   33   21          - -219.69199      -  15.1 141661s
 413536 211225 -199.20774   33   30          - -219.69049      -  15.1 141732s
 413644 211301 -181.72946   45    7          - -219.68439      -  15.1 141836s
 413805 211365 -183.29415   51    2          - -219.67941      -  15.1 141963s
 413868 211429 -182.49440   47    7          - -219.67911      -  15.1 142034s
 414080 211523 -189.16438   45    2          - -219.66779      -  15.1 142338s
 414192 211577 -188.99277   45    2          - -219.66490      -  15.1 142451s
 414305 211693 -179.58193   47   11          - -219.65693      -  15.1 142591s
 414423 211734 -177.77407   52    -          - -219.65481      -  15.1 142676s
 414541 211807 -173.78380   47    -          - -219.65302      -  15.1 142777s
 414586 211826 -196.25959   38   17          - -219.65240      -  15.1 142805s
 414731 211943 -179.26400   44    6          - -219.64308      -  15.1 142997s
 414795 211967 -198.62486   42    6          - -219.64195      -  15.1 143033s
 414969 212049 -168.67537   43    2          - -219.63276      -  15.1 143200s
 415177 212167 -190.33259   49    6          - -219.62451      -  15.1 143391s
 415337 212236 -166.81578   46    -          - -219.61654      -  15.1 143535s
 415398 212296 -170.69748   38    9          - -219.61488      -  15.1 143603s
 415623 212437 -210.24786   43   17          - -219.60535      -  15.1 143809s
 415690 212465 -182.98397   47    2          - -219.60526      -  15.1 143845s
 415866 212572 -204.62470   39   13          - -219.59510      -  15.1 144052s
 416041 212680 -193.27923   40   12          - -219.58989      -  15.1 144208s
 416287 212819 -196.24323   44   11          - -219.58452      -  15.1 144448s
 416512 212975 -167.95932   46    -          - -219.57351      -  15.1 144717s
 416717 213068 -197.80664   36   13          - -219.56675      -  15.1 144910s
 416886 213191 -190.46032   38   17          - -219.55024      -  15.2 145117s
 416941 213210 -192.08658   46   10          - -219.55018      -  15.2 145160s
 417004 213233 -193.16178   42    6          - -219.54978      -  15.2 145188s
 417119 213305 -171.17152   50    -          - -219.54411      -  15.2 145330s
 417175 213333 -192.77188   41   14          - -219.54397      -  15.2 145432s
 417363 213474 -167.88879   51    -          - -219.53262      -  15.2 145631s
 417579 213561 -192.86840   46    2          - -219.52566      -  15.2 145753s
 417638 213589 -167.98169   50    -          - -219.52495      -  15.2 145848s
 417801 213699 -182.04617   45    5          - -219.51439      -  15.2 145998s
 417959 213762 -190.84500   48    5          - -219.50568      -  15.2 146171s
 418006 213821 -183.08662   46    7          - -219.50412      -  15.2 146254s
*418111 167458              48    -185.9037304 -219.50248  18.1%  15.2 146334s
 418124 167459 -195.25544   51    5 -185.90373 -219.50049  18.1%  15.2 146342s
 418184 167468 -196.51439   44   12 -185.90373 -219.49459  18.1%  15.2 146356s

Cutting planes:
  Gomory: 2
  Lazy constraints: 32374

Explored 418253 nodes (6340222 simplex iterations) in 146396.04 seconds (6283.49 work units)
Thread count was 8 (of 8 available processors)

Solution count 1: -185.904

Solve interrupted
Best objective -1.859037304013e+02, best bound -2.194945865842e+02, gap 18.0690%

User-callback calls 929520, time in user-callback 142611.44 sec
(nothing, MathOptInterface.INTERRUPTED)



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

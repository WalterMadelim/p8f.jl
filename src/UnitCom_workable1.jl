import JuMP
import Gurobi
import LinearAlgebra
import Distributions
import Statistics
import Random
using Logging

# set a time limit for argZ to make sure we won't get stuck at this subprocedure
# sometimes, the global convergence is attained in a short time even if argZ might sometimes not solved to optimal
# However, in this 6-bus test case, the global convergence (exact convergence, tolerance = 0) is assured even we don't set a time limit
# my_timer reports about 2 minutes
# 30/12/24

GRB_ENV = Gurobi.Env()
Random.seed!(3)
macro my_timer(f) # t_elapsed = @my_timer(f())
    return quote
    t = time()
    $f
    t_elapsed = round((time() - t)/60; digits = 1)
    @info "Time: $t_elapsed minutes"
    t_elapsed
    end
end
function clear_log_file(file_string) return open(io -> nothing, file_string, "w") end # open, truncate, close
function set_logfile_for_model(file_string, model) # optional
    JuMP.set_attribute(model, "OutputFlag", 1)
    JuMP.set_attribute(model, "LogToConsole", 0)
    JuMP.set_attribute(model, "LogFile", file_string)
end
function ip(x, y) return LinearAlgebra.dot(x, y) end
function jvr6(x) return round(JuMP.value(x); digits = 6) end
function jdr6(x) return round( JuMP.dual(x); digits = 6) end
function GurobiDirectModel() return JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) end
function optimise(m) return (JuMP.optimize!(m); JuMP.termination_status(m)) end
function optimize_assert_optimal(m)
    status = optimise(m)
    status == JuMP.OPTIMAL || error("$status")
end
function brcs(v) return ones(T) * transpose(v) end
function gen_load_pattern(fLM)
    fL = length(fLM)
    fLP = rand(Distributions.Uniform(0.7 * fLM[1], 1.4 * fLM[1]), T, 1)
    for i in 2:fL
        fLP = [fLP rand(Distributions.Uniform(0.7 * fLM[i], 1.4 * fLM[i]), T)]
    end
    return fLP
end
Δ2 = Dict( # used in argmaxY
    "f" => Float64[],
    "x" => Matrix{Float64}[],
    "iY" => Int[],
    "b2" => Matrix{Float64}[]
)
function push_Δ2(f, x, iY, b2)
    push!(Δ2["f"],   f)
    push!(Δ2["x"],   x)
    push!(Δ2["iY"], iY)
    push!(Δ2["b2"], b2)
end

UPDTOL = 0.04 # If this param is inappropriate (e.g too small), program may get stuck at argZ
# argZ_TimeLimit = 15.0

G = 2
UT = DT = 3
PI = [0.45, 0.375, 0.5];
PS = [4.5, 4, 5.5];
ZS = [0, 0,   1]
CST = [0.63, 0.60, 0.72]
CG = [0.8, 0.68, 0.72]
CL_BASE = [1.6, 1.3776, 1.4442]
NW = [2, 3]
W = 2
NL = [4, 5, 6]
L = 3
LM = [4.0, 3.5, 3.0]
T = 24
(CL = brcs(CL_BASE); CL[end, :] *= 2);

MZ = gen_load_pattern(LM)
MY, yM, NY = let
    YABSMAX = [1.6, 2.3]
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
        ø = GurobiDirectModel()
        JuMP.set_silent(ø)
        JuMP.@variable(ø, c[1:NY] >= 1e-5)
        JuMP.@constraint(ø, sum(c) == 1)
        JuMP.@constraint(ø, sum(yM[:, :, i] * c[i] for i in 1:NY) .== MY)
        status = optimise(ø)
        status == JuMP.OPTIMAL && return true
        return false
    end
    @assert is_strict_in(MY, yM)
    MY, yM, size(yM)[3]
end;

argZ = GurobiDirectModel(); # 1️⃣  given x, Y, b2, which are all only in obj
# JuMP.set_attribute(argZ, "TimeLimit", argZ_TimeLimit);
# JuMP.set_silent(argZ);
set_logfile_for_model("argZ.log", argZ);
JuMP.@variable(argZ, argZ_Z[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]);
argZ_c_x, argZ_c_Y, argZ_c_b2, argZ_cn = let
    (JuMP.set_lower_bound.(argZ_Z, 0.9 * MZ); JuMP.set_upper_bound.(argZ_Z, 1.1 * MZ));
    JuMP.@variable(argZ, argZ_a[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]);
    JuMP.@constraint(argZ, argZ_a .>= argZ_Z .- MZ);
    JuMP.@constraint(argZ, argZ_a .>= MZ .- argZ_Z);
    JuMP.@constraint(argZ, [t = 1:T], sum(argZ_a[t, :] ./ (0.1 * MZ[t, :])) <= L/3);
    JuMP.@variable(argZ, Dbal[t = 1:T]);
    JuMP.@variable(argZ, Dvp[t = 1:T, w = 1:W] >= 0);
    JuMP.@variable(argZ, Dzt[t = 1:T, l = 1:L] >= 0);
    JuMP.@variable(argZ, Dvr[t = 1:T, g = 1:G+1] >= 0);
    JuMP.@variable(argZ, Dpi[t = 1:T, g = 1:G+1] >= 0);
    JuMP.@variable(argZ, Dps[t = 1:T, g = 1:G+1] >= 0);
    JuMP.@constraint(argZ, [t = 1:T, g = 1:G+1], Dps[t, g] - Dpi[t, g] - Dvr[t, g] + CG[g] == 0);
    JuMP.@constraint(argZ, [t = 1:T, g = 1:G+1], Dbal[t] - CG[g] + Dvr[t, g] >= 0);
    JuMP.@constraint(argZ, [t = 1:T, w = 1:W], Dvp[t, w] + Dbal[t] >= 0);
    JuMP.@constraint(argZ, [t = 1:T, l = 1:L], Dzt[t, l] - CL[t, l] - Dbal[t] >= 0);
    JuMP.@expression(argZ, argZ_c_x, brcs(PI) .* Dpi .- brcs(PS) .* Dps); # in x2
    JuMP.@expression(argZ, argZ_c_Y, -Dvp); # in x2
    JuMP.@expression(argZ, argZ_c_b2, -argZ_Z); # is beta2
    JuMP.@expression(argZ, argZ_cn, ip(CL .- Dzt, argZ_Z)); # the rest
    argZ_c_x, argZ_c_Y, argZ_c_b2, argZ_cn
end;
function set_argZ_objective(x, Y, b2) return JuMP.@objective(argZ, Max, argZ_cn + ip(argZ_c_x, x) + ip(argZ_c_Y, Y) + ip(argZ_c_b2, b2)) end
tr1 = GurobiDirectModel(); # 2️⃣
JuMP.set_silent(tr1);
JuMP.@variable(tr1, oΛ1); # is also the last term in obj_expr
JuMP.@variable(tr1, tr1_b1[eachindex(eachrow(MY)), eachindex(eachcol(MY))]); # objterm = ip(MY, tr1_b1)
JuMP.@variable(tr1, tr1_u[t = 1:T, g = 1:G+1], Bin);
JuMP.@variable(tr1, tr1_v[t = 1:T, g = 1:G+1], Bin);
JuMP.@variable(tr1, tr1_x[t = 1:T, g = 1:G+1], Bin);
let # 1st stage feasible region
    JuMP.@constraint(tr1, tr1_x .- vcat(transpose(ZS), tr1_x)[1:end-1, :] .== tr1_u .- tr1_v)
    JuMP.@constraint(tr1, [g = 1:G+1, t = 1:T-UT+1], sum(tr1_x[i, g] for i in t:t+UT-1) >= UT * tr1_u[t, g])
    JuMP.@constraint(tr1, [g = 1:G+1, t = T-UT+1:T], sum(tr1_x[i, g] - tr1_u[t, g] for i in t:T) >= 0)
    JuMP.@constraint(tr1, [g = 1:G+1, t = 1:T-DT+1], sum(1 - tr1_x[i, g] for i in t:t+DT-1) >= DT * tr1_v[t, g])
    JuMP.@constraint(tr1, [g = 1:G+1, t = T-DT+1:T], sum(1 - tr1_x[i, g] - tr1_v[t, g] for i in t:T) >= 0)
end;
JuMP.@objective(tr1, Min, ip(brcs(CST), tr1_u)); # ⏰ temporary obj to produce feasible solution
optimize_assert_optimal(tr1);
x = jvr6.(tr1_x)
Y = MY # ⏰ initial
tr2 = GurobiDirectModel(); # 3️⃣
JuMP.set_silent(tr2);
JuMP.@variable(tr2, oΛ2);
JuMP.@variable(tr2, tr2_b2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]); # objterm = ip(MZ, tr2_b2)
JuMP.@variable(tr2, tr2_x[eachindex(eachrow(x)), eachindex(eachcol(x))]);
JuMP.@variable(tr2, tr2_Y[eachindex(eachrow(Y)), eachindex(eachcol(Y))]);
tr2_cpx = JuMP.@constraint(tr2, tr2_x .== x);
tr2_cpY = JuMP.@constraint(tr2, tr2_Y .== Y);
function add_cut_for_oΛ2(cn, px, pY, pb) return JuMP.@constraint(tr2, oΛ2 >= cn + ip(px, tr2_x) + ip(pY, tr2_Y) + ip(pb, tr2_b2)) end
b2 = zero(MZ) # ⏰ a feasible solution
set_argZ_objective(x, Y, b2);
ini_argZ_con = JuMP.@constraint(argZ, argZ_Z .== MZ); # to generate bottom cut
clear_log_file("argZ.log");
optimize_assert_optimal(argZ);
argvZ_cn, argvZ_c_x, argvZ_c_Y, argvZ_c_b2 = JuMP.value(argZ_cn), jvr6.(argZ_c_x), jvr6.(argZ_c_Y), jvr6.(argZ_c_b2)
JuMP.delete.(argZ, ini_argZ_con);
add_cut_for_oΛ2(argvZ_cn, argvZ_c_x, argvZ_c_Y, -MZ) # use -MZ: a trick
JuMP.@objective(tr2, Min, ip(MZ, tr2_b2) + oΛ2); # ❄️ stable obj
optimize_assert_optimal(tr2); # if this fails, reconsider the bottom cut generation procedure
tr2v_cpx = jdr6.(tr2_cpx)
tr2v_cn = JuMP.objective_value(tr2) - ip(tr2v_cpx, x)
function add_cut_for_oΛ1(cn, px, pb) return JuMP.@constraint(tr1, oΛ1 >= cn + ip(px, tr1_x) + ip(pb, tr1_b1)) end
add_cut_for_oΛ1(tr2v_cn, tr2v_cpx, -MY) # -MY = -Y here
JuMP.@objective(tr1, Min, ip(brcs(CST), tr1_u) + ip(MY, tr1_b1) + oΛ1); # ❄️ stable obj
optimize_assert_optimal(tr1); # After this, boundedness is assured
#######################################################################################################################
function gen_quasi_iY(x, b1) # NOTE: it's ret(Y) is different from the underlying true argmaxY due to inexact Δ2. Nonetheless, the resultant ub is valid.
    fullVec = Vector{Float64}(undef, NY)
    for iY in Random.shuffle(1:NY)
        v = overrate_phi1xby(x, b1, iY)
        v == Inf && return iY
        fullVec[iY] = v
    end
    return findmax(fullVec)[2]
end
function overrate_phi1xby(x, b1, iY) return -ip(b1, yM[:, :, iY]) + overrate_psi(x, iY) end
function overrate_psi(x, iY)
    i_vec = findall(j -> j == iY, Δ2["iY"])
    isempty(i_vec) && return Inf
    R2 = length(i_vec)
    xV2, b2V2, fV2 = Δ2["x"][i_vec], Δ2["b2"][i_vec], Δ2["f"][i_vec]
    ø = GurobiDirectModel();
    JuMP.@variable(ø, b2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))])
    (JuMP.@variable(ø, λ[1:R2] >= 0); JuMP.@constraint(ø, sum(λ) == 1))
    JuMP.@constraint(ø, sum(b2V2[r] * λ[r] for r in 1:R2) .== b2)
    JuMP.@constraint(ø, sum( xV2[r] * λ[r] for r in 1:R2) .==  x)
    JuMP.@objective(ø, Min, ip(MZ, b2) + ip(fV2, λ))
    JuMP.set_silent(ø)
    optimise(ø) == JuMP.OPTIMAL || return Inf
    return JuMP.objective_value(ø)
end
function fast_get_Y_etc(x, b1)
    iY = gen_quasi_iY(x, b1)
    Y, oΛ1_hat = yM[:, :, iY], overrate_phi1xby(x, b1, iY)
    return iY, Y, oΛ1_hat
end
function endstage_manipulations(x, b1, iY, Y, b2, oΛ1_hat, oΛ2_check)
    argvZ_ub = JuMP.objective_bound(argZ)
    cn, cx, cY, cb = JuMP.value(argZ_cn), jvr6.(argZ_c_x), jvr6.(argZ_c_Y), jvr6.(argZ_c_b2)
    if -ip(b1, Y) + ip(MZ, b2) + argvZ_ub < oΛ1_hat - UPDTOL
        push_Δ2(argvZ_ub, x, iY, b2)
    else
        updVec[3] = false
    end
    if oΛ2_check < cn + ip(cx, x) + ip(cY, Y) + ip(cb, b2) - UPDTOL
        add_cut_for_oΛ2(cn, cx, cY, cb)
    else
        updVec[2] = false
    end
end
macro routine_code()
    return esc(quote
        JuMP.delete.(tr2, tr2_cpx)
        tr2_cpx = JuMP.@constraint(tr2, tr2_x .== x)
        iY, Y, oΛ1_hat = fast_get_Y_etc(x, b1) # 🥑✂️ 
        JuMP.delete.(tr2, tr2_cpY)
        tr2_cpY = JuMP.@constraint(tr2, tr2_Y .== Y)
        optimize_assert_optimal(tr2)
        b2 = jvr6.(tr2_b2) # 🥑
        oΛ2_check = JuMP.value(oΛ2) # ✂️
        set_argZ_objective(x, Y, b2)
        clear_log_file("argZ.log")
        optimize_assert_optimal(argZ)
        updVec .= true
        endstage_manipulations(x, b1, iY, Y, b2, oΛ1_hat, oΛ2_check)
        # reopt iY ???
        optimize_assert_optimal(tr2) # we reoptimize, because a new cut might be added
        tr2v_cpx = jdr6.(tr2_cpx)
        tr2v_cn = JuMP.objective_value(tr2) - ip(tr2v_cpx, x)
        pb1 = -Y # to emphasize
    end)
end
updVec = trues(3) # [1] for Λ1, [2] for Λ2, [3] for Δ2
ubsV = [Inf]
xsV = [x]
b1sV = [zero(MY)]
function main_loop()
    while true
        global tr2_cpx, tr2_cpY
        optimize_assert_optimal(tr1)
        lb = JuMP.objective_bound(tr1)
        x, b1 = jvr6.(tr1_x), jvr6.(tr1_b1)
        oΛ1_check = JuMP.value(oΛ1) # ✂️
        same_part = JuMP.objective_value(tr1) - oΛ1_check
        @routine_code()
        if oΛ1_check < tr2v_cn + ip(tr2v_cpx, x) + ip(pb1, b1) - UPDTOL
            add_cut_for_oΛ1(tr2v_cn, tr2v_cpx, pb1)
        else
            updVec[1] = false
        end
        ub = same_part + oΛ1_hat
        if ub < ubsV[1] # a superior feasible solution is found
            ubsV[1] = ub
            xsV[1] = x
            b1sV[1] = b1
        end
        @info "updVec = $updVec, lb = $lb | $(ubsV[1]) ≤ $ub = ub"
        all(updVec .== false) && break
    end
end
@my_timer main_loop()

# ...
# [ Info: updVec = Bool[0, 0, 1], lb = 1.2300014052885944 | 2072.7921895574495 ≤ Inf = ub
# [ Info: updVec = Bool[0, 0, 1], lb = 1.2300014052885944 | 2072.7921895574495 ≤ Inf = ub
# [ Info: updVec = Bool[0, 0, 1], lb = 1.2300014052885944 | 2072.7921895574495 ≤ Inf = ub
# [ Info: updVec = Bool[0, 0, 0], lb = 1.2300014052885944 | 1.2300620382061696 ≤ 1.2300620382061696 = ub
# [ Info: Time: 2.0 minutes


import JuMP
import Gurobi
import LinearAlgebra
import Distributions
import Statistics
import Random
using Logging

# set a time limit for argZ to make sure we won't get stuck at this subprocedure
# in this test case, the global convergence is attained in a short time even if argZ might sometimes not solved to optimal
# 25/12/24

GRB_ENV = Gurobi.Env()
Random.seed!(3)
function ip(x, y) return LinearAlgebra.dot(x, y) end
function GurobiDirectModel() return JuMP.direct_model(Gurobi.Optimizer(GRB_ENV)) end
function optimise(m) return (JuMP.optimize!(m); JuMP.termination_status(m)) end
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
macro optimize_assert_optimal(m)
    return esc(quote
        model_status = optimise($m)
        model_status == JuMP.OPTIMAL || error("$model_status")
    end)
end

UPDTOL = 0.04 # If this param is inappropriate (e.g too small), program may get stuck at argZ
B1BND, B2BND = 6.0, 3.6

G = 2
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
    YABSMAX = [3.5, 2.2]
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

UT = DT = 3
tr1 = GurobiDirectModel();
JuMP.set_silent(tr1);
JuMP.@variable(tr1, oΛ1); # is also the last term in obj_expr
JuMP.@variable(tr1, tr1_b1[eachindex(eachrow(MY)), eachindex(eachcol(MY))]);
JuMP.@expression(tr1, tr1_obj_expr_b1, ip(MY, tr1_b1));
JuMP.@variable(tr1, 0 <= tr1_u[t = 1:T, g = 1:G+1] <= 1);
JuMP.@expression(tr1, tr1_obj_expr_u, ip(brcs(CST), tr1_u));
JuMP.@variable(tr1, 0 <= tr1_v[t = 1:T, g = 1:G+1] <= 1);
JuMP.@variable(tr1, 0 <= tr1_x[t = 1:T, g = 1:G+1] <= 1);
let # 1st stage feasible region
    JuMP.@constraint(tr1, tr1_x .- vcat(transpose(ZS), tr1_x)[1:end-1, :] .== tr1_u .- tr1_v)
    JuMP.@constraint(tr1, [g = 1:G+1, t = 1:T-UT+1], sum(tr1_x[i, g] for i in t:t+UT-1) >= UT * tr1_u[t, g])
    JuMP.@constraint(tr1, [g = 1:G+1, t = T-UT+1:T], sum(tr1_x[i, g] - tr1_u[t, g] for i in t:T) >= 0)
    JuMP.@constraint(tr1, [g = 1:G+1, t = 1:T-DT+1], sum(1 - tr1_x[i, g] for i in t:t+DT-1) >= DT * tr1_v[t, g])
    JuMP.@constraint(tr1, [g = 1:G+1, t = T-DT+1:T], sum(1 - tr1_x[i, g] - tr1_v[t, g] for i in t:T) >= 0)
end;
(JuMP.set_lower_bound.(tr1_b1, -B1BND); JuMP.set_upper_bound.(tr1_b1,  B1BND));
JuMP.@objective(tr1, Min, tr1_obj_expr_u + tr1_obj_expr_b1); # ⏰ initial obj
@optimize_assert_optimal(tr1);
x = JuMP.value.(tr1_x); # 🥑
b1 = JuMP.value.(tr1_b1); # 🥑
(iY = rand(1:NY); Y = yM[:, :, iY]); # 🥑
tr2 = GurobiDirectModel();
JuMP.set_silent(tr2);
JuMP.@variable(tr2, oΛ2);
JuMP.@variable(tr2, tr2_b2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]);
JuMP.@variable(tr2, tr2_x[eachindex(eachrow(x)), eachindex(eachcol(x))]);
JuMP.@variable(tr2, tr2_Y[eachindex(eachrow(Y)), eachindex(eachcol(Y))]);
(JuMP.set_lower_bound.(tr2_b2, -B2BND); JuMP.set_upper_bound.(tr2_b2,  B2BND));
JuMP.@objective(tr2, Min, ip(MZ, tr2_b2)); # ⏰ initial obj
@optimize_assert_optimal(tr2);
b2 = JuMP.value.(tr2_b2); # 🥑
macro initial_endstage_code()
    return esc(quote
        argvZ_obj  = JuMP.objective_bound(argZ)
        argvZ_cn, argvZ_c_x, argvZ_c_Y, argvZ_c_b2 = JuMP.value(argZ_cn), JuMP.value.(argZ_c_x), JuMP.value.(argZ_c_Y), JuMP.value.(argZ_c_b2)
        push_Δ2(argvZ_obj, x, iY, b2) # the upper bound is valid since it is the objBound of a Max Program
        # the lower-bounding-cut generated for the previous model `tr2` is valid since the solution is feasible
        JuMP.@constraint(tr2, oΛ2 >= argvZ_cn + ip(argvZ_c_x, tr2_x) + ip(argvZ_c_Y, tr2_Y) + ip(argvZ_c_b2, tr2_b2)); # the 1st oΛ2 cut 
    end)
end
begin # initialize the end stage subprocedure
    argZ = GurobiDirectModel(); # given x, Y, b2, which are all only in obj
    JuMP.set_attribute(argZ, "TimeLimit", 15.0);
    JuMP.set_silent(argZ);
    JuMP.@variable(argZ, argZ_Z[eachindex(eachrow(b2)), eachindex(eachcol(b2))]);
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
        JuMP.@expression(argZ, argZ_cn, ip(CL .- Dzt, argZ_Z));
        JuMP.@expression(argZ, argZ_c_x, brcs(PI) .* Dpi .- brcs(PS) .* Dps);
        JuMP.@expression(argZ, argZ_c_Y, -Dvp);
        JuMP.@expression(argZ, argZ_c_b2, -argZ_Z);
    JuMP.@objective(argZ, Max, argZ_cn + ip(argZ_c_x, x) + ip(argZ_c_Y, Y) + ip(argZ_c_b2, b2));
    @optimize_assert_optimal(argZ)
    @initial_endstage_code()
end;

JuMP.@objective(tr2, Min, ip(MZ, tr2_b2) + oΛ2); # ❄️ stable obj
tr2_cpx = JuMP.@constraint(tr2, tr2_x .== x);
tr2_cpY = JuMP.@constraint(tr2, tr2_Y .== Y);
while true
    (JuMP.delete_lower_bound.(tr2_b2); JuMP.delete_upper_bound.(tr2_b2));
    optimise(tr2) == JuMP.OPTIMAL && (@info "🥑 we can generate cut for oΛ1 now"; break)
    (JuMP.set_lower_bound.(tr2_b2, -B2BND); JuMP.set_upper_bound.(tr2_b2,  B2BND));
    @optimize_assert_optimal(tr2)
    b2 = JuMP.value.(tr2_b2) # 🥑
    JuMP.@objective(argZ, Max, argZ_cn + ip(argZ_c_x, x) + ip(argZ_c_Y, Y) + ip(argZ_c_b2, b2));
    @optimize_assert_optimal(argZ)
    @initial_endstage_code()
end

tr2v_cpx = JuMP.dual.(tr2_cpx)
tr2v_cn = JuMP.objective_value(tr2) - ip(tr2v_cpx, x)
pb1 = -Y # to emphasize
JuMP.@constraint(tr1, oΛ1 >= tr2v_cn + ip(tr2v_cpx, tr1_x) + ip(pb1, tr1_b1)); # the 1st oΛ1 cut
JuMP.@expression(tr1, tr1_obj_expr, tr1_obj_expr_u + tr1_obj_expr_b1 + oΛ1); # ❄️ stable obj
JuMP.@objective(tr1, Min, tr1_obj_expr);
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

macro mature_endstage_code()
    return esc(quote
        argvZ_obj  = JuMP.objective_bound(argZ)
        argvZ_cn, argvZ_c_x, argvZ_c_Y, argvZ_c_b2 = JuMP.value(argZ_cn), JuMP.value.(argZ_c_x), JuMP.value.(argZ_c_Y), JuMP.value.(argZ_c_b2)
        if -ip(b1, Y) + ip(MZ, b2) + argvZ_obj < oΛ1_hat - UPDTOL
            push_Δ2(argvZ_obj, x, iY, b2)
        else
            updVec[3] = false
        end
        if oΛ2_check < argvZ_cn + ip(argvZ_c_x, x) + ip(argvZ_c_Y, Y) + ip(argvZ_c_b2, b2) - UPDTOL
            JuMP.@constraint(tr2, oΛ2 >= argvZ_cn + ip(argvZ_c_x, tr2_x) + ip(argvZ_c_Y, tr2_Y) + ip(argvZ_c_b2, tr2_b2));
        else
            updVec[2] = false
        end
    end)
end
macro refresh_rhs_of_tr2()
    return esc(quote
        JuMP.delete.(tr2, tr2_cpx)
        JuMP.delete.(tr2, tr2_cpY)
        tr2_cpx = JuMP.@constraint(tr2, tr2_x .== x)
        tr2_cpY = JuMP.@constraint(tr2, tr2_Y .== Y)
    end)
end
macro routine_code()
    return esc(quote
        iY, Y, oΛ1_hat = fast_get_Y_etc(x, b1) # 🥑✂️ 
        @refresh_rhs_of_tr2()
        @optimize_assert_optimal(tr2)
        b2 = JuMP.value.(tr2_b2) # 🥑
        oΛ2_check = JuMP.value(oΛ2) # ✂️
        JuMP.@objective(argZ, Max, argZ_cn + ip(argZ_c_x, x) + ip(argZ_c_Y, Y) + ip(argZ_c_b2, b2)); # update obj
        let argZ_status = optimise(argZ)
            if argZ_status != JuMP.OPTIMAL
                argZ_status != JuMP.TIME_LIMIT && error("argZ: $argZ_status")
                @info "argZ: JuMP.TIME_LIMIT"
            end
        end
        updVec .= true
        @mature_endstage_code() # <- oΛ2_check, oΛ1_hat
        # updVec[3] && ?? reopt Y ??
        @optimize_assert_optimal(tr2)
        tr2v_cpx = JuMP.dual.(tr2_cpx)
        tr2v_cn = JuMP.objective_value(tr2) - ip(tr2v_cpx, x)
        pb1 = -Y # to emphasize
    end)
end
updVec = trues(3) # [1] for Λ1, [2] for Λ2, [3] for Δ2
while true
    global tr2_cpx, tr2_cpY
    (JuMP.delete_lower_bound.(tr1_b1); JuMP.delete_upper_bound.(tr1_b1));
    tr1_status = optimise(tr1)
    tr1_status == JuMP.OPTIMAL && (@info "🥑 b1 now has auto-bound"; break)
    (JuMP.set_lower_bound.(tr1_b1, -B1BND); JuMP.set_upper_bound.(tr1_b1,  B1BND));
    @optimize_assert_optimal(tr1)
    lb = JuMP.objective_value(tr1)
    x, b1 = JuMP.value.(tr1_x), JuMP.value.(tr1_b1) # 🥑
    oΛ1_check = JuMP.value(oΛ1) # ✂️
    @routine_code()
    if oΛ1_check < tr2v_cn + ip(tr2v_cpx, x) + ip(pb1, b1) - 1e-13 # at initial stage, we want looser updates
        JuMP.@constraint(tr1, oΛ1 >= tr2v_cn + ip(tr2v_cpx, tr1_x) + ip(pb1, tr1_b1));
    else
        updVec[1] = false
    end
    @info "updVec = $updVec, lb = $lb"
    all(updVec .== false) && error("Initialization fails before b1 has auto-bound, consider enlarging B1BND")
end

function my_callback_function(cb_data, cb_where::Cint)
    jvcb(x) = JuMP.callback_value(cb_data, x)
    cb_where == Gurobi.GRB_CB_MIPSOL || return
    lb = let resultP = Ref{Cdouble}()
        Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJBND, resultP)
        resultP[]
    end
    Gurobi.load_callback_variable_primal(cb_data, cb_where)
    x, b1 = jvcb.(tr1_x), jvcb.(tr1_b1) # 🥑
    oΛ1_check = jvcb(oΛ1) # ✂️ 
    while true # must generate violating cut unless all saturation
        global tr2_cpx, tr2_cpY
        @routine_code()
        if oΛ1_check < tr2v_cn + ip(tr2v_cpx, x) + ip(pb1, b1) - UPDTOL
            JuMP.MOI.submit(tr1, JuMP.MOI.LazyConstraint(cb_data),
                JuMP.@build_constraint(oΛ1 >= tr2v_cn + ip(tr2v_cpx, tr1_x) + ip(pb1, tr1_b1))
            )
            mywr("$lb")
            return
        else
            updVec[1] = false
        end
        if all(updVec .== false)
            abs_gap = oΛ1_hat - oΛ1_check
            mywr("all S⋅A⋅T, gap = $abs_gap = $oΛ1_hat - $oΛ1_check, UPDTOL = $UPDTOL, lb = $lb")
            return
        end
    end
end

(JuMP.delete_lower_bound.([tr1_u tr1_v tr1_x]); JuMP.delete_upper_bound.([tr1_u tr1_v tr1_x]); JuMP.set_binary.([tr1_u tr1_v tr1_x]));
JuMP.MOI.set(tr1, JuMP.MOI.RawOptimizerAttribute("LazyConstraints"), 1)
JuMP.MOI.set(tr1, Gurobi.CallbackFunction(), my_callback_function)
function mywr(s) return open(f -> println(f, s), "output.log", "a") end # to seperate from Gurobi's warning
mywr("A pristine output.log")
@optimize_assert_optimal(tr1)

lb = JuMP.objective_bound(tr1);
mywr("After quit optimization, lb = $lb")








 
    




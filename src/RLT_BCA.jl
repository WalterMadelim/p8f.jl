import JuMP, Gurobi; GRB_ENV = Gurobi.Env(); import LinearAlgebra.diag as diag
import JuMP.value as 𝒗
using Logging; global_logger(ConsoleLogger(Logging.Debug))

# 9-by-10 modest scale case
# The example was (derived) from CCG2012
# the subproblem was approximately solved

COT = 1e-4
M, N = 9, 10; ♭ = N/2
𝙺, C_y, C_z, 𝚃, 𝙱, 𝚅 = let
    𝚃 = [2.75 12.33 8.58 0.67 13.58 7.33 9.42 5.67 9.83 4.0; 1.08 5.67 6.5 14.42 0.67 4.0 12.75 6.08 4.0 9.42; 4.42 6.92 12.75 4.42 11.5 7.75 6.92 7.33 13.17 11.08; 4.0 11.08 4.0 13.58 12.33 5.25 4.83 4.0 11.5 14.0; 3.17 1.08 11.92 8.58 11.5 1.08 7.75 1.5 5.67 3.17; 3.58 2.75 11.5 2.75 2.33 14.42 4.83 5.67 3.17 10.25; 11.92 9.83 11.92 2.33 2.75 13.17 1.92 2.33 12.75 8.17; 4.0 14.0 4.0 12.33 1.08 9.83 1.92 10.67 5.67 3.58; 14.0 9.83 12.33 14.0 7.33 11.92 8.58 2.75 12.33 10.67]
    C_z = 2 * [4.0, 2.5, 4.0, 4.0, 2.5, 2.5, 3.0, 3.5, 3.0]
    C_y = [10, 12, 10, 9, 12, 9, 12, 9, 11] .* C_z
    𝙺 = [224, 224, 221, 223, 221, 226, 224, 219, 218]
    𝙱 = [394, 70, 238, 6, 225, 383, 65, 136, 13, 83]
    𝚅 = [40, 40, 40, 30, 10, 10, 40, 10, 30, 30]
    𝙺, C_y, C_z, 𝚃, 𝙱, 𝚅
end; 𝚁 = 20 * C_z;

function sm(p, x) return sum(p .* x) end;
function optimise(m)
    JuMP.optimize!(m)
    return (
        JuMP.termination_status(m),
        JuMP.primal_status(m),
        JuMP.dual_status(m)
    )
end;
macro set_objective_function(m, f) return esc(:(JuMP.set_objective_function($m, JuMP.@expression($m, $f)))) end;
function Model(name)
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    JuMP.MOI.set(m, JuMP.MOI.Name(), name)
    s = name[end-1]
    s == 'm' && JuMP.set_objective_sense(m, JuMP.MIN_SENSE)
    s == 'M' && JuMP.set_objective_sense(m, JuMP.MAX_SENSE)
    last(name) == 's' && JuMP.set_silent(m)
    m
end;
function solve_to_normality(m)
    t, p, d = optimise(m)
    name = JuMP.name(m)
    t == JuMP.OPTIMAL || error("$name: $(string(t))")
    p == JuMP.FEASIBLE_POINT || error("$name: (primal) $(string(p))")
    (name[end-2] == 'l' && d != p) && error("$name: (dual) $(string(d))")
    return JuMP.result_count(m)
end;
function stage2_problem_primal(z, d)
    st2 = Model("st2_primal_lms")
    JuMP.@variable(st2, x[i = 1:M, j = 1:N] ≥ 0)
    JuMP.@variable(st2, ζ[i = 1:M] ≥ 0) # an expensive substitute for `z`
    JuMP.@constraint(st2, 𝙸[i = 1:M], z[i] + ζ[i]  ≥ sum(x[i, :]))
    JuMP.@constraint(st2, 𝙹[j = 1:N], sum(x[:, j]) ≥ d[j]        )
    @set_objective_function(st2, sm(𝚃, x) + sm(𝚁, ζ))
    solve_to_normality(st2)
    return JuMP.objective_value(st2)
end;
function g2d(g) return 𝙱 .+ 𝚅 .* g end;
function set_𝙻_obj(z) return JuMP.set_objective_coefficient.(𝙻, ı, -z) end # a ★ Partial ★ setting
function set_𝙳_obj(z, g) return JuMP.set_objective_coefficient.(𝙳, [𝙹; 𝙸], [g2d(g); -z]) end
function set_𝙶_obj(z, I, J) return @set_objective_function(𝙶, sm(J, g2d(𝚐)) - sm(I, z)) end # also involve a constant
𝙼 = Model("st1_bms"); JuMP.@variable(𝙼, 𝚘); 𝙼ˈs = (ul = [Inf, -Inf], z = Vector{Float64}(undef, M))
JuMP.@variables(𝙼, begin 𝚢[1:M], Bin; 𝚣[1:M] ≥ 0 end); JuMP.@constraint(𝙼, 𝚣 .≤ 𝙺 .* 𝚢); JuMP.set_objective_coefficient.(𝙼, [𝚢; 𝚣], [C_y; C_z]);
𝙳 = Model("st2_dual_lMs");
JuMP.@variable(𝙳, 0 <= 𝙸[i = 1:M] <= 𝚁[i]);
JuMP.@variable(𝙳, 0 <= 𝙹[j = 1:N]); JuMP.@constraint(𝙳, [j = 1:N, i = 1:M], 
                       𝙹[j] <= 𝚃[i, j] + 𝙸[i]
); 𝙶 = Model("opt_g_lMs"); JuMP.@variable(𝙶, 0 ≤ 𝚐[n = 1:N] ≤ 1); JuMP.@constraint(𝙶, sum(𝚐) ≤ ♭);
𝙻 = Model("lpr_lMs");      JuMP.@variable(𝙻, 0 ≤ 𝐠[n = 1:N] ≤ 1); JuMP.@constraint(𝙻, sum(𝐠) ≤ ♭);
JuMP.@variable(𝙻, 0 <= ı[i = 1:M] <= 𝚁[i]);
JuMP.@variable(𝙻, 0 <= ȷ[j = 1:N]); JuMP.@constraint(𝙻, 𝐌[i = 1:M, j = 1:N],
                       ȷ[j] <= 𝚃[i, j] + ı[i]
);
JuMP.@variable(𝙻, 0 ≤ 𝐠ȷ[nj = 1:N]) # only diagonal, as it occurs in Obj
JuMP.@variable(𝙻, 0 ≤ 𝐠Xı[n = 1:N, i = 1:M]) # needed to specify the upper_bound of `𝐠ȷ`
JuMP.set_objective_coefficient.(𝙻, [ȷ; 𝐠ȷ], [𝙱; 𝚅]) # Obj: the ★ Fixed ★ part
JuMP.@constraint(𝙻, 𝐠ȷ .<= ȷ) # (𝐠 ≤ 1) ⨉ (0 <= ȷ)  [a part of 1/10] ✅
JuMP.@constraint(𝙻, [n = 1:N, i = 1:M], 𝐠ȷ[n] <= 𝐠[n] * 𝚃[i, n] + 𝐠Xı[n, i]) # (g ≥ 0) ⨉ 𝐌 [a part of 1/10] ✅
JuMP.@constraint(𝙻, [n = 1:N, i = 1:M], 𝐠Xı[n, i] <= 𝚁[i] * 𝐠[n]) # (g ≥ 0) ⨉ (𝙸 <= 𝚁) [1/10] 🟣
JuMP.@constraint(𝙻, [n = 1:N, i = 1:M], 𝐠Xı[n, i] <= ı[i]) # (𝐠 ≤ 1) ⨉ (0 <= 𝙸) [1/10] 🟣
JuMP.@constraint(𝙻, [i = 1:M], sum(𝐠Xı[:, i]) <= ♭ * ı[i]) # (sum(𝐠) ≤ ♭) ⨉ (0 <= 𝙸) [1/10]
#  🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 
solve_to_normality(𝙼); z = 𝒗.(𝚣); # ✂️ the initial solve of `st1_bms`, No cut yet, and lb is invalid
set_𝙻_obj(z); Lˈs = (ul = [Inf, -Inf], g = Vector{Float64}(undef, N)); solve_to_normality(𝙻);
Lˈs.ul[1], Lˈs.g[:] = JuMP.objective_bound(𝙻), 𝒗.(𝐠) # setting upper bound is one-off, get a heuristic primal solution
while true
    set_𝙳_obj(z, Lˈs.g); solve_to_normality(𝙳)
    set_𝙶_obj(z, 𝒗.(𝙸), 𝒗.(𝙹)); solve_to_normality(𝙶);
    lb = JuMP.objective_value(𝙶)
    if lb > Lˈs.ul[2]
        Lˈs.ul[2], Lˈs.g[:] = lb, 𝒗.(𝚐)
    else
        @debug "After RLT_BCA, (sub)lb = $(Lˈs.ul[2]) < $(Lˈs.ul[1]) = (sub)ub"
        break
    end
end
𝙼ˈs.ul[1], 𝙼ˈs.z[:] = JuMP.objective_value(𝙼) + Lˈs.ul[1], z # The feasible value and solution at initialization
JuMP.@expression(𝙼, common_expr, JuMP.objective_function(𝙼)); JuMP.set_objective_coefficient(𝙼, 𝚘, 1); # 1️⃣ a one-shot turn on
set_𝙳_obj(z, Lˈs.g); solve_to_normality(𝙳); JuMP.@constraint(𝙼, 𝚘 >= sm(𝒗.(𝙹), g2d(Lˈs.g)) - sm(𝒗.(𝙸), 𝚣))
#  🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 🥎 
solve_to_normality(𝙼); # ✅ This is indeed a regular solve
𝙼ˈs.ul[2], z = JuMP.objective_bound(𝙼), 𝒗.(𝚣);
@info "The 1st global lb = $(𝙼ˈs.ul[2]) < $(𝙼ˈs.ul[1]) = ub, then we start main loop"
for ite = 1:999999
    global z, Lˈs
    set_𝙻_obj(z); Lˈs = (ul = [Inf, -Inf], g = Vector{Float64}(undef, N)); solve_to_normality(𝙻);
    Lˈs.ul[1], Lˈs.g[:] = JuMP.objective_bound(𝙻), 𝒗.(𝐠) # setting upper bound is one-off, get a heuristic primal solution
    ub = 𝒗(common_expr) + Lˈs.ul[1]
    if ub < 𝙼ˈs.ul[1]
        𝙼ˈs.ul[1], 𝙼ˈs.z[:] = ub, z
        @info "ite = $ite ▶ $(𝙼ˈs.ul[2]) < $(𝙼ˈs.ul[1]) ✪"
    else
        @info "ite = $ite ▶ $(𝙼ˈs.ul[2]) < $(𝙼ˈs.ul[1]) = ubs | ub = $ub"
    end
    while true
        set_𝙳_obj(z, Lˈs.g); solve_to_normality(𝙳);
        set_𝙶_obj(z, 𝒗.(𝙸), 𝒗.(𝙹)); solve_to_normality(𝙶);
        lb = JuMP.objective_value(𝙶)
        if lb > Lˈs.ul[2]
            Lˈs.ul[2], Lˈs.g[:] = lb, 𝒗.(𝚐)
        else
            @debug "After RLT_BCA, (sub)lb = $(Lˈs.ul[2]) < $(Lˈs.ul[1]) = (sub)ub"
            break
        end
    end
    set_𝙳_obj(z, Lˈs.g); solve_to_normality(𝙳); cn, pz = sm(𝒗.(𝙹), g2d(Lˈs.g)), -𝒗.(𝙸)
    if 𝒗(𝚘) < cn + sm(pz, z) - COT
        JuMP.@constraint(𝙼, 𝚘 >= cn + sm(pz, 𝚣))
    else
        @info "Quit the algorithm due to Benders cut saturation"
        break
    end
    solve_to_normality(𝙼); # ✅ This is indeed a regular solve
    𝙼ˈs.ul[2], z = JuMP.objective_bound(𝙼), 𝒗.(𝚣);
end

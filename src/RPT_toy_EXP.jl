import JuMP
import LinearAlgebra.dot as ip
import MosekTools

# Toy example in §4
# 7/3/25

function myModel(str)
    m = JuMP.Model(() -> MosekTools.Optimizer())
    occursin("m", str) && JuMP.set_objective_sense(m, JuMP.MIN_SENSE)
    occursin("M", str) && JuMP.set_objective_sense(m, JuMP.MAX_SENSE)
    occursin("s", str) && JuMP.set_silent(m)
    m
end
function optimise(m) return (JuMP.optimize!(m); JuMP.termination_status(m)) end
macro opt_ass_opt(m)
    name_str = string(m, ": ")
    return esc(quote
        let status = optimise($m)
            status == JuMP.OPTIMAL || error($name_str * string(status))
        end
    end)
end
macro set_objective_function(m, f) return esc(:(JuMP.set_objective_function($m, JuMP.@expression($m, $f)))) end # ✅ a handy macro

# p, n are epi-variables
# Λ := x * p
# V := x * n
E = 2 + exp(-1);
CR = myModel("m"); # Convex Relaxation
JuMP.@variable(CR, x[1:3] <= 10); # primal linear
JuMP.@constraint(CR, x[1] + x[2] + 1 >= 0); # primal linear
JuMP.@constraint(CR, [x[2] - x[3], 1, x[1]] in JuMP.MOI.ExponentialCone()); # primal convex
JuMP.@variable(CR, p1); JuMP.@variable(CR, p3); JuMP.@variable(CR, n1); JuMP.@variable(CR, n2);
JuMP.@constraint(CR, n1 + n2 <= E); # primal linear
JuMP.@constraint(CR, [x[1], 1, p1] in JuMP.MOI.ExponentialCone()); JuMP.@constraint(CR, [-x[1], 2, n1] in JuMP.MOI.ExponentialCone()); # primal convex
JuMP.@constraint(CR, [x[3], 1, p3] in JuMP.MOI.ExponentialCone()); JuMP.@constraint(CR, [-x[2], 2, n2] in JuMP.MOI.ExponentialCone()); # primal convex
JuMP.@variable(CR, Λ11); JuMP.@variable(CR, Λ13); JuMP.@variable(CR, Λ21); JuMP.@variable(CR, Λ23);
@set_objective_function(CR, ip([3, -3, 3.], x) + p1 + p3 + Λ11 + Λ13 + Λ21 + Λ23);
# 🥎 convex * convex: omit p * p, p * n, n * n
JuMP.@variable(CR, X11); JuMP.@variable(CR, V11); JuMP.@variable(CR, V12);
JuMP.@constraint(CR, [x[1] + x[2] - x[3], 1, Λ11]         in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [x[2], 1, Λ13]                       in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [2 * x[2] - 2 * x[3] - x[1], 2, V11] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [x[2] - 2 * x[3], 2, V12]            in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [2 * (x[2] - x[3]), 1, X11]          in JuMP.MOI.ExponentialCone());
# 🥎 linear * linear: omit n * n
JuMP.@variable(CR, X22); JuMP.@variable(CR, X23); JuMP.@variable(CR, X33);
JuMP.@variable(CR, V21); JuMP.@variable(CR, V22); JuMP.@variable(CR, X12);
JuMP.@variable(CR, V31); JuMP.@variable(CR, V32); JuMP.@variable(CR, X13); 
JuMP.@constraint(CR, X12 - 10 * x[1] - 10 * x[2] + 100 >= 0); JuMP.@constraint(CR, X11 - 20 * x[1] + 100 >= 0);
JuMP.@constraint(CR, X23 - 10 * x[2] - 10 * x[3] + 100 >= 0); JuMP.@constraint(CR, X22 - 20 * x[2] + 100 >= 0);
JuMP.@constraint(CR, X13 - 10 * x[1] - 10 * x[3] + 100 >= 0); JuMP.@constraint(CR, X33 - 20 * x[3] + 100 >= 0);
JuMP.@constraint(CR, E * (x[1] + x[2] + 1) >= n1 + n2 + V11 + V12 + V21 + V22);
JuMP.@constraint(CR, E * (10 - x[1]) >= 10 * (n1 + n2) - (V11 + V12));
JuMP.@constraint(CR, E * (10 - x[2]) >= 10 * (n1 + n2) - (V21 + V22));
JuMP.@constraint(CR, E * (10 - x[3]) >= 10 * (n1 + n2) - (V31 + V32));
JuMP.@constraint(CR, (X11 + 2 * X12 + X22) + 1 + 2 * (x[1] + x[2]) >= 0);
JuMP.@constraint(CR, 10 * (x[1] + x[2] + 1) >= x[1] + X11 + X12);
JuMP.@constraint(CR, 10 * (x[1] + x[2] + 1) >= x[2] + X12 + X22);
JuMP.@constraint(CR, 10 * (x[1] + x[2] + 1) >= x[3] + X13 + X23);
# 🥎 linear * convex (Full rectangle ∵ asymmetric): omit n * n, p * n; also omit Λ31, Λ33
JuMP.@constraint(CR, [E * (x[2] - x[3]) + (V31 + V32 - V21 - V22), E - n1 - n2, E * x[1] - V11 - V12] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [(X12 + X22 - X13 - X23) + (x[2] - x[3]), x[1] + x[2] + 1, x[1] + X11 + X12] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [x[1] + X11 + X12, x[1] + x[2] + 1, p1 + Λ11 + Λ21] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [x[3] + X13 + X23, x[1] + x[2] + 1, p3 + Λ13 + Λ23] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [-(x[1] + X11 + X12), 2 * (x[1] + x[2] + 1), n1 + V11 + V21] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [-(x[2] + X12 + X22), 2 * (x[1] + x[2] + 1), n2 + V12 + V22] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * (x[2] - x[3]) - X12 + X13, 10 - x[1], 10 * x[1] - X11] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * (x[2] - x[3]) - X22 + X23, 10 - x[2], 10 * x[1] - X12] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * (x[2] - x[3]) - X23 + X33, 10 - x[3], 10 * x[1] - X13] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [X11 - 10 * x[1], 2 * (10 - x[1]), 10 * n1 - V11] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [X12 - 10 * x[1], 2 * (10 - x[2]), 10 * n1 - V21] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [X13 - 10 * x[1], 2 * (10 - x[3]), 10 * n1 - V31] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [X12 - 10 * x[2], 2 * (10 - x[1]), 10 * n2 - V12] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [X22 - 10 * x[2], 2 * (10 - x[2]), 10 * n2 - V22] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [X23 - 10 * x[2], 2 * (10 - x[3]), 10 * n2 - V32] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * x[1] - X11, 10 - x[1], 10 * p1 - Λ11] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * x[1] - X12, 10 - x[2], 10 * p1 - Λ21] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * x[3] - X13, 10 - x[1], 10 * p3 - Λ13] in JuMP.MOI.ExponentialCone());
JuMP.@constraint(CR, [10 * x[3] - X23, 10 - x[2], 10 * p3 - Λ23] in JuMP.MOI.ExponentialCone());
@opt_ass_opt(CR);
JuMP.objective_bound(CR) # 19.787109815709954
x = JuMP.value.(x) # [1.1856541749087137, 0.9203368055034131, 0.750042146869932], which is feasible by the construction of CR
feasible_OBJVAL = ip([3, -3, 3.], x) + (x[1] + x[2] + 1) * (exp(x[1]) + exp(x[3])) # 19.787110167187194


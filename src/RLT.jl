import JuMP
import LinearAlgebra.dot as ip
import Distributions
import Random
using Logging
import Gurobi

# an example of bilinear programming which can be solved (ub = lb) merely via an application of RLT (relaxation linearization technique)
# 5/3/25

GRB_ENV = Gurobi.Env()
Random.seed!(4)
function GurobiDirectModel(str)
    m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
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

# x's dimensions X L ; Bx >= a
# y's dimensions Y R ; Ay >= b
X = 24; L = 16
Y = 25; R = 18

DA = Distributions.Uniform(1, 41);
A = rand(DA, R, Y);
B = rand(DA, L, X);
Db = Distributions.Uniform(-10, 10);
a = rand(Db, L);
b = rand(Db, R);
DQ = Distributions.Uniform(-20, 20);
Q = rand(DQ, X, Y);

# BL = GurobiDirectModel("m");
# JuMP.@variable(BL, x[1:X] >= 0)
# JuMP.@variable(BL, y[1:Y] >= 0)
# JuMP.@constraint(BL, X >= sum(x))
# JuMP.@constraint(BL, Y >= sum(y))
# JuMP.@constraint(BL, B * x .>= a)
# JuMP.@constraint(BL, A * y .>= b)
# @set_objective_function(BL, ip(x, Q, y)) #  ip(Q, ♭) where ♭ := x * transpose(y)
# Ⓢ = optimise(BL) # -11968.665

CR = GurobiDirectModel("ms");
# JuMP.set_attribute(CR, "DualReductions", 0);
JuMP.@variable(CR, ♭[1:X, 1:Y]); # transpose(♭) = y * transpose(x)
@set_objective_function(CR, ip(Q, ♭));
JuMP.@variable(CR, x[1:X] >= 0);
JuMP.@variable(CR, y[1:Y] >= 0);
JuMP.@constraint(CR, X >= sum(x));
JuMP.@constraint(CR, Y >= sum(y));
JuMP.@constraint(CR, B * x .>= a);
JuMP.@constraint(CR, A * y .>= b);
JuMP.set_lower_bound.(♭, 0); # VI's
JuMP.@constraint(CR, Y * transpose(x) .>= transpose(ones(Y)) * transpose(♭));
JuMP.@constraint(CR, X * transpose(y) .>= transpose(ones(X)) * ♭);
JuMP.@constraint(CR, X * Y - Y * sum(x) - X * sum(y) + sum(♭) >= 0);
JuMP.@constraint(CR, A * transpose(♭) * transpose(B) .- A * y * transpose(a) .- b * transpose(x) * transpose(B) .+ b * transpose(a) .>= 0);
JuMP.@constraint(CR, Y * B * x .- B * ♭ * ones(Y) .- Y * a .+ a * sum(y) .>= 0);
JuMP.@constraint(CR, -A * transpose(♭) * ones(X) .+ X * A * y .- X * b .+ sum(x) * b .>= 0);
JuMP.@constraint(CR, B * ♭ .>= a * transpose(y));
JuMP.@constraint(CR, A * transpose(♭) .>= b * transpose(x));
@opt_ass_opt(CR)
lb = JuMP.objective_bound(CR) # -11968.66468736564
xt = JuMP.value.(x)
yt = JuMP.value.(y)
ip(xt, Q, yt) # -11968.66468736565

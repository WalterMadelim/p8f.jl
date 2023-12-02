using CairoMakie
import Ipopt
import JuMP
function f(x) # a 1-dimensional continuous function constructed for optimization
    # (x = range(-2.5, 3.8, length = 400); y = f.(x); lines(x,y))
    -2.692e-3 * (x - 0.17767) * (x + 1.0349) * (x - 3.76882) * (x + 4.35247) * (x - 4.17767) * (x + 3.0349) * (x - 2.76882) * (x + 1.35247) * (x - 1.0) * (x + 2.0) * (x - 3.0)
end
m = JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(m, -2.5 <= x <= 3.8, start = -2.) # try different start points
JuMP.@objective(m, Min, f(x)) # or Max
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
_x = JuMP.value(x)
println(JuMP.value(_x))
println(f(_x))
println(f(_x) - JuMP.objective_value(m))

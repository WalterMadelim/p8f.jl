using CairoMakie
import Ipopt
import JuMP
import IntervalOptimisation

function f(x) # a 1-dimensional continuous function constructed for optimization
    # (x = range(-2.5, 3.8, length = 400); y = f.(x); lines(x,y))
    # [The Min] f(1.8833787463035296) = -19.511359073980536
    # [The Max] f(-2.500000024960437) = 25.492479946830965
    -2.692e-3 * (x - 0.17767) * (x + 1.0349) * (x - 3.76882) * (x + 4.35247) * (x - 4.17767) * (x + 3.0349) * (x - 2.76882) * (x + 1.35247) * (x - 1.0) * (x + 2.0) * (x - 3.0)
end
function aminimizer(f, l, h)
    interval = IntervalOptimisation.minimise(f, IntervalOptimisation.interval(l, h), tol=1e-4)[2][1]
    (interval.lo + interval.hi)/2
end

function dim1_x_ast(f, l, h) # find a global minimum x∗ of f over [l,h]
    x1 = aminimizer(f, l, h)
    println(x1)
    m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
    JuMP.@variable(m, l <= x <= h, start = x1)
    JuMP.@objective(m, Min, f(x))
    JuMP.set_silent(m)
    JuMP.optimize!(m) 
    @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
    x2 = JuMP.value(x)
    println(x2)
    if f(x2) >= f(x1)
        error("check if f(x2) > f(x1) which is invalid, or f(x1) == f(x2) which is actually acceptable.")
    elseif abs(x2 - x1) > 1e-3
        error("the improved minimizer is moving far away!")
    end
    return x2
end
function dim1_x_max(f, l, h) # find a global minimum x∗ of f over [l,h]
    x1 = aminimizer(x -> -f(x), l, h)
    m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
    JuMP.@variable(m, l <= x <= h, start = x1)
    JuMP.@objective(m, Max, f(x))
    JuMP.set_silent(m)
    JuMP.optimize!(m) 
    @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
    x2 = JuMP.value(x)
    if f(x2) <= f(x1)
        error("check if f(x2) < f(x1) which is invalid, or f(x1) == f(x2) which is actually acceptable.")
    elseif abs(x2 - x1) > 1e-3
        error("the improved maximizer is moving far away!")
    end
    return x2
end

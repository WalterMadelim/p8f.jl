# using CairoMakie
import JuMP
import Ipopt
import Gurobi
import IntervalOptimisation
if true # functions
    function g(x)
        sin(x) + sin((10.0 / 3.0) * x)
    end
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
        m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
        JuMP.@variable(m, l <= x <= h, start = x1)
        JuMP.@objective(m, Min, f(x))
        JuMP.set_silent(m)
        JuMP.optimize!(m) 
        @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
        x2 = JuMP.value(x)
        if f(x2) >= f(x1)
            error("check if f(x2) > f(x1) which is invalid, or f(x1) == f(x2) which is actually acceptable.")
        elseif abs(x2 - x1) > .01
            error("the improved minimizer is moving far away!")
        end
        return x2
    end
    function dim1_x_max(f, l, h) # find a global minimum x∗ of f over [l,h]
        x1 = aminimizer(x -> -f(x), l, h)
        # println("x1 == $x1")
        m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
        JuMP.@variable(m, l <= x <= h, start = x1)
        JuMP.@objective(m, Max, f(x))
        JuMP.set_silent(m)
        JuMP.optimize!(m) 
        @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
        x2 = JuMP.value(x)
        # println("x2 == $x2")
        if f(x2) <= f(x1)
            error("check if f(x2) < f(x1) which is invalid, or f(x1) == f(x2) which is actually acceptable.")
        elseif abs(x2 - x1) > .01
            error("the improved maximizer is moving far away!")
        end
        return x2
    end
    function slo(f, xp, s) # (NL function, PWL ends, seg_num)
        (f(xp[s+1]) - f(xp[s])) / (xp[s+1] - xp[s])
    end
    function err_bnd(f,xp) # (NL function, PWL ends)
        n = length(xp) - 1
        epso, epsu = zeros(n), zeros(n)
        for s in 1:n
            anf = x -> ((f(xp[s]) + slo(f, xp, s) * (x - xp[s])) - f(x))
            tmpx = dim1_x_max(anf, xp[s], xp[s+1])
            epso[s] = anf(tmpx)
            anf = x -> (f(x) - (f(xp[s]) + slo(f, xp, s) * (x - xp[s])))
            tmpx = dim1_x_max(anf, xp[s], xp[s+1])
            epsu[s] = anf(tmpx)
        end
        return epso, epsu
    end
end

GRB_ENV = Gurobi.Env()
xp = [-2.5, -1.5 ,3.8] # piecewise_ends
n = length(xp) - 1
# epso, epsu = err_bnd(f, xp)


m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
JuMP.@variable(m, x)
JuMP.@variable(m, y)
JuMP.@variable(m, 0. <= d[1:n] <= 1.)
JuMP.@variable(m, z[1:n-1], Bin)
JuMP.@constraint(m, [i = 1:n-1], z[i] >= d[i+1])
JuMP.@constraint(m, [i = 1:n-1], z[i] <= d[i])
JuMP.@constraint(m, x == xp[1] + sum(d[s] * (xp[s+1] - xp[s]) for s in 1:n))
e = y - (f(xp[1]) + sum(d[s] * (f(xp[s+1]) - f(xp[s])) for s in 1:n))
JuMP.@constraint(m,  e <= epsu[1] + sum(z[s] * (epsu[s+1] - epsu[s]) for s in 1:n-1))
JuMP.@constraint(m, -e <= epso[1] + sum(z[s] * (epso[s+1] - epso[s]) for s in 1:n-1))
JuMP.@objective(m, Min, y)
JuMP.set_silent(m)
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL
JuMP.value(x)
JuMP.value(y)


# fig = Figure();
# ax = Axis(fig[1, 1]);
# xd = range(-2.5, 3.8, length = 400);
# lines!(ax, xd, f.(xd))
# lines!(ax, xp, f.(xp))

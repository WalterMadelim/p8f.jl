using CairoMakie
import JuMP
import Ipopt
import Gurobi
import IntervalOptimisation

# use copy-paste workflow !!!
# 1, solve NLP with iterated MILPs
# 2, Use IntervalOptimisation's global 1d optimization for a good start point, then we use Ipopt's global search to decide err_bnd
# 03/12/23

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
    function aminimizer(f, l, h) # a global solution, IntervalOptimisation boasted 
        interval = IntervalOptimisation.minimise(f, IntervalOptimisation.interval(l, h), tol=1e-4)[2][1]
        (interval.lo + interval.hi)/2
    end
    # function dim1_x_ast(f, l, h) # deprecated because only dim1_x_max is useful in this application
    #     x1 = aminimizer(f, l, h)
    #     m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
    #     JuMP.@variable(m, l <= x <= h, start = x1)
    #     JuMP.@objective(m, Min, f(x))
    #     JuMP.set_silent(m)
    #     JuMP.optimize!(m) 
    #     @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
    #     x2 = JuMP.value(x)
    #     if f(x2) >= f(x1)
    #         @error "(IntervalOptimisation) x1 = $x1 vs. $x2 = x2 (Ipopt)"
    #     elseif abs(x2 - x1) > 1.00000
    #         @error("the improved maximizer is moving far away with $(abs(x2-x1)).  $x1 vs. $x2")
    #     end
    #     return x2
    # end
    function dim1_x_max(f, l, h) # find a quasi global minimum x∗ of f over [l,h]
        x1 = aminimizer(x -> -f(x), l, h)
        m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
        JuMP.@variable(m, l <= x <= h, start = x1)
        JuMP.@objective(m, Max, f(x))
        JuMP.set_silent(m)
        JuMP.optimize!(m) 
        @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
        x2 = JuMP.value(x)
        if abs(x2 - x1) > .5
            @error("x2 leaves x1 by $(abs(x2-x1)), where x1 = $x1 vs. $x2 = x2")
        end
        return (f(x2) > f(x1)) ? x2 : x1 # the quasi global optimum ("quasi" owing to no dual bounding provided)
    end
    function slo(f, xp, s) # (NL function, PWL ends, seg_num)
        (f(xp[s+1]) - f(xp[s])) / (xp[s+1] - xp[s])
    end
    function err_bnd(f, xp) # (NL function, PWL ends)
        n = length(xp) - 1
        epso, epsu = zeros(n), zeros(n)
        for s in 1:n
            anf = x -> ((f(xp[s]) + slo(f, xp, s) * (x - xp[s])) - f(x))
            epso[s] = anf(dim1_x_max(anf, xp[s], xp[s+1]))
            anf = x -> (f(x) - (f(xp[s]) + slo(f, xp, s) * (x - xp[s])))
            epsu[s] = anf(dim1_x_max(anf, xp[s], xp[s+1]))
        end
        return epso, epsu
    end
    function loc_seg(_x, xp) # locate the segment for which _x resides
        if _x >= xp[end]
            return n
        elseif _x <= xp[1]
            return 1
        end
        tmp = findall(x -> x == 0., xp .- _x)
        if !isempty(tmp)
            return (rand() > .5) ? tmp[1] : (tmp[1]-1)
        end
        return findall(x -> x > 0., xp .- _x)[1] - 1
    end
    function vis(f, xp) # visualization
        fig = Figure();
        ax = Axis(fig[1, 1]);
        xd = range(xp[1], xp[end], length = 400);
        lines!(ax, xd, f.(xd))
        lines!(ax, xp, f.(xp))
        fig
    end
end

GRB_ENV = Gurobi.Env()
xp = [-2.5, 3.8] # initial segment 
xast = [xp[1]] # initial incumbent
ub = f(xp[1]) # initial upper bound for the NLP
xp = [xp[1]; (xp[1]+xp[2])/2; xp[2]] # manmade the initial central node = .65
n = length(xp) - 1 # number of segments currently
for cnt in 1:20000
    epso, epsu = err_bnd(f, xp)
    m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # MILP
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
    JuMP.@objective(m, Min, y) # Need to change for concrete applications!
    JuMP.set_silent(m)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    lb = JuMP.objective_value(m)
    _x = JuMP.value(x)
    nub = f(_x)
    if nub < ub
        ub, xast[1] = nub, _x # record incumbent ub and solution
    end
    gap = ub - lb
    @info "Ite = $cnt: $lb < $ub, gap = $gap"
    if gap < 1e-6
        @info "Global optimum found at $(xast[1]) with (feas) val = $ub and lb = $lb"
        break
    end
    s = loc_seg(_x, xp) # locate the segment s where the current MILP solution _x resides
    xp = [xp[1:s]; (xp[s] + xp[s+1])/2; xp[s+1:end]] # longest edge bisection
    n = length(xp) - 1
end

vis(f, xp) # visualization, at xp's current state


import MathOptInterface as MOI
import Gurobi
import Random
import Plots
# N = 40, TSP problem, using Gurobi_Lazy_cuts
function generate_distance_matrix(n; random_seed = 1)
    rng = Random.MersenneTwister(random_seed)
    X = 100 * rand(rng, n)
    Y = 100 * rand(rng, n)
    d = [sqrt((X[i] - X[j])^2 + (Y[i] - Y[j])^2) for i in 1:n, j in 1:n]
    return X, Y, d
end
function selected_edges(x::Matrix{Float64})
    n = size(x,1)
    return Tuple{Int,Int}[(i, j) for i in 1:n, j in 1:n if x[i, j] > 0.5]
end
function plot_tour(X, Y, x)
    plot = Plots.plot()
    for (i, j) in selected_edges(x)
        Plots.plot!([X[i], X[j]], [Y[i], Y[j]]; legend = false)
    end
    return plot
end
function subtour(edges::Vector{Tuple{Int,Int}}, n)
    shortest_subtour, unvisited = collect(1:n), Set(collect(1:n))
    while !isempty(unvisited)
        this_cycle, neighbors = Int[], unvisited
        while !isempty(neighbors)
            current = pop!(neighbors)
            push!(this_cycle, current)
            if length(this_cycle) > 1
                pop!(unvisited, current)
            end
            neighbors =
                [j for (i, j) in edges if i == current && j in unvisited]
        end
        if length(this_cycle) < length(shortest_subtour)
            shortest_subtour = this_cycle
        end
    end
    return shortest_subtour
end
subtour(x_k::Matrix{Float64}) = subtour(selected_edges(x_k), size(x_k, 1))
plot_tour(x_k::Matrix{Float64}) = plot_tour(X,Y,x_k)

function my_GRB_cb_function(cb_data, cb_where::Cint)
    # push!(cb_registry, cb_where) # cb_where can be any possible Cint 0-9
    if cb_where == Gurobi.GRB_CB_MIPSOL # whenever a new MIP solution is found (cb_where = 4)
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        # - - - - - - - - - - - manipulations here - - - - - - - - - - -
        x_k = MOI.get.(o, MOI.CallbackVariablePrimal(cb_data), x)
        cycle = subtour(x_k) # [4,9,2,7]
        l = length(cycle) # 4
        if 1 < l < N # subtour exists, must add cut
            S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
            c = length(S) # combinatorial num c_S_2
            terms = fill(tmp_sat,c)
            for i in 1:c
                terms[i] = MOI.ScalarAffineTerm(1.,x[S[i]...])
            end
            f = MOI.ScalarAffineFunction(terms, 0.)
            MOI.submit(o,MOI.LazyConstraint(cb_data),f,MOI.LessThan(l - .5)) # <= 3.5, eq2 <= 3 for integers
        end
        # - - - - - - - - - - - manipulations here - - - - - - - - - - - 
    end
    return nothing
end


N = 40
X,Y,d = generate_distance_matrix(N)

tmp_vindex = MOI.add_variable(Gurobi.Optimizer())
x = fill(tmp_vindex,N,N)
tmp_sat = MOI.ScalarAffineTerm(1.,tmp_vindex)
objterms = fill(tmp_sat,N^2)


o = Gurobi.Optimizer()
for i in 1:N
    for j in 1:N
        x[i,j] = MOI.add_variable(o)
    end # MOI.get(o,MOI.NumberOfVariables())
end 
MOI.add_constraint.(o, x, MOI.ZeroOne())
for i in 1:N
    terms = MOI.ScalarAffineTerm.(1.,x[i,:])
    f = MOI.ScalarAffineFunction(terms, 0.)
    s = MOI.EqualTo(2.)
    MOI.add_constraint(o,f,s)
    objterms[(i-1)N+1:(i)N] .= MOI.ScalarAffineTerm.(d[i,:],x[i,:])
    for j in i:N
        if i == j
            MOI.add_constraint(o,x[i,i],MOI.LessThan(.5)) # set zero
        else
            terms = [MOI.ScalarAffineTerm(1.,x[i,j]),MOI.ScalarAffineTerm(-1.,x[j,i])]
            f = MOI.ScalarAffineFunction(terms, 0.)
            # s = MOI.Interval(-.5,.5) # not support 
            MOI.add_constraint(o,f,MOI.LessThan(.5))
            MOI.add_constraint(o,f,MOI.GreaterThan(-.5))
        end
    end
end
# obj function and SENSE
f = MOI.ScalarAffineFunction(objterms, 0.)
type_matters = MOI.ObjectiveFunction{typeof(f)}()
MOI.set(o,type_matters,f)
type_matters = MOI.ObjectiveSense()
MOI.set(o, type_matters, MOI.MIN_SENSE)

# start optimizing 
MOI.set(o, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
MOI.set(o, Gurobi.CallbackFunction(), my_GRB_cb_function)
MOI.optimize!(o)

attrs = [
    MOI.TerminationStatus(),
    MOI.PrimalStatus(),
    MOI.DualStatus(), # NO_SOLUTION, due to an MIP
    MOI.ResultCount(),
    MOI.ObjectiveValue()
]
attrs = MOI.get.(o, attrs) # get a first insight
println("ObjVal = $(attrs[end]/2)") # ObjVal = 525.7039004442727
x_k = MOI.get.(o, MOI.VariablePrimal(), x)
plot_tour(x_k) # same as the one in Doc

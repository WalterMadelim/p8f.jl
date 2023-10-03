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
# when N set to 300, the results by Gurobi
# julia> MOI.optimize!(o)
# Gurobi Optimizer version 10.0.2 build v10.0.2rc0 (win64)

# CPU model: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, instruction set [SSE2|AVX|AVX2|AVX512]
# Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

# Optimize a model with 90000 rows, 90000 columns and 269400 nonzeros
# Model fingerprint: 0x9147df54
# Variable types: 0 continuous, 90000 integer (90000 binary)
# Coefficient statistics:
#   Matrix range     [1e+00, 1e+00]
#   Objective range  [3e-01, 1e+02]
#   Bounds range     [5e-01, 5e-01]
#   RHS range        [5e-01, 2e+00]
# Presolve removed 89700 rows and 45150 columns
# Presolve time: 0.20s
# Presolved: 300 rows, 44850 columns, 89700 nonzeros
# Variable types: 0 continuous, 44850 integer (44850 binary)

# Root relaxation: objective 2.456625e+03, 468 iterations, 0.02 seconds (0.01 work units)

#     Nodes    |    Current Node    |     Objective Bounds      |     Work
#  Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

#      0     0 2456.62490    0   52          - 2456.62490      -     -    0s
#      0     0 2470.79119    0   42          - 2470.79119      -     -    0s
#      0     0 2475.42178    0   72          - 2475.42178      -     -    0s
#      0     0 2488.71573    0   38          - 2488.71573      -     -    0s
#      0     0 2488.71573    0   38          - 2488.71573      -     -    0s
#      0     2 2488.71573    0   38          - 2488.71573      -     -    1s
#   1270  1312 2639.18549  250   16          - 2492.93987      -   3.3    5s
#   3304  3229 2533.72909   26   67          - 2517.42471      -   3.6   10s
#   3964  3701 2554.38373   85    8          - 2538.06666      -   4.1   15s
#   5990  5143 2710.35220  319   14          - 2538.06666      -   4.7   20s
#   8280  6560 2593.67718   87   16          - 2538.64186      -   5.1   25s
#  10234  7565 2671.87349  224    6          - 2538.64186      -   5.4   33s
#  10262  8184 2675.87472  228    6          - 2538.64186      -   5.5   35s
#  13148 10072 2855.92413  508   12          - 2538.64186      -   5.7   40s
#  15442 12067 3143.78779  752   14          - 2539.15415      -   5.8   45s
#  17579 14312 2614.15454   73   22          - 2539.56466      -   5.9   50s
#  20469 16921 2665.23980  323   12          - 2539.56613      -   6.0   55s
#  22840 19068 2644.63533  165    6          - 2539.76761      -   6.1   60s
#  24717 20903 infeasible  439               - 2539.77802      -   6.2   65s
# *25726 21496            1591    3830.2907021 2539.77802  33.7%   6.3   66s
#  25957 21564 2682.27396  197    6 3830.29070 2539.77802  33.7%   6.3   70s
#  26243 21850 2728.17497  303    6 3830.29070 2539.77802  33.7%   6.3   76s
#  26456 21956 2762.45307  356    8 3830.29070 2539.77802  33.7%   6.3   81s
#  26893 22629 2843.11319  449   10 3830.29070 2539.77802  33.7%   6.3   86s
#  28197 24266 3185.52228  755    8 3830.29070 2539.77802  33.7%   6.2   91s
#  30370 26297 3011.46305  424    8 3830.29070 2540.13211  33.7%   6.2   95s
#  32178 27962     cutoff  839      3830.29070 2540.65712  33.7%   6.2  100s
#  33896 29607 3324.38398  606    6 3830.29070 2540.65712  33.7%   6.2  105s
#  35752 31113 2738.48729  245    8 3830.29070 2540.77476  33.7%   6.2  112s
#  37561 32905 2619.84808   65   18 3830.29070 2541.28786  33.7%   6.2  116s
# H39287 10067                    2662.4267518 2541.30125  4.55%   6.1  121s
#  39991 11199 2607.26630   43   38 2662.42675 2541.81519  4.53%   6.1  125s
#  41983 12991 2589.67383   36   89 2662.42675 2542.68003  4.50%   6.1  130s
# *42584  9978              86    2642.2391653 2542.77641  3.76%   6.1  130s
#  44006 11241     cutoff  143      2642.23917 2543.06408  3.75%   6.1  136s
#  44992 12237 2632.67453   99   20 2642.23917 2543.58929  3.73%   6.1  140s
#  46862 13750 2609.80971   66   92 2642.23917 2544.46712  3.70%   6.1  145s
#  48642 15170 2626.28699   87    8 2642.23917 2545.13169  3.68%   6.2  150s
#  50472 16675 2626.46399   79   35 2642.23917 2546.00009  3.64%   6.2  156s
# *51951 17067              94    2640.4047197 2546.82628  3.54%   6.2  159s
#  52255 17525     cutoff  138      2640.40472 2546.99218  3.54%   6.2  161s
#  53414 18243     cutoff  197      2640.40472 2547.56218  3.52%   6.2  165s
#  54858 19156 2633.79796   73   38 2640.40472 2549.03313  3.46%   6.2  175s
# H54879 18212                    2619.9693742 2598.10466  0.83%   6.2  177s
# H54913 17323                    2613.2738205 2602.07240  0.43%   6.2  179s
# H54920 16462                    2613.2020741 2602.25726  0.42%   6.3  179s
# H54928 15644                    2611.8448679 2602.60760  0.35%   6.3  180s
# H55214 14990                    2610.8744261 2604.47398  0.25%   6.6  184s
#  55561 15020 2609.94618   89  144 2610.87443 2604.64940  0.24%   6.8  185s
#  60326 14613     cutoff   90      2610.87443 2607.64487  0.12%   8.6  190s

# Cutting planes:
#   Gomory: 30
#   Lift-and-project: 2
#   Cover: 9
#   MIR: 4
#   StrongCG: 1
#   Flow cover: 25
#   Inf proof: 2
#   Zero half: 110
#   Lazy constraints: 10

# Explored 67143 nodes (710029 simplex iterations) in 194.29 seconds (168.09 work units)
# Thread count was 8 (of 8 available processors)

# Solution count 9: 2610.87 2611.84 2613.2 ... 3830.29

# Optimal solution found (tolerance 1.00e-04)
# Best objective 2.610874426066e+03, best bound 2.610873053171e+03, gap 0.0001%

# User-callback calls 141699, time in user-callback 3.21 sec

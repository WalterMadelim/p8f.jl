# Kelleys_cutting_plane from SDDP.jl
using JuMP
import ForwardDiff,Gurobi

function g()
    K = 0 # inside a function
    while K < 5
        K += 1 # DO NOT add any prefix
    end
    println("End of g() with K = ",K)
end
g()

function kelleys_cutting_plane(
    f::Function = x -> (x[1]-1)^2 + (x[2]+2)^2 + 1.,
    dfdx::Function = x -> ForwardDiff.gradient(f,x);
    input_dimension::Int = 2,
    lower_bound::Float64 = 0.,
    iteration_limit::Int = 20,
    tolerance::Float64 = 1e-6
)
    K = 0 # local K in the FUNCTION SCOPE (this is important)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model,θ ≥ lower_bound)
    @variable(model,x[1:input_dimension])
    @objective(model,Min,θ)
    x_k = fill(NaN,input_dimension) # pre_allocate
    lower_bound,upperbound = -Inf,Inf
    while true
        optimize!(model)
        termination_status(model) != OPTIMAL && error("termination_status NOT OPTIMAL")
        x_k .= value.(x) # get current solutions
        lower_bound = objective_value(model)
        upper_bound = min(upperbound,f(x_k)) # f is the true function, not the one you want to approximate in JuMP model
        println("K = $K: $lower_bound ≤ f(x*) ≤ $upper_bound")
        @constraint(model,θ ≥ f(x_k) + dfdx(x_k)' * (x .- x_k)) # add a cut, due to the true function being convex
        K += 1 # this K is neither global nor local!
        if K == iteration_limit
            println("---- Termination due to iteration_limit ----")
            break
        elseif upper_bound < lower_bound - tolerance
            error("Due to ub < lb")
        elseif abs(upper_bound - lower_bound) < tolerance
            println("---- Termination due to ub-lb < ϵ ----")
            break
        end
    end
    println("Found solution: x_k = ", x_k)
end

# call
kelleys_cutting_plane()
# or call
kelleys_cutting_plane(;iteration_limit=50)

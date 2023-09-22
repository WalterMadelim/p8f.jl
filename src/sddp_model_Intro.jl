using JuMP
import Gurobi, Statistics

struct State # a single state, consisting of 2 actual variables
    in::VariableRef
    out::VariableRef
end
struct Uncertainty
    parameterize::Function # configuring the r.v.
    Ω::Vector{Any}
    P::Vector{Float64}
end
struct Node
    subproblem::Model # Every-stage JuMP model
    states::Dict{Symbol,State} # State is a composite type just defined
    uncertainty::Uncertainty
    cost_to_go::VariableRef
end
struct PolicyGraph # the SDDP model
    nodes::Vector{Node}
    arcs::Vector{Dict{Int,Float64}} # transition prob
end
function Base.show(io::IO,model::PolicyGraph)
    println(io,"A PolicyGraph with $(model.nodes |> length) nodes")
    println(io,"Arcs:")
    for (from,arcs) in enumerate(model.arcs)
        for (to,probability) in arcs
            println(io, "   $from => $to w.p. $probability")
        end
    end
end

function subproblem_initializer!(subproblem::Model, t::Int)
    @variable(subproblem, volume_in == 200) # firstly fix the volume_in, modified later, before optimize!
    @variable(subproblem, 0≤ volume_out ≤200) # define state_in_out explicitly
    states = Dict(:volume => State(volume_in,volume_out)) # a field of Node, irrelevant to node index t
    @variables(subproblem, begin
        thermal_generation ≥ 0
        hydro_generation ≥ 0
        hydro_spill ≥ 0
        inflow # a r.v., which is to be fixed outside of this function later
    end)
    @constraints(
        subproblem,
        begin
            volume_out == volume_in + inflow - hydro_generation - hydro_spill
            demand_constraint, thermal_generation + hydro_generation == 150.
        end
    )
    fuel_cost = [50.,100,150]
    @objective(subproblem, Min, fuel_cost[t] * thermal_generation) # here, t is from argument
    uncertainty = Uncertainty(ω -> fix(inflow,ω),[0.,50,100],[1/3,1/3,1/3])
    states, uncertainty
end

function PolicyGraph(subproblem_initializer!;arcs= [Dict(2 => 1.0), Dict(3 => 1.0), Dict{Int,Float64}()],a_lower_bound=0.,optimizer=Gurobi.Optimizer)
    nodes = Node[] # isempty(nodes) == true, eltype(nodes) == Node
    for t in eachindex(arcs) # t = 1,2,...,3
        model = Model(optimizer) # empty JuMP model with Gurobi.Optimizer
        set_silent(model)
        states, uncertainty = subproblem_initializer!(model,t)
        @variable(model, cost_to_go ≥ a_lower_bound) # introduce θ (cost_to_go)
        obj = objective_function(model) # save current obj temply
        @objective(model,Min,obj + cost_to_go) # then modify it!
        isempty(arcs[t]) && fix(cost_to_go, 0.;force = true) # last stage do not need θ
        push!(nodes,Node(model,states,uncertainty,cost_to_go))
    end
    PolicyGraph(nodes,arcs)
end

function sample_uncertainty(uncertainty::Uncertainty)::Float64
    r = rand()
    for (p,ω) in zip(uncertainty.P,uncertainty.Ω)
        r -= p
        r < 0. && return ω
    end
    error("Shouldn't reach here since P(Ω)=1")
end

function sample_next_node(sdmd::PolicyGraph, current::Int)
    isempty(sdmd.arcs[current]) && return nothing
    r = rand()
    for (to,p) in sdmd.arcs[current]
        r -= p
        r < 0. && return to
    end
    nothing
end

function forward_pass(sdmd::PolicyGraph, io::IO = stdout)
    println(io, "| Forward Pass")
    incoming_state = Dict(aSymbol => fix_value(aState.in) for (aSymbol,aState) in sdmd.nodes[1].states) # read info & save temply
    simulation_cost = 0.
    trajectory = Tuple{Int,Dict{Symbol,Float64}}[]
    t = 1 # root
    while t !== nothing
        node = sdmd.nodes[t]
        println(io,"|| Visiting node $t")
        println(io,"||| x =", incoming_state) # when t == root, this incoming_state is an deterministic initial state
        ω = sample_uncertainty(node.uncertainty)
        println(io,"||| ω =", ω)
        node.uncertainty.parameterize(ω) # fix(inflow,ω) which inflow? indicated by node
        for (aSymbol,aFloat64) in incoming_state
            fix(node.states[aSymbol].in,aFloat64;force=true) # now modify the in_state value in time
        end
        optimize!(node.subproblem)
        termination_status(node.subproblem) != OPTIMAL && error("optimize! with errors.")
        outgoing_state = Dict(aSymbol => value(aState.out) for (aSymbol,aState) in node.states) # record the value of out_state temply
        println(io,"||| x' = ", outgoing_state)
        stage_cost = objective_value(node.subproblem) - value(node.cost_to_go)
        println(io, "||| C(x,u,w)=", stage_cost)
        simulation_cost += stage_cost
        push!(trajectory, (t,outgoing_state))
        t, incoming_state = sample_next_node(sdmd,t), outgoing_state # prepare for the next subproblem
    end
    trajectory, simulation_cost
end

function backward_pass(
    sdmd::PolicyGraph,
    trajectory::Vector{Tuple{Int,Dict{Symbol,Float64}}}=[(1, Dict(:volume => 0.0)), (2, Dict(:volume => 0.0)), (3, Dict(:volume => 0.0))],
    io::IO = stdout
)
    println(io, "| Backward pass")
    for i in reverse(eachindex(trajectory)) # i = 3,2,1
        index, outgoing_states = trajectory[i] # 3,Dict(:volume => 0.0)
        node = sdmd.nodes[index]
        println(io, "| | Visiting node $(index)")
        if isempty(sdmd.arcs[index])
            println(io, "| | | Skipping node because the cost-to-go is 0")
            continue
        end
        # Create an empty affine expression that we will use to build up the
        # right-hand side of the cut expression.
        cut_expression = AffExpr(0.) # initial const val 0.
        for (j, P_ij) in sdmd.arcs[index] # For each node j ∈ i⁺, and transition probability
            next_node = sdmd.nodes[j]
            for (aSymbol, aFloat64) in outgoing_states
                fix(next_node.states[aSymbol].in, aFloat64; force = true)
            end
            for (pφ, φ) in zip(next_node.uncertainty.P, next_node.uncertainty.Ω)# Then for each realization of φ ∈ Ωⱼ
                println(io, "| | | Solving φ = ", φ)
                next_node.uncertainty.parameterize(φ) # fix(inflow,φ)
                optimize!(next_node.subproblem)
                # Then prepare the cut `P_ij * pφ * [V + dVdxᵀ(x - x_k)]``
                V = objective_value(next_node.subproblem)
                println(io, "| | | | V = ", V)
                dVdx = Dict(aSymbol => reduced_cost(aState.in) for (aSymbol, aState) in next_node.states)
                println(io, "| | | | dVdx′ = ", dVdx)
                cut_expression += @expression(
                    node.subproblem,
                    P_ij * pφ * (V + sum(
                        dVdx[aSymbol] * (aState.out - outgoing_states[aSymbol]) for (aSymbol, aState) in node.states
                    )),
                ) # use @ to create an expr that associate with node.subproblem, but do not register explicitly!
            end
        end
        # And then refine the cost-to-go variable by adding the cut:
        c = @constraint(node.subproblem, node.cost_to_go >= cut_expression) # create a constr named c in the node.subproblem, but do not register!
        println(io, "| | | Adding cut : ", c)
    end
end

function lower_bound(sdmd::PolicyGraph)
    node,bound = sdmd.nodes[1],0. # root
    for (p, ω) in zip(node.uncertainty.P, node.uncertainty.Ω)
        node.uncertainty.parameterize(ω)
        optimize!(node.subproblem)
        bound += p * objective_value(node.subproblem)
    end
    bound
end

function upper_bound(sdmd::PolicyGraph; replications::Int=100)
    # Pipe the output to `devnull` so we don't print too much!
    simulations = [forward_pass(sdmd, devnull) for _ in 1:replications]
    z = [s[2] for s in simulations]
    μ = Statistics.mean(z)
    tσ = 1.96 * Statistics.std(z) / sqrt(replications)
    return μ, tσ
end

function train(
    sdmd::PolicyGraph;
    iteration_limit::Int=3,
    replications::Int=100,
    io::IO = stdout,
)
    for i in 1:iteration_limit
        println(io, "Starting iteration $(i)")
        trajectory, _ = forward_pass(sdmd, io)
        backward_pass(sdmd, trajectory, io)
        println(io, "| Finished iteration")
        println(io, "| | lower_bound = ", lower_bound(sdmd))
    end
    println(io, "Termination status: iteration limit")
    μ, tσ = upper_bound(sdmd; replications = replications)
    println(io, "Upper bound = $(μ) ± $(tσ)")
end

function evaluate_policy(
    sdmd::PolicyGraph;
    node::Int = 1,
    incoming_state::Dict{Symbol,Float64} = Dict(:volume => 150.0),
    random_variable = 75,
)
    the_node = sdmd.nodes[node]
    the_node.uncertainty.parameterize(random_variable)
    for (k, v) in incoming_state
        fix(the_node.states[k].in, v; force = true)
    end
    optimize!(the_node.subproblem)
    Dict(k => value.(v) for (k, v) in object_dictionary(the_node.subproblem))
end

sdmd = PolicyGraph(subproblem_initializer!)
train(sdmd)
evaluate_policy(sdmd)

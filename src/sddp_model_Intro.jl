using JuMP
import Gurobi

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
    @variable(subproblem, volume_in == 200) # firstly fix the volume_in
    @variable(subproblem, 0≤ volume_out ≤200)
    states = Dict(:volume => State(volume_in,volume_out)) # a field of Node
    @variables(subproblem, begin
        thermal_generation ≥ 0
        hydro_generation ≥ 0
        hydro_spill ≥ 0
        inflow # a r.v. actually
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
    nodes = Node[]
    for t in eachindex(arcs)
        model = Model(optimizer)
        set_silent(model)
        states, uncertainty = subproblem_initializer!(model,t)
        @variable(model, cost_to_go ≥ a_lower_bound)
        obj = objective_function(model)
        @objective(model,Min,obj + cost_to_go)
        isempty(arcs[t]) && fix(cost_to_go, 0.;force = true)
        push!(nodes,Node(model,states,uncertainty,cost_to_go))
    end
    PolicyGraph(nodes,arcs)
end

sdmd = PolicyGraph(subproblem_initializer!)

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


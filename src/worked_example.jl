import Gurobi
import JuMP
using Logging
import Random
import Distributions

# Worked example 2S Inventory control problem
# all-emcompassing general coding is involved, thus unfinished
# suggest: use hardcode instead
# `multistage robust mixed-integer optimization with adaptive partitions`
# 2024/3/18

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function get_lb_of_ARO(xi_snf)
    m = JumpModel()
    JuMP.@variable(m, x1 >= 0.)
    JuMP.@variable(m, y2A[eachindex(xi_snf)], Bin)
    JuMP.@variable(m, y2B[eachindex(xi_snf)], Bin)
    JuMP.@variable(m, z)
    for (k, xi) in enumerate(xi_snf) # the set should be the full Xi, but we couldn't due to inf cardinality
        JuMP.@constraint(m, x1 - xi + 25. * y2A[k] + 25. * y2B[k] >= 0.)
        JuMP.@constraint(m, 50. * x1 + 65. * (x1 - xi + 25. * y2A[k] + 25. * y2B[k]) + 1500. * y2A[k] + 1875. * y2B[k] <= z)
    end
    JuMP.@objective(m, Min, z)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(z)
end
function SRO_master(xi_snf)
    m = JumpModel()
    JuMP.@variable(m, x1 >= 0.)
    JuMP.@variable(m, y2A, Bin)
    JuMP.@variable(m, y2B, Bin)
    JuMP.@variable(m, z)
    for xi in xi_snf # the set should be the full Xi, but we couldn't due to inf cardinality
        JuMP.@constraint(m, x1 - xi + 25. * y2A + 25. * y2B >= 0.)
        JuMP.@constraint(m, 50. * x1 + 65. * (x1 - xi + 25. * y2A + 25. * y2B) + 1500. * y2A + 1875. * y2B <= z)
    end
    JuMP.@objective(m, Min, z)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value.([x1, y2A, y2B, z])
end
function SRO_subproblem(xi, y2A, y2B, z)
    m = JumpModel()
    JuMP.@variable(m, Xi[begin] <= xi <= Xi[end])
    JuMP.@variable(m, vio[1:2])
    JuMP.@constraint(m, vio[1] == 50. * x1 + 65. * (x1 - xi + 25. * y2A + 25. * y2B) + 1500. * y2A + 1875. * y2B - z)
    JuMP.@constraint(m, vio[2] == -(x1 - xi + 25. * y2A + 25. * y2B))
    JuMP.@variable(m, maxvio)
    errcode_norm = Gurobi.GRBaddgenconstrMax(JuMP.backend(m), "", column(maxvio), Cint(2), column.(vio), Cdouble(0.))
    @assert errcode_norm == 0 "Gurobi's Max constr fail"
    JuMP.@objective(m, Max, maxvio)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(maxvio), JuMP.value(xi)
end
function new_leaves_c1(xi, y2A, y2B, z)
    m = JumpModel()
    JuMP.@variable(m, Xi[begin] <= xi <= Xi[end])
    vio_c1 = -(x1 - xi + 25. * y2A + 25. * y2B)
    JuMP.@objective(m, Max, vio_c1)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(xi)
end
function new_leaves_obj(xi, y2A, y2B, z)
    m = JumpModel()
    JuMP.@variable(m, Xi[begin] <= xi <= Xi[end])
    vio_obj = 50. * x1 + 65. * (x1 - xi + 25. * y2A + 25. * y2B) + 1500. * y2A + 1875. * y2B - z
    JuMP.@objective(m, Max, vio_obj)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(xi)
end
function grow_new_leaves(xi, y2A, y2B, z) # ✏️ set_A_gen
    leaves = Float64[]
    push!(leaves, new_leaves_c1(xi, y2A, y2B, z))
    push!(leaves, new_leaves_obj(xi, y2A, y2B, z))
    leaves
end
function get_node_template(parent::Int, id::Int, xi::Float64)
    Dict{String, Union{Float64, Int, Vector{Int}}}("id" => id, "xi" => xi, "parent" => parent)
end


global_logger(ConsoleLogger(Debug))
GRB_ENV = Gurobi.Env()
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))

Xi = [5., 95.] # the Ξ
# build tree
node_dling = 0
root_node = get_node_template(-1, node_dling, 50.)
tree = Dict(root_node["id"] => root_node) # create the tree

# solve SRO
xi_snf = deepcopy(Xi)
x1, y2A, y2B, z = SRO_master(xi_snf)
maxvio, xi = SRO_subproblem(x1, y2A, y2B, z)
@assert maxvio <= 1e-6 "SRO problem not convergent yet"
# grow leaves
set_A = grow_new_leaves(xi, y2A, y2B, z)
tree[node_dling]["children"] = collect(node_dling .+ eachindex(set_A)) # add children id's to the primal leaf
for (ind, id) in enumerate(tree[node_dling]["children"]) # add info to the current leaf
    tree[id] = get_node_template(node_dling, id, set_A[ind])
end
# calculate lower(dual) bound
xi_snf = [50.; set_A] # hardcode
lb = get_lb_of_ARO(xi_snf)

# the 2nd iteration
# do Vor partition



import Gurobi
import JuMP
import LinearAlgebra
import OffsetArrays
using Logging

# solve 2S-Mi-ARO with the Ξ-partition method [Bertsimas 2016]
# 21/3/24
# Gurobi callback is exploited to solve a SRO problem
# instance in section 2.4 of "multistage robust mixed-integer optimization with adaptive partitions"

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function is_leaf_new(list, xi)
    for l in list
        isapprox(l, xi; atol = 1e-5, norm = v -> LinearAlgebra.norm(v, 1)) && return false
    end
    return true
end
function normalize(slope::Vector, rhs::Float64) # ~ the coefficients of a cut
    k = LinearAlgebra.norm(slope, 1) + abs(rhs)
    slope / k, rhs / k
end
function collect_layer_nodes(layer)
    vec = Int[]
    for node in tree_vec
        node["layer"] == layer && push!(vec, node["id"])
    end
    vec
end
function subproblem_K_adapt(Xi, x1, y2A, y2B, z)
    m = JumpModel()
    JuMP.@variable(m, xi[eachindex(Xi["slope"][1])])
    for (slope, rhs) in zip(Xi["slope"], Xi["rhs"])
        JuMP.@constraint(m, slope' * xi <= rhs)
    end
    vio_len = 2
    JuMP.@variable(m, vio[1:vio_len])
    JuMP.@constraint(m, vio[1] == -(x1 - xi[1] + 25. * (y2A + y2B)))
    JuMP.@constraint(m, vio[vio_len] == 50. * x1 + 65. * (x1 - xi[1] + 25. * (y2A + y2B)) + 1500. * y2A + 1875. * y2B - z)
    JuMP.@variable(m, maxvio)
    errcode = Gurobi.GRBaddgenconstrMax(JuMP.backend(m), "", column(maxvio), Cint(vio_len), column.(vio), Cdouble(0.))
    @assert errcode == 0 "Gurobi's Max constr fail"
    JuMP.@objective(m, Max, maxvio)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(maxvio), JuMP.value.(xi)
end
function K_adapt(node_id_vec)
    K = length(node_id_vec)
    m = JumpModel()
    JuMP.@variable(m, x1 >= 0.)
    JuMP.@variable(m, y2A[1:K], Bin)
    JuMP.@variable(m, y2B[1:K], Bin)
    JuMP.@variable(m, z[1:K])
    JuMP.@variable(m, max_z)
    JuMP.@objective(m, Min, max_z)
    JuMP.@constraint(m, [k in 1:K], max_z >= z[k])
    for k in 1:K # cf. ensuing
        xi = tree_vec[node_id_vec[k]]["xi"] # the nominal xi
        JuMP.@constraint(m, x1 - xi[1] + 25. * (y2A[k] + y2B[k]) >= 0.)
        JuMP.@constraint(m, 50. * x1 + 65. * (x1 - xi[1] + 25. * (y2A[k] + y2B[k])) + 1500. * y2A[k] + 1875. * y2B[k] <= z[k])
    end
    function my_callback_function(cb_data, cb_where::Cint)
        if cb_where == 4
            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            x1t = JuMP.callback_value(cb_data, x1)
            y2At = JuMP.callback_value.(cb_data, y2A)
            y2Bt = JuMP.callback_value.(cb_data, y2B)
            zt = JuMP.callback_value.(cb_data, z)
            for k in 1:K # cf. foregoing
                vio, xi = subproblem_K_adapt(tree_vec[node_id_vec[k]]["Xi"], x1t, y2At[k], y2Bt[k], zt[k])
                if vio > 1e-5
                    JuMP.MOI.submit(m, JuMP.MOI.LazyConstraint(cb_data), JuMP.@build_constraint(
                        x1 - xi[1] + 25. * (y2A[k] + y2B[k]) >= 0.
                    ))
                    JuMP.MOI.submit(m, JuMP.MOI.LazyConstraint(cb_data), JuMP.@build_constraint(
                        50. * x1 + 65. * (x1 - xi[1] + 25. * (y2A[k] + y2B[k])) + 1500. * y2A[k] + 1875. * y2B[k] <= z[k]
                    ))
                end
            end
        end
    end
    JuMP.MOI.set(m, JuMP.MOI.RawOptimizerAttribute("LazyConstraints"), 1)
    JuMP.MOI.set(m, Gurobi.CallbackFunction(), my_callback_function)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(x1), JuMP.value.(y2A), JuMP.value.(y2B), JuMP.value.(z), JuMP.value(max_z)
end
function lb_program(xi_vec_global)
    K = length(xi_vec_global)
    m = JumpModel()
    JuMP.@variable(m, x1 >= 0.)
    JuMP.@variable(m, y2A[1:K], Bin)
    JuMP.@variable(m, y2B[1:K], Bin)
    JuMP.@variable(m, z[1:K])
    JuMP.@variable(m, max_z)
    JuMP.@objective(m, Min, max_z)
    JuMP.@constraint(m, [k in 1:K], max_z >= z[k])
    for (k, xi) in enumerate(xi_vec_global)
        JuMP.@constraint(m, x1 - xi[1] + 25. * (y2A[k] + y2B[k]) >= 0.)
        JuMP.@constraint(m, 50. * x1 + 65. * (x1 - xi[1] + 25. * (y2A[k] + y2B[k])) + 1500. * y2A[k] + 1875. * y2B[k] <= z[k])
    end
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    JuMP.value(max_z)
end
function write_solution(node_id_vec, x1, y2A, y2B, z)
    for (k, node_id) in enumerate(node_id_vec)
        tree_vec[node_id]["solution"] = Dict( # record the solution of the K-adapt problem
            "x1" => x1,
            "y2A" => y2A[k],
            "y2B" => y2B[k],
            "z" => z[k]
        )
    end
end
function grow_leaves_at(node_id)
    node = tree_vec[node_id]
    solution, Xi = node["solution"], node["Xi"]
    x1, y2A, y2B, z = solution["x1"], solution["y2A"], solution["y2B"], solution["z"]
    m = JumpModel() # ✏️ eq.(9)
    JuMP.@variable(m, xi[eachindex(Xi["slope"][1])])
    for (slope, rhs) in zip(Xi["slope"], Xi["rhs"])
        JuMP.@constraint(m, slope' * xi <= rhs)
    end
    vio = -(x1 - xi[1] + 25. * (y2A + y2B))
    JuMP.@objective(m, Max, vio)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    leaf_xi_list = [JuMP.value.(xi)] # the 1st one
    vio = 50. * x1 + 65. * (x1 - xi[1] + 25. * (y2A + y2B)) + 1500. * y2A + 1875. * y2B - z
    JuMP.@objective(m, Max, vio)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    new_xi = JuMP.value.(xi)
    is_leaf_new(leaf_xi_list, new_xi) && push!(leaf_xi_list, new_xi) # >= 2nd leaf
    # grow new leaves for the current leaf indexed by `node_id`
    node["child"] = Int[]
    for xi in leaf_xi_list
        new_id = length(tree_vec) # ∵ it's index is 0 ab initio
        push!(tree_vec, Dict(
            "id" => new_id,
            "xi" => xi,
            "layer" => tree_vec[node_id]["layer"] + 1,
            "parent" => node_id # verbalize
        ))
        push!(node["child"], new_id) # echo
    end
end
function upd_xi_vec_global(layer)
    for node_id in collect_layer_nodes(layer)
        new_xi = tree_vec[node_id]["xi"]
        is_leaf_new(xi_vec_global, new_xi) && push!(xi_vec_global, new_xi) # >= 2nd leaf
    end
end
function next_layer_Xi_endow(node_id_vec)
    for parent_id in node_id_vec
        siblings = tree_vec[parent_id]["child"]
        for leaf_id in siblings
            tree_vec[leaf_id]["Xi"] = deepcopy(tree_vec[parent_id]["Xi"]) # inherit
            for adversary_id in setdiff(siblings, leaf_id)
                xia, xil = tree_vec[adversary_id]["xi"], tree_vec[leaf_id]["xi"]
                slope = xia - xil
                rhs = slope' * (xia + xil) / 2 # ✏️ Voronoi
                slope, rhs = normalize(slope, rhs)
                push!(tree_vec[leaf_id]["Xi"]["id"], length(tree_vec[leaf_id]["Xi"]["id"]) + 1)
                push!(tree_vec[leaf_id]["Xi"]["slope"], slope)
                push!(tree_vec[leaf_id]["Xi"]["rhs"], rhs)
            end
        end
    end
end

global_logger(ConsoleLogger(Debug))
GRB_ENV = Gurobi.Env()
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))

tree_vec = [Dict(
        "id" => 0, # a null node
        "layer" => 0,
        "child" => [1]
    ), Dict(
        "id" => 1, # the 1-block (i.e. no) partition 
        "layer" => 1, # layer 1 comprise only this node
        "xi" => [5.], # is a nominal xi at the same time
        "Xi" => Dict(
            "id" => [1, 2], # every entry 1-1 a cut describing the partition
            "slope" => [[1/96], [-1/6]],
            "rhs" => [95/96, -5/6]
        )
    )
]
tree_vec = OffsetArrays.OffsetVector(tree_vec, -1)
xi_vec_global = [tree_vec[1]["xi"]] # globally identified significant scenes, TBUPD

for layer in 1:typemax(Int)
    node_id_vec = collect_layer_nodes(layer) # current partitions
    x1, y2A, y2B, z, max_z = K_adapt(node_id_vec) # solve the K-adapt problem
    write_solution(node_id_vec, x1, y2A, y2B, z)
    grow_leaves_at.(node_id_vec)
    upd_xi_vec_global(layer + 1)
    ub, lb = max_z, lb_program(xi_vec_global)
    rel_gap = abs(ub - lb) / abs(ub)
    @info "▶ layer = $layer" ub lb rel_gap
    if rel_gap < 0.01/100
        @info "😊 Ξ-partition method convergent" x1 max_z num_of_parts=length(z)
        break
    end
    next_layer_Xi_endow(node_id_vec)
end

# 📕 test results
# ┌ Info: ▶ layer = 14
# │   ub = 7250.3051757812555
# │   lb = 7249.904576511365
# └   rel_gap = 5.5252745943507e-5
# ┌ Info: 😊 Ξ-partition method convergent
# │   x1 = 45.00610351562503
# │   max_z = 7250.3051757812555
# └   num_of_parts = 8192

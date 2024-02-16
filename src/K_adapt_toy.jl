import Gurobi
import JuMP
using Logging
# solving Example 1 [K-adapt, Subramanyam2020] via the B&B algorithm  
# 2024/02/16

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
Np, N2, L = 2, 4, 8
K = 2

node_list = [ [[Float64[1, 1]], Vector{Float64}[]] , [Vector{Float64}[], [Float64[1, 1]]] ]
# node_list[1] # choose a node #1
# node_list[2][1] # list of significant scenes handled by plan #1
# node_list[3][2][1] :: Vector{Float64} # the 1st significant scene handled by plan #2

function algorithm()
    incumbent = Dict(
        "th" => Inf,
        "w" => zeros(N2, 1+Np, K)
    )
    for ite in 1:typemax(Int)
        node = popfirst!(node_list) # the selected node in step 3 
        m = JumpModel() # (6)
        JuMP.@variable(m, -3. <= w[iy = 1:N2, ixi = 1:1+Np, k = 1:K] <= 3.) # weights about y
        JuMP.@variable(m, th)
        for (k, Xi_k) in enumerate(node)
            for xi in Xi_k
                std_basis = [1. ; xi]
                y1 = w[1, :, k]' * std_basis
                y2 = w[2, :, k]' * std_basis
                y3 = w[3, :, k]' * std_basis
                y4 = w[4, :, k]' * std_basis
                y = [y1, y2, y3, y4]
                JuMP.@constraint(m, th >= sum(y))
                JuMP.@constraint(m, y[1] >= xi[1] + xi[2])
                JuMP.@constraint(m, y[2] >= xi[1] - xi[2])
                JuMP.@constraint(m, y[3] >= -xi[1] + xi[2])
                JuMP.@constraint(m, y[4] >= -xi[1] - xi[2])
                JuMP.@constraint(m, [i = 1:N2], y[i] >= 0.)
            end
        end
        JuMP.@objective(m, Min, th)
        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.OPTIMAL
        th_hat, w_hat = JuMP.value(th), JuMP.value.(w)
        @debug "ite = $ite, dealing with" node th_hat incumbent["th"]
        if th_hat < incumbent["th"] - 1e-6 # this node is still promising
            m = JumpModel() # (8)
            JuMP.@variable(m, -1. <= xi[1:Np] <= 1.)
            JuMP.@variable(m, vio[1:K, 1:1+L])
            JuMP.@variable(m, vio_at_k[1:K])
            JuMP.@variable(m, phi)
            for k in 1:K
                y = w_hat[:, :, k] * [1. ;xi]
                JuMP.@constraint(m, vio[k, 1] == sum(y) - th_hat)
                JuMP.@constraint(m, vio[k, 2] == xi[1] + xi[2] - y[1])
                JuMP.@constraint(m, vio[k, 3] == xi[1] - xi[2] - y[2])
                JuMP.@constraint(m, vio[k, 4] == -xi[1] + xi[2] - y[3])
                JuMP.@constraint(m, vio[k, 5] == -xi[1] - xi[2] - y[4])
                JuMP.@constraint(m, [i = 1:N2], vio[k, 5+i] == -y[i])
                errcode_norm = Gurobi.GRBaddgenconstrMax(JuMP.backend(m), "", column(vio_at_k[k]), Cint(1+L), column.(vio[k, :]), Cdouble(0.))
                @assert errcode_norm == 0 "Gurobi's Max constr fail"
                JuMP.@constraint(m, phi <= vio_at_k[k])
            end
            JuMP.@objective(m, Max, phi)
            JuMP.optimize!(m)
            @assert JuMP.termination_status(m) == JuMP.OPTIMAL
            phi_hat = JuMP.value(phi)
            if phi_hat > 1e-6 # separation success
                xi_hat = JuMP.value.(xi) # the new significant scene
                for k in 1:K
                    n = deepcopy(node)
                    push!(n[k], xi_hat)
                    push!(node_list, n) 
                end
                continue
            else # update incumbent solution
                @info "incumbent upd @ ite = $ite ⋅ val = $th_hat" phi=phi_hat node
                incumbent["th"] = th_hat
                incumbent["w"] .= w_hat
            end
        end
        if isempty(node_list) # step 2
            @info "node_list is empty, thus return 😊"
            return incumbent
        end
    end
end

incumbent = algorithm()

@info "the value" incumbent["th"]
@info "the solution" incumbent["w"]


import Gurobi
import JuMP
using Logging
import Random
import Distributions

# capital budgeting primal side
# using 2-stage rule => K-adapt approx method
# 15/3/24

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end

global_logger(ConsoleLogger(Debug))
GRB_ENV = Gurobi.Env()
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))

I, T, L = 5, 3, 12 # L is the num of CONSTRs involved in vio_calculation
K = 2 # K = 2, th = -5.59962
B_ini = 10.
# t = 1 is deterministic, variables are all recourse @ t = 3, therefore we only need to apply ADR to x^s_2
c_loan = t -> .8 * 1.2 ^ (t-1)
# node_list initialization
initial_xi = ones(I, T-1, 2) # the last index [1]: δᶜ, [2]: δʳ
Xi_1 = [initial_xi]
Xi_empty = typeof(initial_xi)[]
initial_node = typeof(Xi_1)[[Xi_1]; [Xi_empty for _ in 1:K-1]]
node_list = [initial_node]

Random.seed!(9999)
c0,     sigma   = rand(Distributions.Uniform(0, 10), I, T), rand(Distributions.Uniform(0, 5), I, T) # test case generation
ttmp = 1
c_1,    r_1     = c0[:, ttmp] + rand(5) .* sigma[:, ttmp],  c0[:, ttmp] / 5 + rand(5) .* sigma[:, ttmp] # 1st stage uncertainty is fixed

incumbent = Dict(
    "D" => Dict{String, Any}(),
    "th" => Inf
)

for ite in 1:typemax(Int)
    node = popfirst!(node_list) # the selected node in step 3 
    m = JumpModel() # (6)
    # inherent 1st stage
    JuMP.@variable(m, C1)
    JuMP.@variable(m, B1)
    JuMP.@variable(m, L1 >= 0.)
    JuMP.@variable(m, x1[1:I], Bin)
    JuMP.@constraint(m, B1 == B_ini + L1) # ✏️
    JuMP.@constraint(m, C1 == c_1' * x1) # ✏️
    JuMP.@constraint(m, C1 <= B1) # ✏️ these 3 constrs are not ξ related
    # ·  ·  2S decision rule induced weights (pre-prepared in 1st stage)
    JuMP.@variable(m, w_B2[1:2*I])
    JuMP.@variable(m, b_B2)
    JuMP.@variable(m, w_C2[1:2*I])
    JuMP.@variable(m, b_C2)
    # · K-adapt
    # inherent recourse plans
    JuMP.@variable(m, x2[1:I, 1:K], Bin)
    JuMP.@variable(m, x3[1:I, 1:K], Bin)
    # K-ADR induced weights
    JuMP.@variable(m, kw_L2[1:2*I, 1:K])
    JuMP.@variable(m, kb_L2[1:K])
    JuMP.@variable(m, kw_B3[1:2*2*I, 1:K])
    JuMP.@variable(m, kb_B3[1:K])
    JuMP.@variable(m, kw_C3[1:2*2*I, 1:K])
    JuMP.@variable(m, kb_C3[1:K])
    JuMP.@variable(m, kw_L3[1:2*2*I, 1:K])
    JuMP.@variable(m, kb_L3[1:K])
    # ·
    JuMP.@variable(m, th) # trial value
    for (k, Xi_k) in enumerate(node)
        for xi in Xi_k # xi = ones(I, T-1, 2), e.g.
            v2xi, v23xi = reshape(xi[:, 1, :], (:,)), reshape(xi, (:,))
            B2 = v2xi' * w_B2 + b_B2
            C2 = v2xi' * w_C2 + b_C2
            L2 = v2xi' * kw_L2[:, k] + kb_L2[k]
            B3 = v23xi' * kw_B3[:, k] + kb_B3[k]
            C3 = v23xi' * kw_C3[:, k] + kb_C3[k]
            L3 = v23xi' * kw_L3[:, k] + kb_L3[k]
            # constrs (#8 if use ==, #12 if use all <=)
            ttmp = 2
            c_2 = c0[:, ttmp] + xi[:, ttmp-1, 1] .* sigma[:, ttmp] # model assumption: affine dependency
            JuMP.@constraint(m, C2 == c_2' * x2[:, k])
            JuMP.@constraint(m, B2 == B1 - C1 + L2)
            JuMP.@constraint(m, C2 <= B2)
            JuMP.@constraint(m, L2 >= 0.)
            ttmp = 3
            c_3 = c0[:, ttmp] + xi[:, ttmp-1, 1] .* sigma[:, ttmp] # model assumption: affine dependency
            JuMP.@constraint(m, C3 == c_3' * x3[:, k])
            JuMP.@constraint(m, B3 == B2 - C2 + L3)
            JuMP.@constraint(m, C3 <= B3)
            JuMP.@constraint(m, L3 >= 0.)
            # obj
            ttmp = 2
            r_2 = c0[:, ttmp] / 5 + xi[:, ttmp-1, 2] .* sigma[:, ttmp] # model assumption: affine dependency
            ttmp = 3
            r_3 = c0[:, ttmp] / 5 + xi[:, ttmp-1, 2] .* sigma[:, ttmp] # model assumption: affine dependency
            JuMP.@constraint(m, th >= c_loan(1) * L1 - r_1' * x1 + c_loan(2) * L2 - r_2' * x2[:, k] + c_loan(3) * L3 - r_3' * x3[:, k])
        end
    end
    JuMP.@objective(m, Min, th)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    hatD = Dict(
        "x1" => JuMP.value.(x1),
        "x2" => JuMP.value.(x2),
        "x3" => JuMP.value.(x3),
        "B1" => JuMP.value(B1),
        "C1" => JuMP.value(C1),
        "L1" => JuMP.value(L1),
        "b_B2" => JuMP.value(b_B2),
        "b_C2" => JuMP.value(b_C2),
        "kb_L2" => JuMP.value.(kb_L2),
        "w_B2" => JuMP.value.(w_B2),
        "w_C2" => JuMP.value.(w_C2),
        "kw_L2" => JuMP.value.(kw_L2),
        "kb_B3" => JuMP.value.(kb_B3),
        "kb_C3" => JuMP.value.(kb_C3),
        "kb_L3" => JuMP.value.(kb_L3),
        "kw_B3" => JuMP.value.(kw_B3),
        "kw_C3" => JuMP.value.(kw_C3),
        "kw_L3" => JuMP.value.(kw_L3),
        "th" => JuMP.value(th)
    )
    @debug "ite = $ite, dealing with" node th=hatD["th"] x1=hatD["x1"]
    if hatD["th"] < incumbent["th"] - 1e-6 # this node is still promising
        m = JumpModel() # (8)
        JuMP.@variable(m, -1. <= xi[1:I, 1:T-1, 1:2] <= 1.) # the last index [1]: super^c, [2]: super^r
        JuMP.@variable(m, vio[1:K, 1:L+1])
        JuMP.@variable(m, vio_at_k[1:K])
        JuMP.@variable(m, phi)
        ttmp = 2
        c_2 = c0[:, ttmp] + xi[:, ttmp-1, 1] .* sigma[:, ttmp] # model assumption: affine dependency
        ttmp = 3
        c_3 = c0[:, ttmp] + xi[:, ttmp-1, 1] .* sigma[:, ttmp] # model assumption: affine dependency
        ttmp = 2
        r_2 = c0[:, ttmp] / 5 + xi[:, ttmp-1, 2] .* sigma[:, ttmp] # model assumption: affine dependency
        ttmp = 3
        r_3 = c0[:, ttmp] / 5 + xi[:, ttmp-1, 2] .* sigma[:, ttmp] # model assumption: affine dependency 
        for k in 1:K
            v2xi, v23xi = reshape(xi[:, 1, :], (:,)), reshape(xi, (:,))
            B2 = hatD["w_B2"]' * v2xi + hatD["b_B2"]
            C2 = hatD["w_C2"]' * v2xi + hatD["b_C2"]
            L2 = hatD["kw_L2"][:, k]' * v2xi + hatD["kb_L2"][k]
            B3 = hatD["kw_B3"][:, k]' * v23xi + hatD["kb_B3"][k]
            C3 = hatD["kw_C3"][:, k]' * v23xi + hatD["kb_C3"][k]
            L3 = hatD["kw_L3"][:, k]' * v23xi + hatD["kb_L3"][k]
            JuMP.@constraint(m, vio[k, 1] == C2 - c_2' * hatD["x2"][:, k])
            JuMP.@constraint(m, vio[k, 2] == -C2 + c_2' * hatD["x2"][:, k])
            JuMP.@constraint(m, vio[k, 3] == B2 - (hatD["B1"] - hatD["C1"] + L2))
            JuMP.@constraint(m, vio[k, 4] == -B2 + (hatD["B1"] - hatD["C1"] + L2))
            JuMP.@constraint(m, vio[k, 5] == C2 - B2)
            JuMP.@constraint(m, vio[k, 6] == -L2)
            JuMP.@constraint(m, vio[k, 7] == C3 - c_3' * hatD["x3"][:, k])
            JuMP.@constraint(m, vio[k, 8] == -C3 + c_3' * hatD["x3"][:, k])
            JuMP.@constraint(m, vio[k, 9] == B3 - (B2 - C2 + L3))
            JuMP.@constraint(m, vio[k, 10] == -B3 + (B2 - C2 + L3))
            JuMP.@constraint(m, vio[k, 11] == C3 - B3)
            JuMP.@constraint(m, vio[k, L] == -L3)
            JuMP.@constraint(m, vio[k, L+1] == c_loan(1) * hatD["L1"] - r_1' * hatD["x1"] + c_loan(2) * L2 - r_2' * hatD["x2"][:, k] + c_loan(3) * L3 - r_3' * hatD["x3"][:, k] - hatD["th"])
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
            ind_tobeadd = findfirst(isempty, node)
            endind = (ind_tobeadd != nothing) ? ind_tobeadd : K
            for k in 1:endind
                n = deepcopy(node)
                push!(n[k], xi_hat)
                push!(node_list, n)
            end
            continue
        else # update incumbent solution # to be changeed in 03/15
            incumbent["th"] = hatD["th"]
            incumbent["D"] = hatD
            @info "incumbent upd @ ite = $ite ⋅ val = $(incumbent["th"])" phi=phi_hat node
        end
    end
    if isempty(node_list) # step 2
        @info "node_list is empty, thus return 😊"
        return incumbent
    end
end


import JuMP
import Gurobi

# 2S-ARO C&CG operations research letters 2012
# case study in section 4.2 (with no assumption on RCR)
# at initialization phase, we generate an initial scene and adopt Q3 to GC&C
# later, we use Q to GC&C, provided that optimality is attained
# after applying Q3 once on initialization and then applying Q twice, we converge with UB = 13874 + 16662 = 30536 = LB
# 03/02/24

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function Q3(z, d; w::Bool)
    m = JumpModel()
    JuMP.@variable(m, x[1:3, 1:3] >= 0.)
    JuMP.@constraint(m, [i = 1:3], sum(x[i, :]) <= z[i])
    JuMP.@constraint(m, [j = 1:3], sum(x[:, j]) >= d[j])
    JuMP.@objective(m, Min, sum(c_mat .* x))
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    if status == JuMP.INFEASIBLE
        objval = "infeasible"
        w && push!(is_opt_vec, false)
        w && push!(scene_vec, d)
        return objval
    elseif status == JuMP.OPTIMAL
        objval = JuMP.objective_value(m)
        return objval
    else
        error("Inmost: $status")
    end
end
function Q(z; f::Bool, w::Bool)
    m = JumpModel()
    JuMP.unset_silent(m)
    JuMP.@variable(m, 0. <= g[1:3] <= 1.)
    JuMP.@constraint(m, sum(g) <= 1.8)
    JuMP.@constraint(m, g[1] + g[2] <= 1.2)
    d = 40. * g + d_bias_vec # 𝔘 till this line
    JuMP.@variable(m,     x[i = 1:3, j = 1:3] >= 0.)
    JuMP.@variable(m, c_pnt[i = 1:3, j = 1:3] >= 0.) # coefficient of `x` after penalizing, `>= 0` implies dual feas.
    f || JuMP.@objective(m, Max, sum(c_mat .* x)) # (15)
    JuMP.@variable(m, p_z[i = 1:3] >= 0.)
    JuMP.@variable(m, d_z[i = 1:3] >= 0.)
    JuMP.@constraint(m, [i = 1:3], p_z[i] == z[i] - sum(x[i, :]))
    JuMP.@constraint(m, [i = 1:3], [p_z[i], d_z[i]] in JuMP.SOS1())
    JuMP.@variable(m, p_d[j = 1:3] >= 0.)
    JuMP.@variable(m, d_d[j = 1:3] >= 0.)
    JuMP.@constraint(m, [j = 1:3], p_d[j] == sum(x[:, j]) - d[j])
    JuMP.@constraint(m, [j = 1:3], [p_d[j], d_d[j]] in JuMP.SOS1())
    JuMP.@constraint(m, [i = 1:3, j = 1:3], c_pnt[i, j] == c_mat[i, j] + d_z[i] - d_d[j])
    JuMP.@constraint(m, [i = 1:3, j = 1:3], [c_pnt[i, j], x[i, j]] in JuMP.SOS1()) # (19)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    if status == JuMP.INFEASIBLE
        return "problem (15) infeasible"
    elseif status == JuMP.OPTIMAL
        opt_scene = JuMP.value.(d)
        objval = JuMP.objective_value(m)
        if f
            return opt_scene
        else
            w && push!(is_opt_vec, true)
            w && push!(scene_vec, opt_scene)
            return objval, opt_scene
        end
    elseif status == JuMP.INFEASIBLE_OR_UNBOUNDED
        @warn "here" JuMP.result_count(m)
        objval = JuMP.objective_value(m)
        @warn objval
    else
        error("Q: $status")
    end
end

column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
norm_sense = Cdouble(1.0)
GRB_ENV = Gurobi.Env()

c_mat = [22 33 24; 33 23 30; 20 25 27.]
c_y_vec = [400, 414, 326.]
c_z_vec = [18, 25, 20.]
d_bias_vec = [206, 274, 220.]
is_opt_vec = Bool[]
scene_vec = Vector{Float64}[]
k = -1


k += 1
m = JumpModel() # the master problem
JuMP.@variable(m, y[1:3], Bin)
JuMP.@variable(m, z[1:3] >= 0.)
JuMP.@constraint(m, [i = 1:3], z[i] <= 800. * y[i])
JuMP.@variable(m, th >= 0.)
JuMP.@variable(m, x[1:3, 1:3, 1:k] >= 0.)
for s in 1:k
    d = scene_vec[s]
    JuMP.@constraint(m, [i = 1:3], sum(x[i, :, s]) <= z[i])
    JuMP.@constraint(m, [j = 1:3], sum(x[:, j, s]) >= d[j])
    is_opt_vec[s] && JuMP.@constraint(m, th >= sum(c_mat .* x[:, :, s]))
end
cost_1_stage = c_y_vec' * y + c_z_vec' * z
JuMP.@objective(m, Min, cost_1_stage + th)
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL "Master $(JuMP.termination_status(m))"
JuMP.value(cost_1_stage)
JuMP.objective_value(m)
z = JuMP.value.(z)


m = JumpModel() # to generate an initial scene
JuMP.unset_silent(m)
JuMP.@variable(m, 0. <= g[1:3] <= 1.)
JuMP.@constraint(m, sum(g) <= 1.8)
JuMP.@constraint(m, g[1] + g[2] <= 1.2)
d = 40. * g + d_bias_vec # 𝔘 till this line
JuMP.@objective(m, Max, sum(d))
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL "Uncertainty $(JuMP.termination_status(m))"
d = JuMP.value.(d)
Q3(z, d; w = true)


Q(z; f = false, w = true) 


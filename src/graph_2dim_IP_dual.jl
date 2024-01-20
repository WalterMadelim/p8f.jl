using CairoMakie 
import Gurobi
import JuMP

# a simple illustration to the Lagrangian dual of a 2-dim, 2-constr, pureIP
# 20/01/24

function JumpModel(env = GRB_ENV)
    m = JuMP.direct_model(Gurobi.Optimizer(env))
    JuMP.set_silent(m)
    return m
end

f = Figure();
axs = Axis.([f[i...] for i in Iterators.product([1,2],[1,2])]);
for la in 1:9
    scatter!(axs[1], 1:9, fill(la, 9); markersize = 4, color = :gray)
end
lines!(axs[1], [1.3, 3.4], [8.7, 1.]; color = :olive)
lines!(axs[1], [2.0, 8.7], [1.0, 3.2]; color = :chocolate)

function p2l(x1, y1, x2, y2)
    # cx * x + cy * y == rhs
    cx = y2 - y1
    cy = x1 - x2
    rhs = x1 * (y2-y1) - y1 * (x2-x1)
    -cx, -cy, -rhs # this line can *(-1) simultaneously 
end

cx1, cy1, rhs1 = p2l(1.3, 8.7, 3.4, 1.)
cx2, cy2, rhs2 = p2l(2.0, 1., 8.7, 3.2)


GRB_ENV = Gurobi.Env()
m = JumpModel()
JuMP.@variable(m, x)
JuMP.@variable(m, y)
# JuMP.set_integer(x)
# JuMP.set_integer(y)
JuMP.@constraint(m, c_olive, cx1 * x + cy1 * y >= rhs1)
JuMP.@constraint(m, c_chocolate, cx2 * x + cy2 * y >= rhs2)
JuMP.@objective(m, Min, x + 2.0 * y)
JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.OPTIMAL

pi1 = JuMP.dual(c_olive)
pi2 = JuMP.dual(c_chocolate)
x, y = JuMP.value(x), JuMP.value(y)
val_LP = JuMP.objective_value(m) # 6.1287671232876715
val_IP = JuMP.objective_value(m) # 8.0
# scatter!(axs[1], [x], [y]; markersize = 4, color = :cyan)
scatter!(axs[1], [x], [y]; markersize = 4, color = :cyan)
text!(axs[1], [x], [y]; text = "$val_LP", fontsize = 8, color = :tomato)

# obj_coeff, after dualizing c_olive
tmp1 = 1.0 - pi1 * cx1
tmp2 = 2.0 - pi1 * cy1
@assert isapprox(tmp2/tmp1, cy2/cx2; atol = 1e-6) # this means that for 2-dim 2-constr pure IP, dualizing one constraint gives a tight dual bound 
# of course we can also dualizing 2 constraint, the tightness remains the same

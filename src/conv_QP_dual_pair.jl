import LinearAlgebra
import MathOptInterface as MOI
import Gurobi

# conv_QP_dual_problems
# 06/11/23
# to check if A is psd: we need to check all its eigenvalues
# But we need to exclude Float_Point_Error!

function pri_solve(decision_dim)
    @info "primal_conv_QP"
    o = Gurobi.Optimizer()
    x = MOI.add_variables(o,decision_dim)
    cis = MOI.add_constraints(o, A * x, MOI.LessThan.(b))
    f = x' * half_Q * x + c' * x
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    println("Max_Vio: $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
    obj_val = MOI.get(o, MOI.ObjectiveValue())
    @info "" obj_val
    variable_primal = MOI.get.(o, MOI.VariablePrimal(), x)
    @info "" variable_primal
    constraint_dual = MOI.get.(o, MOI.ConstraintDual(), cis)
    @info "" constraint_dual
end

function dual_solve(decision_dim)
    @info "dual_conv_QP"
    o = Gurobi.Optimizer()
    m = MOI.add_variables(o,decision_dim)
    cis = MOI.add_constraints(o, m, MOI.GreaterThan.(0.))
    f = m' * -half_P * m - t' * m - cnst_dual
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    println("Max_Vio: $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
    obj_val = MOI.get(o, MOI.ObjectiveValue())
    @info "" obj_val
    variable_primal = MOI.get.(o, MOI.VariablePrimal(), m)
    @info "" variable_primal
    constraint_dual = MOI.get.(o, MOI.ConstraintDual(), cis)
    @info "" constraint_dual
end

# primal_side_data
half_Q = [1. .5 0.;
.5 1. .5;
0. .5 1.]
c = [2., 0., 0.]
A = -[1. 2. 3.;
1. 1. 0.;
1. 0. 0.;
0. 1. 0.;
0. 0. 1.]
b = -[4.,1.,0.,0.,0.]

# check the psd (or pd) property
eigenvalues = LinearAlgebra.eigvals(half_Q);
@info "" eigenvalues

# dual_side_data
tmp = inv(half_Q)
half_P = .25 * A * tmp * A';
@info "" half_P
half_P = round.(half_P;digits = 10);
@info "" half_P
eigenvalues = LinearAlgebra.eigvals(half_P);
@warn "" eigenvalues
t = b .+ .5 * A * tmp * c
cnst_dual = .25 * c' * tmp * c

# go to check the strong duality and the dual_variables
pri_solve(size(A,2))
dual_solve(size(A,1))

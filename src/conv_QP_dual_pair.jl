import LinearAlgebra
import MathOptInterface as MOI
import Gurobi

# conv_QP_dual_problems
# primal_side matrix Q must be Positive Definite!
# p437 nonlinear programming, Bertsekas
# 06/11/23
# to check if A is psd: we need to check all its eigenvalues
# But we need to exclude Float_Point_Error!

function pri_solve(half_Q,c,A,b)
    dim = size(A,2)
    pri_solve(dim,half_Q,c,A,b)
end

function pri_solve(decision_dim,half_Q,c,A,b) # inner function
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

function dual_solve(half_Q::Matrix,c,A,b)
    # check the psd (or pd) property
    eigenvalues = LinearAlgebra.eigvals(half_Q);
    @info "" eigenvalues
    # dual_side_data
    inv_of_halfQ = inv(half_Q)
    @info "" inv_of_halfQ
    half_P = .25 * A * inv_of_halfQ * A';
    @info "" half_P
    half_P = round.(half_P;digits = 10);
    @info "" half_P
    eigenvalues = LinearAlgebra.eigvals(half_P);
    @warn "" eigenvalues
    t = b .+ .5 * A * inv_of_halfQ * c
    cnst_dual = .25 * c' * inv_of_halfQ * c
    dim = length(t)
    dual_solve(dim,half_P,t,cnst_dual)
    return inv_of_halfQ, half_P, t, cnst_dual
end

function dual_solve(decision_dim::Int,half_P,t,cnst_dual) # inner function
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

pri_solve(half_Q,c,A,b)
inv_of_halfQ, half_P, t, cnst_dual = dual_solve(half_Q,c,A,b)

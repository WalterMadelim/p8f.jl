import LinearAlgebra
import MathOptInterface as MOI
import Gurobi

# General (Q is PSD, not only restrict to PD) dual programs, in the form of p422, book linearprogramming
# 07/11/23

function p24_3(c,hQ,A,b)
    # form (24.3) in book LinearProgramming
    dim_x_z = length(c)
    dim_y = length(b)
    @assert size(A,1) == dim_y
    @assert size(A,2) == dim_x_z
    @assert size(hQ,1) == size(hQ,2) == dim_x_z
    pri_obj, pri_x, pri_w, pri_sen = p24_3(dim_x_z,dim_y,c,hQ,A,b)
    return pri_obj, pri_x, pri_w, pri_sen
end

function p24_3(dim_x_z,dim_y,c,hQ,A,b)
    o = Gurobi.Optimizer()
    x = MOI.add_variables(o, dim_x_z)
    w = MOI.add_variables(o, dim_y)
    cis_eq = MOI.add_constraints(o, A * x .- w, MOI.EqualTo.(b))
    cis_bnd_x = MOI.add_constraints(o, x, MOI.GreaterThan.(0.))
    cis_bnd_w = MOI.add_constraints(o, w, MOI.GreaterThan.(0.))
    f = c' * x + x' * hQ * x
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    println("Max_Vio: $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
    pri_obj_val = MOI.get(o, MOI.ObjectiveValue())
    @info "" pri_obj_val
    variable_primal_x = MOI.get.(o, MOI.VariablePrimal(), x)
    @info "" variable_primal_x
    variable_primal_w = MOI.get.(o, MOI.VariablePrimal(), w)
    @info "" variable_primal_w
    constraint_dual_eq = MOI.get.(o, MOI.ConstraintDual(), cis_eq)
    @info "" constraint_dual_eq
    constraint_dual_bnd_x = MOI.get.(o, MOI.ConstraintDual(), cis_bnd_x)
    @info "" constraint_dual_bnd_x
    constraint_dual_bnd_w = MOI.get.(o, MOI.ConstraintDual(), cis_bnd_w)
    @info "" constraint_dual_bnd_w
    return pri_obj_val, variable_primal_x, variable_primal_w, constraint_dual_eq
end

function d24_3(c,hQ,A,b)
    # p422 in book LinearProgramming
    dim_x_z = length(c)
    dim_y = length(b)
    @assert size(A,1) == dim_y
    @assert size(A,2) == dim_x_z
    @assert size(hQ,1) == size(hQ,2) == dim_x_z
    Q = 2 * hQ
    dual_obj, dual_x, dual_y, dual_z = d24_3(dim_x_z,dim_y,c,hQ,Q,A,b)
    return dual_obj, dual_x, dual_y, dual_z
end

function d24_3(dim_x_z,dim_y,c,hQ,Q,A,b)
    o = Gurobi.Optimizer()
    x = MOI.add_variables(o, dim_x_z)
    z = MOI.add_variables(o, dim_x_z)
    y = MOI.add_variables(o, dim_y)
    cis_eq = MOI.add_constraints(o, A' * y .+ z .- Q * x, MOI.EqualTo.(c))
    cis_y_bnd = MOI.add_constraints(o, y, MOI.GreaterThan.(0.))
    cis_z_bnd = MOI.add_constraints(o, z, MOI.GreaterThan.(0.))
    f = b' * y - x' * hQ * x
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    println("Max_Vio: $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
    dual_obj_val = MOI.get(o, MOI.ObjectiveValue())
    @info "" dual_obj_val
    variable_primal_x = MOI.get.(o, MOI.VariablePrimal(), x)
    @info "" variable_primal_x
    variable_primal_z = MOI.get.(o, MOI.VariablePrimal(), z)
    @info "" variable_primal_z
    variable_primal_y = MOI.get.(o, MOI.VariablePrimal(), y)
    @info "" variable_primal_y
    constraint_dual_eq = MOI.get.(o, MOI.ConstraintDual(), cis_eq)
    @info "" constraint_dual_eq
    constraint_dual_y_bnd = MOI.get.(o, MOI.ConstraintDual(), cis_y_bnd)
    @info "" constraint_dual_y_bnd
    constraint_dual_z_bnd = MOI.get.(o, MOI.ConstraintDual(), cis_z_bnd)
    @info "" constraint_dual_z_bnd
    return dual_obj_val, variable_primal_x, variable_primal_y, variable_primal_z
end

function d24_3_force_x_from_primal(c,hQ,A,b,x)
    dim_x_z = length(c)
    dim_y = length(b)
    @assert size(A,1) == dim_y
    @assert size(A,2) == dim_x_z
    @assert size(hQ,1) == size(hQ,2) == dim_x_z
    Q = 2 * hQ
    dual_obj, dual_x, dual_y, dual_z = d24_3_force_x_from_primal(dim_x_z,dim_y,c,hQ,Q,A,b,x)
    return dual_obj, dual_x, dual_y, dual_z
end

function d24_3_force_x_from_primal(dim_x_z,dim_y,c,hQ,Q,A,b,x_target)
    o = Gurobi.Optimizer()
    x = MOI.add_variables(o, dim_x_z)
    z = MOI.add_variables(o, dim_x_z)
    y = MOI.add_variables(o, dim_y)
    cis_eq = MOI.add_constraints(o, A' * y .+ z .- Q * x, MOI.EqualTo.(c))
    cis_y_bnd = MOI.add_constraints(o, y, MOI.GreaterThan.(0.))
    cis_z_bnd = MOI.add_constraints(o, z, MOI.GreaterThan.(0.))
    MOI.add_constraints(o,x,MOI.EqualTo.(x_target))
    f = b' * y - x' * hQ * x
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    println("Max_Vio: $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
    dual_obj_val = MOI.get(o, MOI.ObjectiveValue())
    @info "" dual_obj_val
    variable_primal_x = MOI.get.(o, MOI.VariablePrimal(), x)
    @info "" variable_primal_x
    variable_primal_z = MOI.get.(o, MOI.VariablePrimal(), z)
    @info "" variable_primal_z
    variable_primal_y = MOI.get.(o, MOI.VariablePrimal(), y)
    @info "" variable_primal_y
    constraint_dual_eq = MOI.get.(o, MOI.ConstraintDual(), cis_eq)
    @info "" constraint_dual_eq
    constraint_dual_y_bnd = MOI.get.(o, MOI.ConstraintDual(), cis_y_bnd)
    @info "" constraint_dual_y_bnd
    constraint_dual_z_bnd = MOI.get.(o, MOI.ConstraintDual(), cis_z_bnd)
    @info "" constraint_dual_z_bnd
    return dual_obj_val, variable_primal_x, variable_primal_y, variable_primal_z
end

hQ = [2.5 0.25 0.25 0.0 0.75; 0.25 0.375 0.125 0.25 -0.125; 0.25 0.125 0.375 -0.25 0.125; 0.0 0.25 -0.25 0.5 -0.25; 0.75 -0.125 0.125 -0.25 0.375]
c = [-5.0, -1.5, -1.5, 0.9999999999999999, -0.4999999999999999]
A = [0.07223509471383482 -0.13606533569648416 -0.33897015539830644 -0.17346277513783082 0.054043178848840134; -0.07260791862941796 0.36872632386526594 0.31710588550609675 0.4302316519096806 0.45837108381941993; 0.33046150605990554 -0.3327563188364244 0.49796674272170216 -0.19253880744518104 0.36973508164932223]
b = [-0.08074161653751588, 0.03270452072625307, 0.3097522085972474]

pri_obj, pri_x, pri_w, pri_sen = p24_3(c,hQ,A,b)
if true # we can fix the dual_x to be pri_x, which is optimal to the primal problem
    dual_obj, dual_x, dual_y, dual_z = d24_3_force_x_from_primal(c,hQ,A,b,pri_x) 
else # otherwise we might get a dual opt solution whose dual_x is not equal to pri_x
    dual_obj, dual_x, dual_y, dual_z = d24_3(c,hQ,A,b)
end
# important Observation: if we call the (fixed) dual_x `fx`, and we call the else dual_x `dx`
# then we found:
# vector `hQ * fx`` == `hQ * dx`
# scalar `fx' * hQ * fx` == `dx' * hQ * dx`

println("\ncheck pri_sensitivity and dual_y:")
for (i,e) in zip(pri_sen,dual_y)
    println("$i \t $e")
end
println("\ncheck pri_eq:")
for (i,e) in zip(A * pri_x .- b, pri_w)
    println("$i \t $e")
end

println("\ncheck dual_eq:")
for (i,e) in zip(A' * dual_y .+ dual_z .- 2 * hQ * dual_x, c)
    println("$i \t $e")
end
println("\ncheck pri_obj:")
i,e = c' * pri_x + pri_x' * hQ * pri_x, pri_obj
println("$i \t $e")
println("\ncheck dual_obj")
i,e = b' * dual_y - dual_x' * hQ * dual_x, dual_obj
println("$i \t $e")
println("\ncheck if strong duality hold")
println("\nAt last check primal and dual variable bounds, do it your self!")


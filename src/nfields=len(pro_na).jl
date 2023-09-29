julia> import Gurobi

julia> t = Gurobi.Optimizer
Gurobi.Optimizer

julia> t.
flags       hash        instance    layout      name        parameters  super       
types
julia> nfields(t)
8

julia> propertynames(t)
(:name, :super, :parameters, :types, :instance, :layout, :hash, :flags)

julia> o = t() # o means optimizer in common sense
Set parameter Username
Academic license - for non-commercial use only - expires 2024-08-09
    sense  : minimize
    ...


julia> o.
affine_constraint_info             callback_state
callback_variable_primal           ...
julia> fieldnames(t) # propertynames(o)
(:inner, :env, :params, :needs_update, :silent, :objective_type, :is_objective_set, :objective_sense, :variable_info, :next_column, :columns_deleted_since_last_update, :last_constraint_index, :affine_constraint_info, :quadratic_constraint_info, :sos_constraint_info, :indicator_constraint_info, :name_to_variable, :name_to_constraint_index, :ret_GRBoptimize, :has_unbounded_ray, :has_infeasibility_cert, :enable_interrupts, :callback_variable_primal, :has_generic_callback, :callback_state, :lazy_callback, :user_cut_callback, :heuristic_callback, :generic_callback, :conflict)

julia> nfields(o) # length(propertynames(o))
30

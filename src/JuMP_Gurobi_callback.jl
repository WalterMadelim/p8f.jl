using JuMP, Gurobi

model = direct_model(Gurobi.Optimizer())
@variable(model, 0 <= x <= 2.5, Int)
@variable(model, 0 <= y <= 2.5, Int)
@objective(model, Max, y)
function my_callback_function(cb_data, cb_where::Cint)
    # You can reference variables outside the function as normal
    # You can select where the callback is run
    if cb_where == GRB_CB_MIPSOL || cb_where == GRB_CB_MIPNODE
        # You can query a callback attribute using GRBcbget
        if cb_where == GRB_CB_MIPNODE
            resultP = Ref{Cint}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultP) # retrieve 3rd what from 2nd where
            if resultP[] != GRB_OPTIMAL
                return nothing
            end
        end
        # Before querying `callback_value`, you must call:
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        x_val = callback_value(cb_data, x)
        y_val = callback_value(cb_data, y)
        # You can submit solver-independent MathOptInterface attributes such as
        # lazy constraints, user-cuts, and heuristic solutions.
        if y_val - x_val > 1 + 1e-6
            con = @build_constraint(y - x <= 1)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end
    return nothing
end
# You _must_ set this parameter if using lazy constraints.
MOI.set(model, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
MOI.set(model, Gurobi.CallbackFunction(), my_callback_function)

optimize!(model)


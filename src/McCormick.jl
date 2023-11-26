import MathOptInterface as MOI
import Gurobi
# McCormick envelope for "bl = x * y" is tight, provided that the obj is a linear function of variables
function test_tightness_Mc_bilinear(r)
    o = Gurobi.Optimizer()
    x = MOI.add_variable(o)
    y = MOI.add_variable(o)
    bl = MOI.add_variable(o)
    # # aux
    # z = MOI.add_variable(o)
    # x̃ = MOI.add_variable(o)
    # ỹ = MOI.add_variable(o)
    # z̃ = MOI.add_variable(o)
    # # aux def
    # MOI.add_constraint(o, 1. * x + y - z, MOI.EqualTo(0.))
    # MOI.add_constraint(o, 1. * x * x - x̃, MOI.EqualTo(0.))
    # MOI.add_constraint(o, 1. * y * y - ỹ, MOI.EqualTo(0.))
    # MOI.add_constraint(o, 1. * z * z - z̃, MOI.EqualTo(0.))
    x̲ = -2.
    x̄ = 3.
    y̲ = -1.
    ȳ = 4.
    MOI.add_constraint(o, x, MOI.GreaterThan(       x̲ ))
    MOI.add_constraint(o, x, MOI.LessThan(          x̄ ))
    MOI.add_constraint(o, y, MOI.GreaterThan(       y̲ ))
    MOI.add_constraint(o, y, MOI.LessThan(          ȳ ))
    # MOI.add_constraint(o, 1. * x * y - bl, MOI.EqualTo(0.)) # (Naive) Bilinear constr: bl = xy
    # MOI.add_constraint(o, .5 * (1. * z̃ - x̃ - ỹ) - bl, MOI.EqualTo(0.)) # (Reformulated) Bilinear constr: bl = xy
    # McCormick envelope containing 4 liear constrs
    MOI.add_constraint(o, x̲ * y + y̲ * x - bl, MOI.LessThan(   x̲ * y̲))
    MOI.add_constraint(o, x̄ * y + ȳ * x - bl, MOI.LessThan(   x̄ * ȳ))
    MOI.add_constraint(o, x̄ * y + y̲ * x - bl, MOI.GreaterThan(x̄ * y̲))
    MOI.add_constraint(o, x̲ * y + ȳ * x - bl, MOI.GreaterThan(x̲ * ȳ))
    f = r' * [x,y,bl] # test a linear function
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(o) # linear program
    MOI.get(o, MOI.TerminationStatus())
    rs1 = [MOI.get(o, MOI.VariablePrimal(), x)
    MOI.get(o, MOI.VariablePrimal(), y)
    MOI.get(o, MOI.VariablePrimal(), bl)]



    o = Gurobi.Optimizer()
    x = MOI.add_variable(o)
    y = MOI.add_variable(o)
    bl = MOI.add_variable(o)
    x̲ = -2.
    x̄ = 3.
    y̲ = -1.
    ȳ = 4.
    MOI.add_constraint(o, x, MOI.GreaterThan(       x̲ ))
    MOI.add_constraint(o, x, MOI.LessThan(          x̄ ))
    MOI.add_constraint(o, y, MOI.GreaterThan(       y̲ ))
    MOI.add_constraint(o, y, MOI.LessThan(          ȳ ))
    MOI.add_constraint(o, 1. * x * y - bl, MOI.EqualTo(0.)) # (Naive) Bilinear constr: bl = xy
    f = r' * [x,y,bl]
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(o, MOI.RawOptimizerAttribute("NonConvex"), 2)
    MOI.optimize!(o) # nonconvex quad program
    MOI.get(o, MOI.TerminationStatus())
    rs2 = [MOI.get(o, MOI.VariablePrimal(), x)
    MOI.get(o, MOI.VariablePrimal(), y)
    MOI.get(o, MOI.VariablePrimal(), bl)]

    for (l,r) in zip(rs1,rs2)
        if abs(l-r) > 1e-4
            error("solution is not the same for 2 programs!")
        end
    end
end

while true # test
    r = rand(3) .- .5
    test_tightness_Mc_bilinear(r)
end

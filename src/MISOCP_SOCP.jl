import MathOptInterface as MOI
import Gurobi

# MISOCP-SOCP bilevel problem 
# 15/11/23

if true
    function silent_new_optimizer()
        o = Gurobi.Optimizer(GRB_ENV); # master
        MOI.set(o,MOI.RawOptimizerAttribute("OutputFlag"),0)
        return o
    end
    function Q1d(x)
        o = silent_new_optimizer()
        # (1d)
        y = MOI.add_variables(o,3)
        MOI.add_constraints(o, B * y, MOI.GreaterThan.(b .- A * x))
        MOI.add_constraint(o, y, MOI.SecondOrderCone(3))
        f = d' * y
        MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.optimize!(o)
        # println("MOI.optimize!(o) returns with MaxVio = $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
        @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
        yt = MOI.get.(o,MOI.VariablePrimal(),y)
        return (o = d' * yt, y = yt)
    end
    function Q1d(x,::Char)
        return Q1d(x).o
    end
    function Q1d(x,::String)
        return Q1d(x).y
    end
    function Q45() # solve as a whole: (5) in (4) elim t
        o = silent_new_optimizer()
        x = MOI.add_variables(o,3)
        y = MOI.add_variables(o,3)
        ψ = MOI.add_variables(o,3)
        MOI.add_constraint.(o, x, MOI.ZeroOne())                            # x in 𝒳
        MOI.add_constraint(o, y, MOI.SecondOrderCone(3))                    # y in 𝒦
        MOI.add_constraint.(o, ψ, MOI.GreaterThan(0.))                      # ψ in ℝ₊
        MOI.add_constraint(o, G_x' * x, MOI.GreaterThan(h))                 # (5b)
        MOI.add_constraints(o, A * x .+ B * y, MOI.GreaterThan.(b))         # (5c)
        aux = MOI.add_variables(o,3)
        MOI.add_constraint.(o, B' * ψ .+ aux, MOI.EqualTo.(d))
        MOI.add_constraint(o, aux, MOI.SecondOrderCone(3))                  # (5d)
        MOI.add_constraint(o, d' * y - ψ' * (b .- A * x), MOI.LessThan(0.)) # (5e)
        f = c_x' * x + c_y' * y
        MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.optimize!(o)
        # println("MOI.optimize!(o) returns with MaxVio = $(MOI.get(o, Gurobi.ModelAttribute("MaxVio")))")
        @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
        xt = MOI.get.(o,MOI.VariablePrimal(),x)
        @info "" xt
        yt = MOI.get.(o,MOI.VariablePrimal(),y)
        @info "" yt
        ψt = MOI.get.(o,MOI.VariablePrimal(),ψ)
        @info "" ψt
    end    
end

if true # data
    const c_x, c_y = [-8., -4, -2], [-3., 0, 0]
    const G_x, h = [-4., -2, -1], -4.
    const d = [-1., 0, 0]
    const A = [4. 2 1;
        -4. -2 -1;
        -4. -2 -1]
    const B = [4. 0 0;
        -2. 0 0;
        -1. 0 0]
    const b = [6.,-10,-6]
end

# Min  c_x' * x + c_y' * y
# s.t.    G_x' * x ≥ h                                         (1b)
#     x ∈ 𝔹³, all linking variables                            (1c)
#         y ∈ argmin            d' * y                         (1d)
#             y ∈ SOC(3)  s.t. A * x + B * y ≥ b

#     Min             cₓ' * x + t
# x ∈ 𝔹³, t ∈ ℝ      t  ≥  f(x)

const GRB_ENV = Gurobi.Env()

# we might try possible x's directly and draw a picture
# coeff_Bin = [4., 2, 1]
# x = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]]
# y = map(e -> Q1d(e,"y"), x)
# MK.lines(map(e -> coeff_Bin' * e, x), [c_x' * i + c_y' * j for (i,j) in zip(x,y)])


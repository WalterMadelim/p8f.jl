function f(uu, vv, xx, YY::Matrix, ZZ::Matrix) # YY, ZZ should be in [0, 1]
    υ = JumpModel(0)
    # curtail and shedding
    JuMP.@variable(υ, ϖ[1:T, 1:W] >= 0.)
    JuMP.@variable(υ, ζ[1:T, 1:L] >= 0.)
    JuMP.@expression(υ, CW[t = 1:T, w = 1:W], Wℶ["CW"][w] * ϖ[t, w])
    JuMP.@expression(υ, CL[t = 1:T, l = 1:L], Lℶ["CL"][l] * ζ[t, l])
    JuMP.@constraint(υ, ℵϖ[t = 1:T, w in 1:W], ϖ[t, w] - Wℶ["MAX"][w] * Y[t, w] <= 0.)
    JuMP.@constraint(υ, ℵζ[t = 1:T, l in 1:L], ζ[t, l] - Lℶ["MAX"][l] * Z[t, l] <= 0.)
    # generations
    ## reserve
    JuMP.@variable(υ, ρ[1:T, 1:G] >= 0.)
    JuMP.@expression(υ, CGres[t = 1:T, g = 1:G], Gℶ["CR"][g] * ρ[t, g])
    ## slack generator, has liability for the power balance
    JuMP.@variable(υ, ϱ[1:T] >= 0.)
    JuMP.@variable(υ, ϱsq[1:T])
    JuMP.@constraint(υ, [t = 1:T], [ϱsq[t] + 1, ϱsq[t] - 1, 2 * ϱ[t]] in JuMP.SecondOrderCone()) # 🍧
    JuMP.@expression(υ, CGgen1[t = 1:T], Gℶ["C2"][1] * ϱsq[t] + Gℶ["C1"][1] * ϱ[t])
    ## normal generators
    JuMP.@variable(υ, p[0:T, 2:G]) # power output of the Generator 2:G
    [ JuMP.fix(p[0, g], Gℶ["ZP"][g]; force = true) for g in 2:G ]
    JuMP.@variable(υ, psq[1:T, 2:G])
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], [psq[t, g] + 1, psq[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍧
    JuMP.@expression(υ, CGgen2[t = 1:T, g = 2:G], Gℶ["C2"][g] * psq[t, g] + Gℶ["C1"][g] * p[t, g] + Gℶ["C0"][g])
    JuMP.@variable(υ, ϕ[1:T, 2:G] >= 0.) # epi-variable of Cost_Generators, only for 2:G
    JuMP.@constraint(υ, ℵϕ[t = 1:T, g = 2:G], CGgen2[t, g] - (1. - x[t, g]) * θ(g, Gℶ["PS"][g]) - ϕ[t, g] <= 0.)
    # ★ Linking ★ 
    JuMP.@constraint(υ, ℵdpl[t = 1:T, g = 2:G], -Gℶ["RD"][g] * x[t, g] - Gℶ["SD"][g] * v[t, g] + p[t-1, g] - p[t, g] <= 0.)
    JuMP.@constraint(υ, ℵdpr[t = 1:T, g = 2:G], p[t, g] - p[t-1, g] - Gℶ["RU"][g] * x[t-1, g] - Gℶ["SU"][g] * u[t, g] <= 0.)
    # physical constrs
    JuMP.@constraint(υ, ℵSRD[t = 1:T], SRD - sum(ρ[t, :]) <= 0.)
    JuMP.@constraint(υ, ℵϱ[t = 1:T], ϱ[t] + ρ[t, 1] - Gℶ["PS"][1] <= 0.)
    JuMP.@constraint(υ, ℵPI[t = 1:T, g = 2:G], Gℶ["PI"][g] * x[t, g] - p[t, g] <= 0.)
    JuMP.@constraint(υ, ℵPS[t = 1:T, g = 2:G], p[t, g] + ρ[t, g] - Gℶ["PS"][g] * x[t, g] <= 0.)
    JuMP.@expression(υ, line_flow[t = 1:T, b = 1:B], sum(F[b, Wℶ["n"][w]] * (Wℶ["MAX"][w] * Y[t, w] - ϖ[t, w]) for w in 1:W) + 
                                                sum(F[b, Gℶ["n"][g]] * p[t, g] for g in 2:G)
                                                - sum(F[b, Lℶ["n"][l]] * (Lℶ["MAX"][l] * Z[t, l] - ζ[t, l]) for l in 1:L))
    JuMP.@constraint(υ, ℵlfl[t = 1:T, b = 1:B], -Bℶ["BC"][b] <= line_flow[t, b])
    JuMP.@constraint(υ, ℵlfr[t = 1:T, b = 1:B], line_flow[t, b] <= Bℶ["BC"][b])
    JuMP.@constraint(υ, ℵbalance[t = 1:T], sum(Wℶ["MAX"][w] * Y[t, w] - ϖ[t, w] for w in 1:W) + sum(p[t, :]) + ϱ[t] == sum(Lℶ["MAX"][l] * Z[t, l] - ζ[t, l] for l in 1:L))
    JuMP.@objective(υ, Min, sum(CW) + sum(CL) + sum(CGres) + sum(CGgen1) + sum(ϕ))
    JuMP.optimize!(υ)
end


function f(u, v, x, Y::Matrix, Z::Matrix)
    υ = JumpModel(0) # dual formulation
    JuMP.@variable(υ, ℵϖ[1:T, 1:W] >= 0.)
    JuMP.@variable(υ, ℵζ[1:T, 1:L] >= 0.)
    JuMP.@variable(υ, ℵϕ[1:T, 2:G] >= 0.)
    JuMP.@variable(υ, ℵdpl[1:T, 2:G] >= 0.)
    JuMP.@variable(υ, ℵdpr[1:T, 2:G] >= 0.)
    JuMP.@variable(υ, ℵSRD[1:T] >= 0.)
    JuMP.@variable(υ, ℵϱ[1:T] >= 0.)
    JuMP.@variable(υ, ℵPI[1:T, 2:G] >= 0.)
    JuMP.@variable(υ, ℵPS[1:T, 2:G] >= 0.)
    JuMP.@variable(υ, ℵbalance[1:T])
    JuMP.@variable(υ, ℵlfl[1:T, 1:B] >= 0.)
    JuMP.@variable(υ, ℵlfr[1:T, 1:B] >= 0.)
    JuMP.@variable(υ, ℵQ11[1:T])
    JuMP.@variable(υ, ℵQ12[1:T])
    JuMP.@variable(υ, ℵQ13[1:T])
    JuMP.@variable(υ, ℵQ21[1:T, 2:G])
    JuMP.@variable(υ, ℵQ22[1:T, 2:G])
    JuMP.@variable(υ, ℵQ23[1:T, 2:G])
    JuMP.@objective(υ, Max, -sum(Bℶ["BC"][b] * (ℵlfl[t, b] + ℵlfr[t, b]) for b in 1:B, t in 1:T) 
        +sum(
            (ℵlfl[t, b] - ℵlfr[t, b]) * (sum(F[b, Lℶ["n"][l]] * Lℶ["MAX"][l] * Z[t, l] for l in 1:L) - sum(F[b, Wℶ["n"][w]] * Wℶ["MAX"][w] * Y[t, w] for w in 1:W))
                for b in 1:B, t in 1:T
            )
        -sum(ℵϖ[t, w] * (Wℶ["MAX"][w] * Y[t, w]) for w in 1:W, t in 1:T)
        -sum(ℵζ[t, l] * (Lℶ["MAX"][l] * Z[t, l]) for l in 1:L, t in 1:T)
        +sum(ℵQ12[t] - ℵQ11[t] for t in 1:T)
        +sum(ℵQ22[t, g] - ℵQ21[t, g] for g in 2:G, t in 1:T)
        +sum(ℵϕ[t, g] * (Gℶ["C0"][g] - (1 - x[t, g]) * θ(g, Gℶ["PS"][g])) for g in 2:G, t in 1:T)
        +SRD * sum(ℵSRD)
        -Gℶ["PS"][1] * sum(ℵϱ)
        +sum(Gℶ["ZP"][g] * (ℵdpl[1, g] - ℵdpr[1, g]) for g in 2:G)
        +sum(ℵdpl[t, g] * (-Gℶ["RD"][g] * x[t, g] - Gℶ["SD"][g] * v[t, g]) for g in 2:G, t in 1:T)
        +sum(ℵdpr[t, g] * (-Gℶ["RU"][g] * x[t-1, g] - Gℶ["SU"][g] * u[t, g]) for g in 2:G, t in 1:T)
        +sum(Gℶ["PI"][g] * x[t, g] * ℵPI[t, g] for g in 2:G, t in 1:T)
        +sum(ℵbalance[t] * (sum(Wℶ["MAX"][w] * Y[t, w] for w in 1:W) - sum(Lℶ["MAX"][l] * Z[t, l] for l in 1:L)) for t in 1:T)
        -sum(Gℶ["PS"][g] * x[t, g] * ℵPS[t, g] for g in 2:G, t in 1:T)
    )
    JuMP.@expression(υ, expr1[t = 1:T, g = 2:G], sum(F[b, Gℶ["n"][g]] * (ℵlfr[t, b] - ℵlfl[t, b]) for b in 1:B) 
        + Gℶ["C1"][g] * ℵϕ[t, g] - 2 * ℵQ23[t, g] + (ℵdpr[t, g] - ℵdpl[t, g]) + (ℵPS[t, g] - ℵPI[t, g]) + ℵbalance[t])
    JuMP.@constraint(υ, p_T[g = 2:G], expr1[T, g] == 0.)
    JuMP.@constraint(υ, p[t = 1:T-1, g = 2:G], expr1[t, g] + (ℵdpl[t+1, g] - ℵdpr[t+1, g]) == 0.)
    JuMP.@constraint(υ, psq[t = 1:T, g = 2:G], -ℵQ21[t, g] - ℵQ22[t, g] + Gℶ["C2"][g] * ℵϕ[t, g] == 0.)
    JuMP.@constraint(υ, ϕ[t = 1:T, g = 2:G], 1. - ℵϕ[t, g] >= 0.)
    JuMP.@constraint(υ, ρ_1[t = 1:T], Gℶ["CR"][1] - ℵSRD[t] + ℵϱ[t] >= 0.)
    JuMP.@constraint(υ, ρ[t = 1:T, g = 2:G], Gℶ["CR"][g] - ℵSRD[t] + ℵPS[t, g] >= 0.)
    JuMP.@constraint(υ, ϱ[t = 1:T], Gℶ["C1"][1] - 2 * ℵQ13[t] + ℵϱ[t] + ℵbalance[t] >= 0.)
    JuMP.@constraint(υ, ϱsq[t = 1:T], Gℶ["C2"][1] - ℵQ11[t] - ℵQ12[t] == 0.)
    JuMP.@constraint(υ, ϖ[t = 1:T, w = 1:W], ℵϖ[t, w] + Wℶ["CW"][w] - ℵbalance[t] + sum(F[b, Wℶ["n"][w]] * (ℵlfl[t, b] - ℵlfr[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(υ, ζ[t = 1:T, l = 1:L], ℵζ[t, l] + Lℶ["CL"][l] + ℵbalance[t] + sum(F[b, Lℶ["n"][l]] * (ℵlfr[t, b] - ℵlfl[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(υ, [t = 1:T], [ℵQ11[t], ℵQ12[t], ℵQ13[t]] in JuMP.SecondOrderCone()) # 🍧
    JuMP.@constraint(υ, [t = 1:T, g = 2:G], [ℵQ21[t, g], ℵQ22[t, g], ℵQ23[t, g]] in JuMP.SecondOrderCone()) # 🍧
    JuMP.optimize!(υ)
end




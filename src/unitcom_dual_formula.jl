
macro prim_skeleton()
    return esc(
        quote 
            JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
            JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
            JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
            JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
            JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
            JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
            JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
            JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
            JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2)) # okay
        end
    )
end

macro dual_skeleton()
    return esc(
        quote
            JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
            JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
            JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
            JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
            JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
            JuMP.@expression(ø, pslCom[t = 1:T], CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
            JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
            JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
            JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
            JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t])  >= 0.)
            JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t])           >= 0.)
            JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] >= 0.)
            JuMP.@expression(ø, dualobj,
                -ip(Y, Dvp) - ip(Z, Dzt)
                + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            )
        end
    )
end

macro prim_Quad()
    return esc(
        quote
            JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
            JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
            JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
            JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
            JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
            JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
            JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
            JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
            JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
            JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
            JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe)) # okay
        end
    )
end

macro dual_Quad()
    return esc(
        quote
            JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
            JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
            JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
            JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
            JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
            JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
            JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
            JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
            JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
            JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
            JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
            JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
            JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t])    >= 0.)
            JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t])             >= 0.)
            JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t]   >= 0.)
            JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
                -ip(Y, Dvp) - ip(Z, Dzt) 
                + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            )
        end
    )
end

macro prim_QuadBf()
    return esc(quote
        JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
        JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
        JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
        JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
        JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
        JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
        JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
        ) # 🌸
        JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
        JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
        JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
        JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
        JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe)) # okay
    end)
end

macro dual_QuadBf()
    return esc(quote
        JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
        JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
        JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
        JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
        JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
        JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
        JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
        JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
        JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
        JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
        JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
            -ip(Y, Dvp) - ip(Z, Dzt) 
            + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
        )
    end)
end

macro prim_code()
    return esc(quote
        JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
        JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
        JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
        JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
        JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
        JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
        JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
        JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
        JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
        JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
        JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
        ) # 🌸
        JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
        JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
        JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
        JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
        JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe))
    end)
end

macro dual_code()
    return esc(quote
        JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
        JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
        JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
        JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
        JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
        JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
        JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
        JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
        JuMP.@constraint(ø, p1[g = 1:G], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0.) # 🍀
        JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0.) # 🍀
        JuMP.@constraint(ø, pT[g = 1:G], pCom[T, g] + Du[T, g] - Dd[T, g] == 0.) # 🍀
        JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
        JuMP.@constraint(ø, psl1, pslCom[1] + Du1[G+1] - Dd1[G+1] + Dd[1+1, G+1] - Du[1+1, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, psl2[t = 2:T-1], pslCom[t] + Du[t, G+1] - Dd[t, G+1] + Dd[t+1, G+1] - Du[t+1, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, pslT, pslCom[T] + Du[T, G+1] - Dd[T, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
            -ip(Y, Dvp) - ip(Z, Dzt) + ip(Dd1 .- Du1, ZP)
            + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
            - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
            - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
            - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
            - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1)
        )
    end)
end


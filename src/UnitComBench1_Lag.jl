# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import LinearAlgebra
import Distributions
import Statistics
import Random
import Gurobi
import MosekTools
import JuMP
using Logging
GRB_ENV = Gurobi.Env()

# In this program, we work on the upper bound value v_MSDRO
# We consider settings
# in dimension 1: T = 4, 6 or 8
# in dimension 2: UC_simple (UC), UC_Ramp (UCR), UC_RampBC (UCRB), UC_RampBCQuad (UCRBQ)
# results include lb and ub values as well as time
# if gap cannot be shrunk to 1/100 within 30 minutes, we just provide the incumbent primal value and dual bound.
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#  ([lb, ub], t) |             UC                          UCR                     UCRB                        UCRBQ
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# T = 4          | ([2.38589, 2.38589], 18s)  ([4.93065, 4.93065], 30s)   ([5.03194, 5.03194], 41s)    ([14.7116, 14.8431], 263s)
# T = 6          | ([2.43250, 2.43250], 26s)  ([4.64843, 4.67767], 147s)  ([4.91033, 4.91033], 1388s)  ([14.8707, 35.3956], 30min)*
# T = 8          | ([8.67519, 8.76274], 85s)  ([23.9188, 24.1423], 229s)  ([13.3761, 474.702], 30min)  ([28.0611, +∞], 30min)**
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# *: this result is derived by apply Lag cut in the lower level, if we use Ben cut as before, we would not even have valid lbs'. It won't help if we apply Lag cut before validity test and reapply Ben cut afterward.
# **: by apply lag cut instead of Ben cut in the lower level

# All 4 formulations has Gurobi's bilinear DualBnd
# how to shorten time:
# 1. if Benders' cut is good, don't opt SB cut or Lag cut
# 2. don't re-choose iY is bwd
# 3. btBnd is a very important hyperparameter, at least 7 times of the optimal value, but not too large

btBnd = 65.0 # improper this value could incur infeasibility of f()
cϵ = 1e-5
function JumpModel(i)
    if i == 0
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # JuMP.set_attribute(ø, "QCPDual", 1)
    elseif i == 1 
        ø = JuMP.Model(MosekTools.Optimizer) # vio = JuMP.get_attribute(ø, Gurobi.ModelAttribute("MaxVio")) 🍀 we suggest trial-and-error to decide if a constr is tight, rather than using this cmd
    elseif i == 2 
        ø = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(ø) # JuMP.unset_silent(ø)
    return ø
end
macro add_β2() return esc(:(JuMP.@variable(ø, β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]))) end
macro add_β1() return esc(:(JuMP.@variable(ø, β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))]))) end
macro addMatVarViaCopy(x, xΓ) return esc(:( JuMP.@variable(ø, $x[eachindex(eachrow($xΓ)), eachindex(eachcol($xΓ))]) )) end
macro addMatCopyConstr(cpx, x, xΓ) return esc(:( JuMP.@constraint(ø, $cpx[i = eachindex(eachrow($x)), j = eachindex(eachcol($x))], $x[i, j] == $xΓ[i, j]) )) end
macro optimise() return esc(:( (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø)) )) end
macro reoptimise()
    return esc(quote
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
    end)
end
function decode_uv_from_x(x::BitMatrix)
    xm1 = vcat(transpose(ZS), x)[1:end-1, :]
    dif = Int.(x .- xm1)
    u = dif .== 1
    v = dif .== -1
    return u, v
end
is_finite(r) = -Inf < r < Inf
function rgap(l, u)
    if is_finite(l) && is_finite(u)
        return abs(u - l) / max(abs(l), abs(u))
    end
    return Inf
end
function rgapsimple(l, u)
    return abs(u - l) / max(abs(l), abs(u))
end
ip(x, y)       = LinearAlgebra.dot(x, y)
norm1(x)       = LinearAlgebra.norm(x, 1)
rd4(f)         = round(f; digits = 4)
rd6(f)         = round(f; digits = 6)
jv(x)          = JuMP.value.(x)
jd(x)          = JuMP.dual.(x)
jo(ø)           = JuMP.objective_value(ø)
get_bin_var(x) = Bool.(round.(jv(x))) # applicable only when status == JuMP.OPTIMAL
brcs(v) = ones(T) * transpose(v) # to broadcast those timeless cost coeffs 
get_safe_Z(Z) = (jv(Z) .> .4) .* brcs(LM)

macro stage1feas_code()
    return esc(quote
    JuMP.@variable(ø, u[t = 1:T, g = 1:G+1], Bin)
    JuMP.@variable(ø, v[t = 1:T, g = 1:G+1], Bin)
    JuMP.@variable(ø, x[t = 1:T, g = 1:G+1], Bin)
    JuMP.@constraint(ø, x .- vcat(transpose(ZS), x)[1:end-1, :] .== u .- v)
    JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-UT+1], sum(x[i, g] for i in t:t+UT-1) >= UT * u[t, g])
    JuMP.@constraint(ø, [g = 1:G+1, t = T-UT+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0)
    JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-DT+1], sum(1 - x[i, g] for i in t:t+DT-1) >= DT * v[t, g])
    JuMP.@constraint(ø, [g = 1:G+1, t = T-DT+1:T], sum(1 - x[i, g] - v[t, g] for i in t:T) >= 0)
    end)
end
macro primobj_code() # UC
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
    JuMP.@constraint(ø, Dbal[t = 1:T], sum(ζ[t, :]) == sum(ϱ[t, :]) + sum(ϖ[t, :]))
    JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
    JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
    JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G+1], p[t, g] >= ϱ[t, g])
    JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
    JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
    JuMP.@expression(ø, gccost, ip(brcs(CG), p .- ϱ))
    JuMP.@expression(ø, lscost_2, -ip(CL, ζ)) 
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, primobj, lscost_2 + gccost + PRIMAL_CN)
    end)
end
macro dualobj_code()
    return esc(quote
    JuMP.@variable(ø, Dbal[t = 1:T]) # adverse: free variable
    JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0)
    JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0)
    JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@constraint(ø, p[t = 1:T, g = 1:G+1], Dps[t, g] - Dpi[t, g] - Dvr[t, g] + CG[g] == 0)
    JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G+1], Dbal[t] - CG[g] + Dvr[t, g] >= 0)
    JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + Dbal[t] >= 0)
    JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] - CL[t, l] - Dbal[t] >= 0)
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z))
    JuMP.@expression(ø, dualobj, ip(x, brcs(PI) .* Dpi) - ip(x, brcs(PS) .* Dps) - ip(Dvp, Y) - ip(Dzt, Z) + PRIMAL_CN)
    end)
end
macro primobj_code() # UCR
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
    JuMP.@constraint(ø, Dbal[t = 1:T], sum(ζ[t, :]) == sum(ϱ[t, :]) + sum(ϖ[t, :]))
    JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
    JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
    JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G+1], p[t, g] >= ϱ[t, g])
    JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
    JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
    JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
    JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
    JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
    JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
    JuMP.@expression(ø, gccost, ip(brcs(CG), p .- ϱ))
    JuMP.@expression(ø, lscost_2, -ip(CL, ζ)) 
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, primobj, lscost_2 + gccost + PRIMAL_CN)
    end)
end
macro dualobj_code()
    return esc(quote
    JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
    JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊 
    JuMP.@variable(ø, Dbal[t = 1:T]) # adverse: free variable
    JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0)
    JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0)
    JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@expression(ø, pCom[t = 1:T, g = 1:G+1], Dps[t, g] - Dpi[t, g] - Dvr[t, g] + CG[g])
    JuMP.@constraint(ø, p1[g = 1:G+1], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0)
    JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G+1], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0)
    JuMP.@constraint(ø, pT[g = 1:G+1], pCom[T, g] + Du[T, g] - Dd[T, g] == 0)
    JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G+1], Dbal[t] - CG[g] + Dvr[t, g] >= 0)
    JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + Dbal[t] >= 0)
    JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] - CL[t, l] - Dbal[t] >= 0)
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, dualobj, ip(x, brcs(PI) .* Dpi) - ip(x, brcs(PS) .* Dps) - ip(Dvp, Y) - ip(Dzt, Z) + PRIMAL_CN
    + ip(Dd1 .- Du1, ZP)
    - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
    - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
    - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
    - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1))
    end)
end
macro primobj_code() # UCRB
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
    JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
    ) # 🌸
    JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
    JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
    JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
    JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
    JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
    JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
    JuMP.@constraint(ø, Dbal[t = 1:T], sum(ζ[t, :]) == sum(ϱ[t, :]) + sum(ϖ[t, :]))
    JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
    JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
    JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G+1], p[t, g] >= ϱ[t, g])
    JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
    JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
    JuMP.@expression(ø, gccost, ip(brcs(CG), p .- ϱ))
    JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, primobj, lscost_2 + gccost + PRIMAL_CN)
    end)
end
macro dualobj_code()
    return esc(quote
    JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
    JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
    JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
    JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊 
    JuMP.@variable(ø, Dbal[t = 1:T]) # Gurobi's bilinear programming has dual bound, despite it's a free variable
    JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0)
    JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0)
    JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@expression(ø, pCom[t = 1:T, g = 1:G+1], Dps[t, g] - Dpi[t, g] - Dvr[t, g] + CG[g])
    JuMP.@constraint(ø, p1[g = 1:G+1], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0)
    JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G+1], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0)
    JuMP.@constraint(ø, pT[g = 1:G+1], pCom[T, g] + Du[T, g] - Dd[T, g] == 0)
    JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G+1], Dbal[t] - CG[g] + Dvr[t, g]  + (g == G+1 ? 0. : sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B)) >= 0)
    JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + Dbal[t]            + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0)
    JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] - CL[t, l] - Dbal[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0)
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, dualobj, ip(x, brcs(PI) .* Dpi) - ip(x, brcs(PS) .* Dps) - ip(Dvp, Y) - ip(Dzt, Z) + PRIMAL_CN
    + ip(Dd1 .- Du1, ZP)
    - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
    - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
    - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
    - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1)
    - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
    )
    end)
end
macro primobj_code() # UCRBQ
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
    JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
    JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
    JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
    JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
    JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
    ) # 🌸
    JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
    JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
    JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
    JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
    JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
    JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
    JuMP.@constraint(ø, Dbal[t = 1:T], sum(ζ[t, :]) == sum(ϱ[t, :]) + sum(ϖ[t, :]))
    JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
    JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
    JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G+1], p[t, g] >= ϱ[t, g])
    JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
    JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
    JuMP.@expression(ø, gccost, ip(brcs(CG), p .- ϱ))
    JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, primobj, lscost_2 + gccost + PRIMAL_CN + sum(pe))
    end)
end
macro dualobj_code()
    return esc(quote
    JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
    JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
    JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
    JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
    JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
    JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
    JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
    JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
    JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊 
    JuMP.@variable(ø, Dbal[t = 1:T]) # Gurobi's bilinear programming has dual bound, despite it's a free variable
    JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0)
    JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0)
    JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0)
    JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
    JuMP.@expression(ø, pCom[t = 1:T, g = 1:G+1], De[t, g] * C1[g] - 2 * D3[t, g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g] + CG[g])
    JuMP.@constraint(ø, p1[g = 1:G+1], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0)
    JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G+1], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0)
    JuMP.@constraint(ø, pT[g = 1:G+1], pCom[T, g] + Du[T, g] - Dd[T, g] == 0)
    JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G+1], Dbal[t] - CG[g] + Dvr[t, g]  + (g == G+1 ? 0. : sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B)) >= 0)
    JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + Dbal[t]            + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0)
    JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] - CL[t, l] - Dbal[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0)
    JuMP.@expression(ø, PRIMAL_CN, ip(CL, Z)) # 🥑 ofc
    JuMP.@expression(ø, dualobj, ip(x, brcs(PI) .* Dpi) - ip(x, brcs(PS) .* Dps) - ip(Dvp, Y) - ip(Dzt, Z) + PRIMAL_CN
    + sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
    + ip(Dd1 .- Du1, ZP)
    - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
    - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
    - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
    - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1)
    - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
    )
    end)
end
macro primobj_code() # ❌ bilinear programming with this dual formulation would have no (dual) BestBd column
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])       # generator power
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G+1] >= 0.) # generator cutback
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.) # wind curtail
    JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.) # load shedding
    JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
    JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
    JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
    JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
    JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w]) 
    JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
    JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G+1], p[t, g] >= ϱ[t, g])
    JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
    JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
    JuMP.@constraint(ø, Dbl[t = 1:T], sum(Z[t, :]) + sum(ϖ[t, :]) + sum(ϱ[t, :]) == sum(Y[t, :]) + sum(p[t, :]) + sum(ζ[t, :]))
    JuMP.@expression(ø, lscost, sum(CL[t, l] * ζ[t, l] for t in 1:T, l in 1:L))
    JuMP.@expression(ø, gccost, sum(   CG[g] * ϱ[t, g] for t in 1:T, g in 1:G+1))
    JuMP.@expression(ø, primobj, lscost + gccost)
    end)
end
macro dualobj_code() # ❌ bilinear programming with this dual formulation would have no (dual) BestBd column
    return esc(quote
    JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
    JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
    JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊 
    JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G+1] >= 0.)
    JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
    JuMP.@variable(ø, Dbl[t = 1:T])
    JuMP.@expression(ø, pCom[t = 1:T, g = 1:G+1], Dps[t, g] - Dvr[t, g] - Dpi[t, g] + Dbl[t])
    JuMP.@constraint(ø, p1[g = 1:G+1], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0)
    JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G+1], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0)
    JuMP.@constraint(ø, pT[g = 1:G+1], pCom[T, g] + Du[T, g] - Dd[T, g] == 0)
    JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G+1], Dvr[t, g] - Dbl[t] + CG[g] >= 0.)
    JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] - Dbl[t] >= 0.)
    JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] + Dbl[t] + CL[t, l] >= 0.)
    JuMP.@expression(ø, xobj, sum(x[t, g] * (PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) for t in 1:T, g in 1:G))
    JuMP.@expression(ø, Yobj, sum(Y[t, w] * ( Dbl[t] - Dvp[t, w]) for t in 1:T, w in 1:W))
    JuMP.@expression(ø, Zobj, sum(Z[t, l] * (-Dbl[t] - Dzt[t, l]) for t in 1:T, l in 1:L))
    JuMP.@expression(ø, dualobj, xobj + Yobj + Zobj
        + ip(Dd1 .- Du1, ZP)
        - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
        - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
        - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
        - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1)
    )
    end)
end



function primobj_value(u, v, x, Y, Z) # f
    ø = JumpModel(0)
    @primobj_code()
    JuMP.@objective(ø, Min, primobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(ø)
end
function primobj_value(x, yM, i, Z) # wrapper
    u, v = decode_uv_from_x(x)
    return value = primobj_value(u, v, x, yM[:, :, i], Z) 
end
function dualobj_value(u, v, x, Y, Z) # f
    ø = JumpModel(0)
    @dualobj_code()
    JuMP.@objective(ø, Max, dualobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(ø)
end
function dualobj_value(x, yM, i, Z) # wrapper
    u, v = decode_uv_from_x(x)
    return value = dualobj_value(u, v, x, yM[:, :, i], Z) 
end

ℶ1, ℶ2, Δ2, ℸ1, ℸ2 = let
    ℸ1 = Dict( # store solutions of lag_subproblem
        "oψ" => Float64[], # trial value
        "u" => Matrix{Float64}[], # trial vector
        "v" => Matrix{Float64}[], # trial vector
        "x" => Matrix{Float64}[], # trial vector
    )
    ℸ2 = Dict( # store solutions of lag_subproblem
        "ofv" => Float64[], # trial value
        "u" => Matrix{Float64}[], # trial vector
        "v" => Matrix{Float64}[], # trial vector
        "x" => Matrix{Float64}[], # trial vector
        "Y" => Matrix{Float64}[], # trial vector
    )
    ℶ1 = Dict(
        # "x" =>  BitMatrix[], # contain x only, where u, v can be decoded from
        # "rv" => Int[], # index of Y
        "st" => Bool[],
        "cn" => Float64[],
        "pu" => Matrix{Float64}[],
        "pv" => Matrix{Float64}[],
        "px" => Matrix{Float64}[],
        "pβ" => Matrix{Float64}[] # slope of β1
    )
    ℶ2 = Dict(
        # "rv" is negation of pβ, thus dropped
        "st" => Bool[],
        "cn" => Float64[],
        "pu" => Matrix{Float64}[],
        "pv" => Matrix{Float64}[],
        "px" => Matrix{Float64}[],
        "pY" => Matrix{Float64}[],
        "pβ" => Matrix{Float64}[] # slope of β2
    )
    Δ2 = Dict( # 🌸 used in argmaxY
        "f" => Float64[],
        "x" => BitMatrix[],
        "Y" => Int[],
        "β" => Matrix{Float64}[] # β2
    )
    ℶ1, ℶ2, Δ2, ℸ1, ℸ2
end
T = 8 # 🫖
G, W, L, B = 2, 2, 3, 11
(RU = [2.5, 1.9, 2.3]; SU = 1.3 * RU; RD = 1.1 * RU; SD = 1.3 * RD) # 🧊🧊🧊🧊
UT = DT = 3
CST = [0.63, 0.60, 0.72]
CSH = [0.15, 0.15, 0.15]
CL = [8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 16.0 13.776 14.443]
CL = CL[end-T+1:end, :]
CG = [4.0, 3.4, 3.6]
LM = [4, 3.5, 3]
ZS = [0, 0,   1]
ZP = [0, 0, 0.5]
PI = [0.45, 0.375, 0.5];
PS = [4.5, 4, 5.5];
C2 = [34, 23, 41]/1000
C1 = [0.67, 0.41, 0.93]/1.5;
C0 = CST / T;
EM = C2 .* PS .* PS .+ C1 .* PS .+ C0;
NG = [3, 2] # 🍀 since `G+1` generator is on the slack bus, it doesn't contribute to any power flow, we omit it
NW = [2, 3]
NL = [4, 5, 6]
FM = let
    [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
end
BC =  2 * [1.0043, 2.191, 1.3047, 0.6604, 1.7162, 0.6789, 1.0538, 1.1525, 1.3338, 0.4969, 0.7816]

T = 6 # 🫖
yM = [17.02789874074452 1.9227091485188643; 1.5383251276556877 1.6096720270813818; 2.185800784380979 1.8409105375468355; 1.772803772552515 1.4140304579002199; 1.9162477568289737 1.4051404662137312; 1.8719582150650769 1.8968856739763085;;; 1.5347287155945872 1.9423271642264108; 17.684082394903417 1.7375129196790082; 1.6151820566070874 2.1145595561674075; 2.5668449052782782 2.1181420207166477; 1.3705214984945078 2.329432068289938; 2.164691489250902 1.7635171507378784;;; 1.8271597456642905 1.5767116399740355; 1.2601374299515007 1.80496966345034; 19.206841243096562 1.6033915319481347; 1.8969443730812634 1.649166028079974; 2.3457214719857813 1.454077496475246; 1.5071329584323814 1.632181567608515;;; 1.4929388455631927 2.2299896095838343; 2.290576390350056 2.383620833670425; 1.975720484808629 1.7824131633587903; 17.25756910253582 1.9404158860231469; 2.025490174866754 1.7229575914805577; 1.2899102608137132 1.7382728330065866;;; 1.8595035706234948 2.0656150240983435; 1.3173737243501296 2.2702800988457645; 2.647618324496989 1.4541138841361199; 2.2486109156505965 1.536533552782479; 16.93009557583139 1.5890571041798522; 1.670322485901853 2.3578176381718836;;; 1.7502993746892992 2.362942848540181; 2.0466290609362257 1.3626603172448664; 1.7441151567732907 1.8092634802836953; 1.4481163474272578 1.8779963520151164; 1.6054078317315548 1.3991298341819185; 18.85200673814176 2.020190139616274;;; 1.5813104049686735 18.33675124583554; 1.6045248327373205 2.015910586091158; 1.593953935140532 1.6331270391759045; 2.1684557930229666 1.8386409280009928; 1.7809604667536325 1.4378229535954379; 2.1432029453657684 1.7881683985108006;;; 1.5802937697183108 2.327931072278277; 1.711731074377038 17.421629738862695; 2.1342324448039562 1.954558823264613; 2.6341075032966765 1.770107196542385; 2.297646027688173 1.667629910246313; 1.454940900257573 2.0022050879935307;;; 1.516826884850012 1.6504421300292713; 1.7940723155316847 1.65985342793086; 1.637948917967999 17.522918730738564; 1.7381944376512892 1.8454009443177752; 1.1867744176447756 2.648659361818211; 1.6068386679626498 1.751687720890158;;; 1.3314309063830085 2.097440120033972; 2.039138881260537 1.7168859023882441; 1.9252075152794499 2.0868850454973873; 2.137681261495258 16.119290934466626; 1.510678187470747 2.256751370623485; 1.9170556408736825 1.8407314853267591;;; 1.2480168949362325 1.6220981258681302; 2.1759049090735414 1.5398845963318855; 1.655594963914435 2.8156194432375368; 1.845698947192382 2.182227350863199; 1.488677719107833 19.810146440602995; 1.3636651032801976 2.0982185586668423;;; 1.7654955154973107 1.998176983581993; 1.6357234043199815 1.9001931868776043; 1.8594324478462045 1.9443812151079842; 1.8867476015169116 1.7919408783649735; 2.2831716658983656 2.123951971465343; 2.0104588215130543 16.645194813496424;;; 0.0 2.5517325534773576; 1.9375504284277267 1.9323323814484492; 2.153501033945049 2.141237526731776; 2.0188388315773573 2.453527233226468; 1.7051844603606505 2.2681683048883285; 2.303894877498385 1.7752329577459616;;; 1.3317607974129801 2.075078674232725; 0.0 1.8631008703787595; 2.6071488439707804 1.8648430025244125; 1.4959403409502718 1.452305097124336; 2.3260433546590344 1.8295746043897458; 1.4827274913556392 2.1589403161003338;;; 1.3619453713044962 3.211044217779356; 2.3864288782783145 1.787086801117053; 0.0 2.583050722393319; 2.057442042320189 2.054774716850402; 1.2441756611268568 2.1759789716802698; 1.855443790133532 1.950939928334532;;; 2.3353684305558704 1.6896731653103658; 1.1069756059557312 1.0631850231903301; 2.0318743289554204 2.442766458523295; 0.0 1.6838795796593744; 1.912525452403533 2.0524202504228257; 2.3310851353206288 1.5284091704672962;;; 1.4638204755484054 2.4365741777778567; 2.1535156090880863 1.1024115721753645; 1.50504842480962 2.648968324997294; 1.6998639098953174 1.9354285697676452; 0.0 2.2939854948700567; 2.2839292497821573 1.3478118011666607;;; 1.8205914133084518 1.83575408385743; 1.3945139927861314 2.1486362054125854; 2.2404674362493817 2.3307395704408718; 2.5446431753222933 1.7438885246989027; 1.5398141135923102 2.304561696184083; 0.0 1.967866542907905;;; 2.1705092327201814 0.0; 1.5459366733619562 1.288642359819994; 2.4948981325544635 2.3374199446480888; 1.9681999132148489 1.7099460924265937; 1.974495550574175 2.0632040020311284; 1.437119708335316 1.8788421735842584;;; 1.9631187906521494 1.9277043027012317; 1.652773341201679 0.0; 1.9607405186059523 1.8299340048801978; 1.412610301760252 2.2689373600823637; 1.268405943820094 1.8398407220473023; 2.444365083810922 1.930591899571325;;; 2.2549582003643747 2.668715570487638; 1.440518923788687 2.16085506652329; 2.7606192938013154 0.0; 2.289430386328699 1.5665135381808415; 2.5352038394986427 1.2063697758321483; 2.3390734458674336 1.6454100241605687;;; 2.1143939929875097 2.0691811108979015; 0.8498031972495005 1.5427032085983325; 2.3115151597912917 2.201782321876842; 2.171776023631531 0.0; 1.7889478405353616 1.392109210597823; 2.008998263877983 1.9669165193144542;;; 2.324646565331344 2.5192030204002145; 1.41120301467952 1.3701896747598683; 2.3073349111390127 1.4847998970627383; 2.1535821429047073 1.247952145205619; 2.2384829692888486 0.0; 2.6091034681579575 1.7620531997317739;;; 1.7265344363126363 2.367503184843714; 1.2151543079524283 1.7274512225139764; 2.37725422762583 2.178295147044981; 2.5370874018661427 1.9065115404988007; 1.3910128213817934 1.5498687098776773; 1.8098597899799922 0.0]
MY = [1.7198946575496272 2.060082453867921; 1.7744917251896695 1.791345142442502; 2.0628883818174555 2.0772642594336923; 2.0259352132038813 1.899109210057581; 1.728155372815513 1.9573704103907723; 1.851895447092089 1.7435506840606958]
MZ = [1.26723829950154 0.6727096064374787 2.2152435305495493; 3.0986377097936217 3.4229623321850227 1.6827441306529918; 2.8331247279184395 0.4899771169412037 2.331676836964725; 1.8849192189104738 3.2395724551885237 1.6774485560558825; 3.091414942577444 3.448985827101145 2.925740192946321; 3.997679285884835 3.4972010509636995 1.7894377023726833]

T = 4 # 🫖
yM = [12.932203243205766 1.3374359039022035; 1.9446659491235203 1.4162431451003246; 1.6511993236129792 2.1773399042968133; 1.782042683828961 2.557790493339905;;; 2.0011822386947595 1.6789225705000335; 13.690850014602509 1.997467388472595; 1.9198034781728475 1.9293910538786738; 2.107819269712466 1.8111662782757756;;; 1.5403048907063075 1.9116904989768762; 1.7523927556949361 1.8959624996410318; 11.758573486702598 2.129373069191475; 2.360835863553524 1.654410550094579;;; 1.5715933164025124 1.7031010579277321; 1.8408536127147777 1.9246653956279802; 2.2612809290337474 2.075380480088833; 11.753746532489489 2.246120859260827;;; 1.4384755553357391 11.943362474288024; 1.72344593236233 1.91548036162283; 2.1236245833170844 2.0635541107258346; 2.0145900767877167 1.9949694636426214;;; 1.421390950378163 1.8195885154671323; 1.9460989041791934 12.203355270334711; 2.0120047378255426 1.2887952780925083; 2.140262568332268 1.7497035506584138;;; 1.8926711067034918 1.6778456616989779; 1.5882059667141126 0.9989786752213483; 1.9555987045048253 13.308074727575503; 2.00116104992196 1.8335475496285436;;; 2.4284307784258465 1.764570097295026; 1.6252902737904777 1.615196030466516; 1.6359452680871918 1.9888566323078063; 2.327210511773217 13.145864096700304;;; 0.0 1.9303046697671453; 1.340036139647707 2.1272564678253922; 1.9643842464561987 1.8073858830728204; 2.124702105273566 1.2240590219268372;;; 1.350599019155928 1.6358596490089128; 0.0 1.4458183973273762; 1.7581241000312189 2.446722329979181; 1.7730088350804067 1.6995757281062374;;; 1.8272891775727969 1.3644451443696721; 1.614578717229169 1.7184087355368423; 0.0 1.9730551415386182; 1.2411914714500063 1.9819256857014036;;; 1.9689580539401672 1.537755504194629; 1.2568111339355166 1.4801389249965697; 1.4256251018077006 1.9976963085038109; 0.0 1.5725426375997271;;; 1.9538021541563502 0.0; 1.4684000134553443 1.3878659549049586; 1.5755518609653678 1.9138930410237733; 1.9252030431249159 1.7622562137125504;;; 1.9884105546097959 1.4568327122536733; 1.4511724621473678 0.0; 1.8340879617445378 2.7129660182996957; 1.9294116598340794 1.9669297233369172;;; 1.3725434547187787 1.689163218152612; 1.6119892285052182 2.582693976219158; 1.7352439054027282 0.0; 2.0012409991934312 1.9773612650438648;;; 1.5651952694036888 1.4769477561641766; 1.7825097976930242 1.8205879836574106; 1.9657939031493161 2.031470938642274; 1.578855990899333 0.0]
MY = [1.8085556385124137 1.7156454852430747; 1.8414127842661827 1.8103587289251535; 1.8688461336064663 2.09619165798559; 2.0042839049623544 1.874316195757285]
MZ = [1.26723829950154 0.6727096064374787 2.2152435305495493; 3.0986377097936217 3.4229623321850227 1.6827441306529918; 2.8331247279184395 0.4899771169412037 2.331676836964725; 1.8849192189104738 3.2395724551885237 1.6774485560558825]

T = 8 # 🫖
yM, MY = let
    [3.05859962924333 0.22129314909074682; 0.31753646450048895 0.2888728744013412; 0.27256656877799856 0.29547258378275026; 0.23910049918737447 0.29184920087643973; 0.2618263982426271 0.21039631467919426; 0.26299411377195747 0.31786708911852707; 0.28175032000925565 0.27988140319606497; 0.3623420214378466 0.22988918082592985;;; 0.3270142614850655 0.2437922786722481; 3.2216666535358964 0.25230084646797907; 0.21077310404968044 0.2228459244541164; 0.29720887486472636 0.2701304595146522; 0.35027456634255405 0.3074463999534698; 0.28675747094003773 0.2582064181512781; 0.2968194724538678 0.28495037759305053; 0.21139582361359377 0.15883249966262034;;; 0.30893982261630343 0.23635489823809003; 0.23766856090340874 0.29275689095104507; 3.2738947421131877 0.30470591369330413; 0.30130579694679144 0.3106466979614247; 0.11492286406562156 0.2110580861494358; 0.31480802584939943 0.2645081523107029; 0.30071445370031447 0.2208679426775983; 0.37525257752802843 0.24475842137879997;;; 0.2674523493864944 0.3119056042295227; 0.31608292807926985 0.22314510875734475; 0.29328439330760653 0.28889060619463297; 3.5764098154049586 0.34346332783825245; 0.2768493752224806 0.28427697285355014; 0.3752931902482649 0.22936086459450508; 0.3061686555722745 0.21354403990817256; 0.27170781142993383 0.053920992052867864;;; 0.2717574415308641 0.31269065480917796; 0.3507278126462144 0.21290953591222375; 0.08848065351555368 0.2640410081018837; 0.2584285683115975 0.2282111920693616; 3.6193300903600685 0.2694961113987438; 0.24444250159946096 0.1996148649554183; 0.26417635173347664 0.29121826303263615; 0.2460955090764697 0.2159730638603265;;; 0.27489650210676875 0.1460037040478942; 0.2891820622902724 0.27424927056837345; 0.29033716034590573 0.35744342175575483; 0.35884372838395623 0.3521243145218948; 0.24641384664603508 0.2875109035930402; 3.2936532197996535 0.27296928504877144; 0.37289850173526257 0.21230948138150044; 0.20244933510854768 0.2036038927478527;;; 0.2932060031681682 0.20805322909527263; 0.29879735862820367 0.26369044329179064; 0.27579688302092215 0.2551060521631845; 0.2892724885320672 0.2879276128215445; 0.2657009916041522 0.22744572825072373; 0.37245179655936383 0.3005439916935925; 3.00544670967344 0.24995680351117416; 0.24298777849540484 0.24301231613239627;;; 0.3930645195856088 0.14779185673282344; 0.2326405247767792 0.24898947586903486; 0.36960182183748574 0.27094715569976885; 0.274078459378576 0.25245375887789573; 0.2668869639359949 0.20636736421419996; 0.2212694449214986 0.2090490078831887; 0.26225459348425445 0.26191891309439796; 3.591212106778445 0.21276299008226401;;; 0.2617508049573694 3.1148472129588427; 0.274772137554294 0.230861039485004; 0.24043930026640756 0.24442340753418412; 0.32401140989702526 0.12893271294086822; 0.3432172673875635 0.33076897305755537; 0.1745589715797054 0.24359433389327995; 0.2370552018029826 0.23643289560639155; 0.15752701445168366 0.22655744179516168;;; 0.327501876602911 0.22903238581995147; 0.2814520516849723 3.239501573505252; 0.29501263931431 0.2963919443857751; 0.23342226075979455 0.3026989763582024; 0.24160749482555655 0.2529212801262277; 0.300975884435132 0.2692718067754279; 0.29086376233444794 0.20413093610958188; 0.2568959799228425 0.2221607401113527;;; 0.3182985424933479 0.22679171037815934; 0.23619408618013743 0.28058890089480293; 0.2911586185655969 2.97297833854534; 0.28336471470611063 0.34932768842856543; 0.2769359235242444 0.14217521335870598; 0.36836699213154117 0.21017326860603525; 0.2664763277148695 0.26496815968706017; 0.26305061626260434 0.20014492265130654;;; 0.3095240358767111 0.10614989207451718; 0.2783274975303469 0.28174480915690403; 0.2919482791233911 0.3441765647182391; 0.3327863126394038 3.0623795357447046; 0.23595498378139604 0.1547047856926807; 0.35789676118735486 0.2705385910014243; 0.2941467646629033 0.29883050722098853; 0.2394060957304048 0.20946338535789719;;; 0.2542653631156345 0.3341803656273732; 0.34183765140533334 0.25816132636109806; 0.21855388074757107 0.16321830308454843; 0.29979417109087036 0.18089899912884955; 0.303434116546947 3.219888185753861; 0.3194775636946691 0.2751678473324358; 0.2598590935282514 0.23093734481226902; 0.2195139145028779 0.3022244344090633;;; 0.36258471452591495 0.24785430343404524; 0.2934462465740891 0.27536042998124566; 0.2728525238797856 0.2320649353028252; 0.24572663980277282 0.2975813814085406; 0.23440144707456897 0.2760164243033833; 0.3057845221213479 3.331975696311318; 0.3338059339420677 0.22986053709744453; 0.22304413514281418 0.262985324648876;;; 0.30473053259902405 0.2208243691427282; 0.300321710011433 0.19035106331097112; 0.20934381824225248 0.2669913303794216; 0.21004131911201157 0.30600480162367627; 0.3061363491473582 0.2119174257787879; 0.22525622244964824 0.2099920410930159; 0.2633502497552206 2.9623619636450154; 0.2560455443495948 0.2427313815310452;;; 0.2812688515815907 0.2374794566842001; 0.20073437343370445 0.23491140866544363; 0.2597648382961558 0.22869863469636953; 0.07694881260940858 0.24316822111328662; 0.25742169132775017 0.30973505672828394; 0.24308117516870226 0.26964736999714906; 0.2829363037291445 0.2692619228837469; 0.23342016269016258 3.1506100778778623;;; 0.0 0.2205983125019094; 0.27199141480366285 0.1906090233476236; 0.13021288989441138 0.20331836742947548; 0.2842829742794221 0.2697917487245768; 0.3397981495029214 0.31193655773381984; 0.2823443504823441 0.176234826087362; 0.2630561015217323 0.22723031908379843; 0.19054805192654278 0.18642541548339175;;; 0.20848714805609664 0.2444063576415555; 0.0 0.2928550834068516; 0.2901051783198936 0.2921797780560889; 0.253472446169264 0.26244226782355234; 0.22642530045139583 0.28229010841320307; 0.23892219968352346 0.17162614725420436; 0.29693471674331573 0.24880800990415988; 0.3327042114543252 0.27476556835581545;;; 0.22992528603728385 0.26405398163803984; 0.291912617203835 0.25751402122238315; 0.0 0.2498254247374867; 0.30519924547217586 0.2421182586676962; 0.42292517610598757 0.3214464416653418; 0.23021411441930376 0.189281906616853; 0.24079393562318688 0.33826848811720733; 0.03195340986618991 0.26178542081349204;;; 0.4098418076830045 0.15624663803685584; 0.20683214015664458 0.3755903448958537; 0.2248970784246649 0.26828553676774564; 0.0 0.1990621344641028; 0.2937579254378023 0.2083458781918536; 0.20409714951575408 0.27640041241730207; 0.21069884499254826 0.3388413400329392; 0.24013411627972875 0.3986402187239855;;; 0.3054986508390967 0.15682136310013758; 0.19657955018218815 0.35014302969979666; 0.4259682760379778 0.26407769827213423; 0.2740162537029863 0.4099158520733623; 0.0 0.26421345555130177; 0.3704322441422444 0.3026931941236391; 0.3203652848542024 0.17013128213377726; 0.25538708941959143 0.27963527253042636;;; 0.2795751351292254 0.38166076182615627; 0.3047446683034931 0.22203356565301804; 0.22496425339992857 0.20345251345065435; 0.17592725628195782 0.19481296048040284; 0.4085627555787813 0.2770057102348909; 0.0 0.20270074830675838; 0.20190532227659966 0.2684310027482865; 0.4470927220314157 0.26373614363422115;;; 0.2996405133659934 0.25472396295341576; 0.23435216072799603 0.22048926994861257; 0.2905293416926053 0.24133591193957094; 0.30447520498024 0.2996592507703861; 0.2787615343971066 0.2621660522297694; 0.22034728601756665 0.19131812205139612; 0.0 0.3217723661475461; 0.23124465789484494 0.29423566421241615;;; 0.17642221094770533 0.36778697319770937; 0.3559273341854472 0.20630640935770006; 0.1901321790191478 0.2611522239150567; 0.3342842871414502 0.3231649263722015; 0.3664131773402556 0.26488346843033844; 0.32003332244441246 0.35434297152078814; 0.3635335962357172 0.25686856918196754; 0.0 0.2513901877315536;;; 0.3152253315870693 0.0; 0.29459020995830515 0.22260125025633778; 0.38539271444372075 0.312037223364905; 0.1877984806701446 0.44473364043047015; 0.28032352897336005 0.23381039872251003; 0.39208414615383047 0.19587197862101377; 0.29767475194908877 0.24358146991355167; 0.383647542334475 0.2408350731059668;;; 0.20529760222814916 0.2601884375745854; 0.2442883254546317 0.0; 0.17269424006133216 0.19815765990720893; 0.32017417164073486 0.257276524619968; 0.32879605257898126 0.23335052613373267; 0.2638385438314035 0.20889464047719322; 0.18810546351924876 0.3192780875837021; 0.26959769819660584 0.2716184254998685;;; 0.2676208503307123 0.3187295199544548; 0.3536893292281777 0.19133832652300806; 0.2579733583908312 0.0; 0.3091152375175884 0.20598822321913973; 0.2710194456106294 0.3873698143896652; 0.184423645596558 0.2642055898783238; 0.2424367727133554 0.2613726266226804; 0.3206002169378032 0.28621734168506746;;; 0.24937940799233618 0.3304345774538696; 0.30896143385607805 0.1790549437359309; 0.21089994230089792 0.27575606855376605; 0.21280860943198932 0.0; 0.3300316931721543 0.36247791245733985; 0.17950324571257817 0.15784142786450156; 0.25195116228308423 0.20362853941848427; 0.2548658659701156 0.27431087009502514;;; 0.2524545174840982 0.21981791337159676; 0.21449139711100362 0.25601819862015834; 0.26841318964643535 0.42711144485034064; 0.253984786175963 0.34255929993636913; 0.25668392849475646 0.0; 0.27390798618603124 0.28111162002209533; 0.3293624894761731 0.327008288413846; 0.36759525821131556 0.15955355357442538;;; 0.2253628270288189 0.2575367749731632; 0.29488426965210357 0.26134686863348194; 0.2574636791435125 0.3535252952896456; 0.2660638105698949 0.10918940695134041; 0.35431963290010443 0.18086109366301267; 0.19610453650530657 0.0; 0.2924830491750049 0.34613381434067925; 0.3150997523189212 0.2394800866951381;;; 0.2669927354648018 0.2648321348951373; 0.2978988298885863 0.26998129148037; 0.31214794721860584 0.2488989005280773; 0.3034388003671083 0.2952083426379613; 0.1613815226632105 0.2881744779667845; 0.33911303032522044 0.2728253995620948; 0.31397893378828684 0.0; 0.3399477072428517 0.20234479090275603;;; 0.3481136599518056 0.20628459985644343; 0.4009250069028098 0.23594682544599205; 0.2666843277965364 0.21526567513382255; 0.45339440966123584 0.32230839722878957; 0.3139790935591275 0.20426511302680414; 0.30825451976317836 0.22109685422563555; 0.279734640421502 0.311793103851355; 0.24234810775272092 0.0], [0.2767427835484444 0.25774469422226304; 0.28105102750237376 0.24083979895680727; 0.2595149062305072 0.2618425548212915; 0.2625761272079569 0.26947622779521374; 0.26999888616058937 0.24738582314805127; 0.28055551906984305 0.24694412275060976; 0.27689914681373523 0.2708880025205796; 0.2599342017075787 0.2512244910308245]
end
MZ = [2.5585 2.4535 0.7035; 1.6849 2.0349 2.3849; 2.4178 1.3678 0.6678; 1.5141 0.4641 1.5141; 0.6136 1.4535 2.8536; 2.9984 2.6484 2.6484; 3.0328 2.9978 2.2977; 2.1375 1.7875 2.4874]

rdZ() = let
    Dload = Distributions.Arcsine.(LM)
    vec = [rand.(Dload) for t in 1:T]
    [vec[t][l] for t in 1:T, l in 1:L]
end
function master() # initialization version ⚠️ will be executed more than once
    ø = JumpModel(0)
    @stage1feas_code()
    JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
    JuMP.@variable(ø, -1/length(MY) <= β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))] <= 1/length(MY))
    JuMP.@objective(ø, Min, o1 + ip(MY, β1))
    @optimise()
    @assert status == JuMP.OPTIMAL
    vldtV[1] = false # because β1 has artificial bounds
    return x, β1 = get_bin_var(x), jv(β1)
end
function master(ℶ1) # portal
    function readCut(ℶ) # for Benders (or SB) cut
        stV2, cnV2, puV2, pvV2, pxV2, pβ1V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2 = readCut(ℶ1)
    if R2 >= 1
        return x, β1, cost1plus2, oℶ1 = master(R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2)
    else # This part is necessary, because it will be executed more than once
        x, β1 = master()
        (oℶ1 = -Inf; cost1plus2 = 0.) # cost1plus2 is a finite value
        return x, β1, cost1plus2, oℶ1
    end
end
function master(R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2) # if o3 has ≥1 cut
    ø = JumpModel(0)
    @stage1feas_code()
    JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
    @add_β1()
    JuMP.@expression(ø, o2, ip(MY, β1))
    JuMP.@variable(ø, o3)
    for r in 1:R2
        if stV2[r]
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pβ1V2[r], β1)] # modify this line
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o3 >= cut_expr)
        end
    end
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        @assert status == JuMP.DUAL_INFEASIBLE
        (JuMP.set_lower_bound.(β1, -btBnd); JuMP.set_upper_bound.(β1, btBnd))
        vldtV[1] = false # because β1 has artificial bounds
        @optimise()
        @assert status == JuMP.OPTIMAL
    end
    x, β1 = get_bin_var(x), jv(β1)
    cost1plus2, oℶ1 = JuMP.value(o1) + JuMP.value(o2), JuMP.value(o3)
    return x, β1, cost1plus2, oℶ1
end
function argmaxZ(u, v, x, Y, β2) # 💻 Feat
    ø = JumpModel(2)
    JuMP.@variable(ø, 0 <= Z[t = 1:T, l = 1:L] <= LM[l])
    @dualobj_code()
    JuMP.@objective(ø, Max, -ip(Z, β2) + dualobj) # dualobj is f's
    JuMP.set_attribute(ø, "TimeLimit", 600) # 10 min
    JuMP.unset_silent(ø) # to check whether there are BestBd associated with Incumbent
    @optimise()
    status == JuMP.OPTIMAL && return get_safe_Z(Z)
    status == JuMP.TIME_LIMIT && error("Bilinear Program cannot stop in 10 min")
    error("argmaxZ(u, v, x, Y, β2): $status")
end
function argmaxZ(x, yM, iY, β2)::Matrix{Float64}
    u, v = decode_uv_from_x(x)
    return argmaxZ(u, v, x, yM[:, :, iY], β2)
end
ub_φ1(Δ2, x, β1, yM, i) = -ip(β1, yM[:, :, i]) + ub_psi(Δ2, x, i)
function ub_psi(Δ2, x::BitMatrix, Y::Int)::Float64
    i_vec = findall(t -> t == x, Δ2["x"]) ∩ findall(t -> t == Y, Δ2["Y"])
    isempty(i_vec) && return Inf
    R2 = length(i_vec)
    β2V2, fV2 = Δ2["β"][i_vec], Δ2["f"][i_vec]
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0)
    JuMP.@constraint(ø, sum(λ) == 1)
    @add_β2()
    JuMP.@constraint(ø, sum(β2V2[r] * λ[r] for r in 1:R2) .== β2)
    JuMP.@objective(ø, Min, ip(MZ, β2) + ip(fV2, λ))
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        status == JuMP.INFEASIBLE && return Inf
        error("$status")
    end
    return JuMP.objective_value(ø)
end
function argmaxindY(Δ2, x, β1, yM)::Int # NOTE: it's ret(Y) is different from the underlying true argmaxY due to inexact Δ2. Nonetheless, the resultant ub is valid.
    (NY = size(yM, 3); fullVec = zeros(NY))
    for i in 1:NY
        v = ub_φ1(Δ2, x, β1, yM, i)
        v == Inf && return i
        fullVec[i] = v
    end
    return findmax(fullVec)[2]
end
function gencut_ψ_uvx(yM, iY, uΓ, vΓ, xΓ)
    function readCut(ℶ)
        stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pY"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCut(ℶ2)
    R2 == 0 && return -Inf, zero(uΓ), zero(vΓ), zero(xΓ) # cn, pu, pv, px
    ø = JumpModel(0)
    @addMatVarViaCopy(u, uΓ)
    @addMatVarViaCopy(v, vΓ)
    @addMatVarViaCopy(x, xΓ)
    @addMatCopyConstr(cpu, u, uΓ)
    @addMatCopyConstr(cpv, v, vΓ)
    @addMatCopyConstr(cpx, x, xΓ)
    @add_β2()
    JuMP.@variable(ø, o2)
    for r in 1:R2
        if stV2[r]
            Y = yM[:, :, iY] # fixed parameter
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], Y), (pβ2V2[r], β2)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o2 >= cut_expr)
        end
    end
    JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        status == JuMP.DUAL_INFEASIBLE && return -Inf, zero(uΓ), zero(vΓ), zero(xΓ) # cn, pu, pv, px
        error("$status")
    end
    obj = jo(ø)
    pu  = jd(cpu)
    pv  = jd(cpv)
    px  = jd(cpx)
    cn = obj - ip(pu, uΓ) - ip(pv, vΓ) - ip(px, xΓ)
    return cn, pu, pv, px
end
function gencut_ℶ1(yM, iY, oψ, u, v, x) # decorator
    cn, pu, pv, px = gencut_ψ_uvx(yM, iY, u, v, x)
    pβ1 = -yM[:, :, iY]
    return cn, pu, pv, px, pβ1
end

function gencut_ℶ2(Z, yM, iY, of, u, v, x) # decorator
    cn, pu, pv, px, pY = gencut_f_uvxY(yM, Z, of, u, v, x, yM[:, :, iY])
    pβ2 = -Z
    return cn, pu, pv, px, pY, pβ2
end
function tryPush_ℶ1(yM, iY, oℶ1, x, β1)::Bool # 👍 use this directly
    u, v = decode_uv_from_x(x)
    cn, pu, pv, px, pβ1 = gencut_ℶ1(yM, iY, NaN, u, v, x)
    cn == -Inf && return false
    new_oℶ1 = cn + ip(pu, u) + ip(pv, v) + ip(px, x) + ip(pβ1, β1)
    if new_oℶ1 > oℶ1 + 1e-5
        push!(ℶ1["st"], true)
        push!(ℶ1["cn"], cn)
        push!(ℶ1["pu"], pu)
        push!(ℶ1["pv"], pv)
        push!(ℶ1["px"], px)
        push!(ℶ1["pβ"], pβ1)
        return true
    end
    @warn "ℶ1 saturation"
    return false
end
function get_trial_β2_oℶ2(ℶ2, x, yM, iY) # invoke next to argmaxY
    function readCut(ℶ)
        stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pY"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCut(ℶ2)
    ø = JumpModel(0)
    @add_β2()
    if R2 == 0
        JuMP.@objective(ø, Min, ip(MZ, β2))
        (JuMP.set_lower_bound.(β2, -btBnd); JuMP.set_upper_bound.(β2, btBnd))
        vldtV[2] = false
        @optimise()
        @assert status == JuMP.OPTIMAL
        return β2, oℶ2 = jv(β2), -Inf
    end
    JuMP.@variable(ø, o2)
    u, v = decode_uv_from_x(x)
    for r in 1:R2
        if stV2[r]
            Y = yM[:, :, iY]
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], Y), (pβ2V2[r], β2)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o2 >= cut_expr)
        end
    end
    JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        @assert status == JuMP.DUAL_INFEASIBLE
        (JuMP.set_lower_bound.(β2, -btBnd); JuMP.set_upper_bound.(β2, btBnd))
        vldtV[2] = false
        @optimise()
        @assert status == JuMP.OPTIMAL
    end
    return β2, oℶ2 = jv(β2), JuMP.value(o2)
end

function argmaxppo_master(of, u, v, x, Y) # 🫖 arg is a trial point
    rsp(matrix) = reshape(matrix, (:,))
    function read(ℸ) # for Benders (or SB) cut
        ofvV2, uV2, vV2, xV2, YV2 = ℸ["ofv"], ℸ["u"], ℸ["v"], ℸ["x"], ℸ["Y"]
        R2 = length(ofvV2)
        return R2, ofvV2, uV2, vV2, xV2, YV2
    end
    R2, ofvV2, uV2, vV2, xV2, YV2 = read(ℸ2)
    ϵ, HYPER_PARAM = 1e-5, 1.0 
    ø = JumpModel(0)
    JuMP.@variable(ø, po >= ϵ)
    @addMatVarViaCopy(pu, u)
    @addMatVarViaCopy(pv, v)
    @addMatVarViaCopy(px, x)
    @addMatVarViaCopy(pY, Y)
        JuMP.@expression(ø, pri,  vcat(rsp(u),  rsp(v),  rsp(x),  rsp(Y)))
        JuMP.@expression(ø, pai, vcat(rsp(pu), rsp(pv), rsp(px), rsp(pY)))
        JuMP.@variable(ø, api[eachindex(pai)])
        JuMP.@constraint(ø, api .>=  pai)
        JuMP.@constraint(ø, api .>= -pai)
        JuMP.@constraint(ø, HYPER_PARAM * po + sum(api) <= 1.)
    JuMP.@expression(ø, o2, -ip(pai, pri))
    JuMP.@expression(ø, o3, -po * of)
    if R2 == 0
        JuMP.@objective(ø, Max, o2 + o3)
        vldt = false
    else
        JuMP.@variable(ø, o1)
        for r in 1:R2
            tmp = [(ofvV2[r], po), (uV2[r], pu), (vV2[r], pv), (xV2[r], px), (YV2[r], pY)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp)) # anonymous because in a loop
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o1 <= cut_expr)
        end
        JuMP.@objective(ø, Max, o1 + o2 + o3)
        vldt = true
    end
    @optimise() # Linear Program
    @assert status == JuMP.OPTIMAL
    po = JuMP.value(po)
    @assert po > ϵ/2 "Gurobi's err"
    o1 = R2 == 0 ? Inf : JuMP.value(o1)
    primVal_2 = JuMP.value(o2) + JuMP.value(o3)
    pu = jv(pu)
    pv = jv(pv)
    px = jv(px)
    pY = jv(pY)
    return o1, primVal_2, po, pu, pv, px, pY
end
function try_gen_vio_lag_cut(yM, Z, ofΓ, uΓ, vΓ, xΓ, YΓ) # 🥑 use suffix "Γ" to avoid clash
    function pushTrial(ofv, u, v, x, Y)
        push!(ℸ2["ofv"], ofv)
        push!(ℸ2["u"], u)
        push!(ℸ2["v"], v)
        push!(ℸ2["x"], x)
        push!(ℸ2["Y"], Y)
    end
    [empty!(ℸ2[k]) for k in keys(ℸ2)] # ∵ param Z may vary
    if ofΓ == -Inf
        cn = lag_subproblem(yM, Z, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ))[1]
        return cn, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ)
    end
    PATIENCE = 0.9
    while true
        o1, primVal_2, po, pu, pv, px, pY = argmaxppo_master(ofΓ, uΓ, vΓ, xΓ, YΓ) # trial slope <-- hat points
        dualBnd = o1 + primVal_2
        dualBnd < cϵ && return -Inf, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ)
        cn, (ofv, u, v, x, Y) = lag_subproblem(yM, Z, po, pu, pv, px, pY) # trial cut generation
        cn > o1 + 5e-5 && @warn "final try_gen_vio_lag_cut cn=$cn | $o1=o1"
        primVal = cn + primVal_2
        primVal > PATIENCE * dualBnd && return cn, po, pu, pv, px, pY
        pushTrial(ofv, u, v, x, Y)
    end
end
function lag_subproblem(yM, Z, po, pu, pv, px, pY) # 🩳 the core of "gen cut for ofv"
    ø = JumpModel(0)
    @stage1feas_code()
    @addMatVarViaCopy(Y, pY)
    tmp = [(u, pu), (v, pv), (x, px), (Y, pY)]
    JuMP.@expression(ø, o1, mapreduce(t -> ip(t[1], t[2]), +, tmp))
    JuMP.drop_zeros!(o1)
    NY = size(yM, 3)
    JuMP.@variable(ø, ly[i = 1:NY], Bin)
    JuMP.@constraint(ø, sum(ly) == 1)
    JuMP.@constraint(ø, sum(ly[i] * yM[:, :, i] for i in 1:NY) .== Y)
    @primobj_code() # 'Z' is fixed
    JuMP.@objective(ø, Min, o1 + po * primobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    cn = JuMP.objective_value(ø)
    of = JuMP.value(primobj)
    u = jv(u)
    v = jv(v)
    x = jv(x)
    Y = jv(Y)
    return cn, (of, u, v, x, Y) # [1] RHS of lag cut [2] ℸ requisites  
end
function gencut_f_uvxY(yM, Z, of, u, v, x, Y) # lag's format to Bens'
    cn, po, pu, pv, px, pY = try_gen_vio_lag_cut(yM, Z, of, u, v, x, Y)
    cn =  cn / po
    pu = -pu / po
    pv = -pv / po
    px = -px / po
    pY = -pY / po
    return cn, pu, pv, px, pY
end
# function gencut_f_uvxY(Z, yM, iY, uΓ, vΓ, xΓ) # Ben cut
#     YΓ = yM[:, :, iY]
#     ø = JumpModel(1)
#     @addMatVarViaCopy(u, uΓ)
#     @addMatVarViaCopy(v, vΓ)
#     @addMatVarViaCopy(x, xΓ)
#     @addMatVarViaCopy(Y, YΓ)
#     @addMatCopyConstr(cpu, u, uΓ)
#     @addMatCopyConstr(cpv, v, vΓ)
#     @addMatCopyConstr(cpx, x, xΓ)
#     @addMatCopyConstr(cpY, Y, YΓ)
#     @primobj_code()
#     JuMP.@objective(ø, Min, primobj)
#     @optimise()
#     @assert status == JuMP.OPTIMAL "$status"
#     obj = jo(ø)
#     pu  = jd(cpu)
#     pv  = jd(cpv)
#     px  = jd(cpx)
#     pY  = jd(cpY)
#     cn = obj - ip(pu, uΓ) - ip(pv, vΓ) - ip(px, xΓ) - ip(pY, YΓ) 
#     return cn, pu, pv, px, pY
# end
function tryPush_ℶ2(Z, yM, iY, oℶ2, x, β2)::Bool # 👍 use this directly
    u, v = decode_uv_from_x(x)
    cn, pu, pv, px, pY, pβ2 = gencut_ℶ2(Z, yM, iY, oℶ2 + ip(β2, Z), u, v, x)
    new_oℶ2 = cn + ip(pu, u) + ip(pv, v) + ip(px, x) + ip(pY, yM[:, :, iY]) + ip(pβ2, β2)
    if new_oℶ2 > oℶ2 + 1e-5
        push!(ℶ2["st"], true)
        push!(ℶ2["cn"], cn)
        push!(ℶ2["pu"], pu)
        push!(ℶ2["pv"], pv)
        push!(ℶ2["px"], px)
        push!(ℶ2["pY"], pY)
        push!(ℶ2["pβ"], pβ2)
        return true
    end
    @warn "ℶ2 saturation"
    return false
end
phi_2(β2, x, yM, i, Z)  = -ip(β2, Z) + primobj_value(x, yM, i, Z)
function evalPush_Δ2(β2, x, yM, iY, Z) # 👍 valid when Z is the argmax, not merely a feasible solution
    push!(Δ2["f"], phi_2(β2, x, yM, iY, Z))
    push!(Δ2["x"], x)
    push!(Δ2["Y"], iY)
    push!(Δ2["β"], β2)
end

function main()
    get_delta_t(t0) = time() - t0
    TIMELIM = 1800.0 # 30 Minutes
    t0 = time()
    while true
        get_delta_t(t0) > TIMELIM && return "30 minutes is over"
        vldtV .= true
        x, β1, cost1plus2, lbℶ1 = master(ℶ1)
        while true
            iY = argmaxindY(Δ2, x, β1, yM)
            ubℶ1 = ub_φ1(Δ2, x, β1, yM, iY) # stage1's rgap(lbℶ1, ubℶ1)
            β2, oℶ2 = get_trial_β2_oℶ2(ℶ2, x, yM, iY)
            lb = cost1plus2 + lbℶ1
            ub = cost1plus2 + ubℶ1
            @info "$vldtV lb = $lb | $ub = ub"
            rgapsimple(lb, ub) < 0.01 && return "time is ($(round(get_delta_t(t0))))s"
            Z = argmaxZ(x, yM, iY, β2)
            # @info "see" pv=primobj_value(x, yM, iY, Z) dv=dualobj_value(x, yM, iY, Z)
            evalPush_Δ2(β2, x, yM, iY, Z)
            tryPush_ℶ2(Z, yM, iY, oℶ2, x, β2)
            tryPush_ℶ1(yM, iY, lbℶ1, x, β1) && break
        end
    end
end
vldtV = trues(2)
retstr = main()



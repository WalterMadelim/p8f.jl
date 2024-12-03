import LinearAlgebra
import Distributions
import Statistics
import Random
import Gurobi
import JuMP
using Logging
GRB_ENV = Gurobi.Env()

# with only lagrangian cuts
# relatively complete recourse -> dual variable has an upper bound -> estimate it -> enforce beta_bound
# although might not be very fast, the lb increasing process is steady and continual ✅
# 3/12/24

macro optimise() return esc(:((_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø)))) end
macro reoptimise()
    return esc(quote
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
    end)
end
rd6(f) = round(f; digits = 6)
gap_lu(lb, ub)       = abs(ub - lb) / max(abs(lb), abs(ub)) # the rtol in isapprox
jv(x) = JuMP.value.(x)
get_bin_var(x) = Bool.(round.(jv(x))) # applicable only when status == JuMP.OPTIMAL
ip(x, y)             = LinearAlgebra.dot(x, y)
norm1(x)             = LinearAlgebra.norm(x, 1)
@enum State begin t1; t2f; t3; t2b end
btBnd = 5.0 # 🧪 if global optimality cannot be attained, you may want to enlarge it
cϵ = 1e-6 # positive, small => more precise
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
T, G, W, L, B = 4, 2, 2, 3, 11 # 🌸 G+1 is the size of (u, v, x)
function load_UC_data(T)
    @assert T in 1:8
    CST = [0.72, 0.60, 0.63]/5;
    CSH = [0.15, 0.15, 0.15]/5;
    CL = [8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 16.0 13.776 14.443]/5;
    CL = CL[end-T+1:end, :]
    CG = [3.6, 3.4, 4.0]/5;
    C2 = [34, 23, 41]/2000
    C1 = [0.67, 0.41, 0.93]/2/5;
    C0 = CST / T;
    PI = [0.45, 0.375, 0.5];
    PS = [5.5,  4,     4.5];
    EM = C2 .* PS .* PS .+ C1 .* PS .+ C0;
    LM = [4, 3.5, 3];
    ZS = [0, 0, 1.0]; # Binary
    ZP = [0, 0, 0.5];
    NG = [3, 2] # 🍀 since `G+1` generator is on the slack bus, it doesn't contribute to any power flow, we omit it
    NW = [2, 3]
    NL = [4, 5, 6]
    FM = let
        [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    end
    BC = 2.1 * [1.0043, 2.191, 1.3047, 0.6604, 1.7162, 0.6789, 1.0538, 1.1525, 1.3338, 0.4969, 0.7816]
    (RU = [2.5, 1.9, 2.3]; SU = 1.3 * RU; RD = 1.1 * RU; SD = 1.3 * RD)
    return CST, CSH, CL, CG, C2, C1, C0, EM, PI, PS, LM, ZS, ZP, NG, NW, NL, FM, BC, RU, SU, RD, SD
end
CST, CSH, CL, CG, C2, C1, C0, EM, PI, PS, LM, ZS, ZP, NG, NW, NL, FM, BC, RU, SU, RD, SD = load_UC_data(T)
ℶ1, ℶ2, Δ1, Δ2, ℸ1, ℸ2 = let
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
        "st" => Bool[],
        "x" =>  BitMatrix[], # contain x only, where u, v can be decoded from
        "rv" => Int[], # index of Y
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
    Δ1 = Dict(
        "f" => Float64[],
        "x" => BitMatrix[],
        "β" => Matrix{Float64}[] # β1
    )
    Δ2 = Dict( # 🌸 used in argmaxY
        "f" => Float64[],
        "x" => BitMatrix[],
        "Y" => Int[],
        "β" => Matrix{Float64}[] # β2
    )
    ℶ1, ℶ2, Δ1, Δ2, ℸ1, ℸ2
end
brcs(v) = ones(T) * transpose(v) # to broadcast those timeless cost coeffs 
begin # load Y and Z's uncertainty data
    T = 4
    yM = [12.932203243205766 1.3374359039022035; 1.9446659491235203 1.4162431451003246; 1.6511993236129792 2.1773399042968133; 1.782042683828961 2.557790493339905;;; 2.0011822386947595 1.6789225705000335; 13.690850014602509 1.997467388472595; 1.9198034781728475 1.9293910538786738; 2.107819269712466 1.8111662782757756;;; 1.5403048907063075 1.9116904989768762; 1.7523927556949361 1.8959624996410318; 11.758573486702598 2.129373069191475; 2.360835863553524 1.654410550094579;;; 1.5715933164025124 1.7031010579277321; 1.8408536127147777 1.9246653956279802; 2.2612809290337474 2.075380480088833; 11.753746532489489 2.246120859260827;;; 1.4384755553357391 11.943362474288024; 1.72344593236233 1.91548036162283; 2.1236245833170844 2.0635541107258346; 2.0145900767877167 1.9949694636426214;;; 1.421390950378163 1.8195885154671323; 1.9460989041791934 12.203355270334711; 2.0120047378255426 1.2887952780925083; 2.140262568332268 1.7497035506584138;;; 1.8926711067034918 1.6778456616989779; 1.5882059667141126 0.9989786752213483; 1.9555987045048253 13.308074727575503; 2.00116104992196 1.8335475496285436;;; 2.4284307784258465 1.764570097295026; 1.6252902737904777 1.615196030466516; 1.6359452680871918 1.9888566323078063; 2.327210511773217 13.145864096700304;;; 0.0 1.9303046697671453; 1.340036139647707 2.1272564678253922; 1.9643842464561987 1.8073858830728204; 2.124702105273566 1.2240590219268372;;; 1.350599019155928 1.6358596490089128; 0.0 1.4458183973273762; 1.7581241000312189 2.446722329979181; 1.7730088350804067 1.6995757281062374;;; 1.8272891775727969 1.3644451443696721; 1.614578717229169 1.7184087355368423; 0.0 1.9730551415386182; 1.2411914714500063 1.9819256857014036;;; 1.9689580539401672 1.537755504194629; 1.2568111339355166 1.4801389249965697; 1.4256251018077006 1.9976963085038109; 0.0 1.5725426375997271;;; 1.9538021541563502 0.0; 1.4684000134553443 1.3878659549049586; 1.5755518609653678 1.9138930410237733; 1.9252030431249159 1.7622562137125504;;; 1.9884105546097959 1.4568327122536733; 1.4511724621473678 0.0; 1.8340879617445378 2.7129660182996957; 1.9294116598340794 1.9669297233369172;;; 1.3725434547187787 1.689163218152612; 1.6119892285052182 2.582693976219158; 1.7352439054027282 0.0; 2.0012409991934312 1.9773612650438648;;; 1.5651952694036888 1.4769477561641766; 1.7825097976930242 1.8205879836574106; 1.9657939031493161 2.031470938642274; 1.578855990899333 0.0]
    MY = [1.8085556385124137 1.7156454852430747; 1.8414127842661827 1.8103587289251535; 1.8688461336064663 2.09619165798559; 2.0042839049623544 1.874316195757285]
    MZ = [1.26723829950154 0.6727096064374787 2.2152435305495493; 3.0986377097936217 3.4229623321850227 1.6827441306529918; 2.8331247279184395 0.4899771169412037 2.331676836964725; 1.8849192189104738 3.2395724551885237 1.6774485560558825]
    vertexY(i) = yM[:, :, i]
    rdZ() = let
        Dload = Distributions.Arcsine.(LM)
        vec = [rand.(Dload) for t in 1:T]
        [vec[t][l] for t in 1:T, l in 1:L]
    end
    rdYZ() = vertexY(rand(1:size(yM, 3))), rdZ() # used in deterministic formulation
end
# macro primobj_code() #  entail (u, v, x, Y, Z)
#     return esc(quote
#         JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
#         JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
#         JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
#         JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
#         JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
#         JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟
#         JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g] - EM[g] * (1 - x[t, g])) # 🍟
#         JuMP.@expression(ø, gencost, sum(pe))
#         JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
#         JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
#         JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
#         JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
#         JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
#         JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
#         JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, :]))
#         JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.)
#         JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.)
#         JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
#         JuMP.@expression(ø, gccost, ip(brcs(CG), p .- [ϱ ϱsl]))
#         JuMP.@expression(ø, primobj, lscost_2 + gencost + gccost) # ⚠️ primobj ≡ ofv, while of = ofv + ofc
#     end)
# end
# macro dualobj_code() #  entail (u, v, x, Y, Z)
#     return esc(quote
#         JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
#         JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
#         JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
#         JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
#         JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
#         JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
#         JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
#         JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
#         JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
#         JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
#         JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
#         JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
#         JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
#         JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
#         JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
#         JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
#         JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
#         JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
#         JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
#         JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
#         JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
#         JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t])    >= 0.)
#         JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t])             >= 0.)
#         JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t]   >= 0.)
#         JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
#             -ip(Y, Dvp) - ip(Z, Dzt) 
#             + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
#         )
#     end)
# end

macro primobj_code() #  entail (u, v, x, Y, Z)
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
    JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
    JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
    JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
    JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
    JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 
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
macro dualobj_code() #  entail (u, v, x, Y, Z)
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
macro stage1feas_code()
    return esc(quote
        JuMP.@variable(ø, u[t = 1:T, g = 1:G+1], Bin)
        JuMP.@variable(ø, v[t = 1:T, g = 1:G+1], Bin)
        JuMP.@constraint(ø, u + v .<= 1) # 💥💥 safe constraint
        JuMP.@variable(ø, x[t = 1:T, g = 1:G+1], Bin)
        JuMP.@expression(ø, xm1, vcat(transpose(ZS), x)[1:end-1, :])
        JuMP.@constraint(ø, x .- xm1 .== u .- v)
    end)
end
macro addMatVarViaCopy(x, xΓ) return esc(:( JuMP.@variable(ø, $x[eachindex(eachrow($xΓ)), eachindex(eachcol($xΓ))]) )) end
# macro addMatCopyConstr(cpx, x, xΓ) return esc(:( JuMP.@constraint(ø, $cpx[i = eachindex(eachrow($x)), j = eachindex(eachcol($x))], $x[i, j] == $xΓ[i, j]) )) end
function decode_uv_from_x(x::BitMatrix)
    xm1 = vcat(transpose(ZS), x)[1:end-1, :]
    dif = Int.(x .- xm1)
    u = dif .== 1
    v = dif .== -1
    return u, v
end
function decode_uv_from_x(x::Matrix)
    return u, v = decode_uv_from_x(Bool.(x))
end
function dualobj_value(u, v, x, Y, Z) # Inner layer
    ø = JumpModel(0)
    @dualobj_code()
    JuMP.@objective(ø, Max, dualobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(ø)
end
function dualobj_value(x, yM, i, Z) # For debugging purpose
    u, v = decode_uv_from_x(x)
    return value = dualobj_value(u, v, x, yM[:, :, i], Z) 
end
function primobj_value(u, v, x, Y, Z) # Inner layer
    ø = JumpModel(0)
    @primobj_code()
    JuMP.@objective(ø, Min, primobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(ø)
end
function primobj_value(x, yM, i, Z)
    u, v = decode_uv_from_x(x)
    return value = primobj_value(u, v, x, yM[:, :, i], Z) 
end
# 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄
function master() # initialization version ⚠️ will be executed more than once
    ø = JumpModel(0)
    @stage1feas_code()
    JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
    JuMP.@variable(ø, -1/length(MY) <= β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))] <= 1/length(MY))
    JuMP.@objective(ø, Min, o1 + ip(MY, β1))
    @optimise()
    @assert status == JuMP.OPTIMAL
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
        return vldt, lb, x, β1, cost1plus2, oℶ1 = master(R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2)
    else # This part is necessary, because it will be executed more than once
        x, β1 = master()
        (vldt = false; lb = oℶ1 = -Inf; cost1plus2 = Inf)
        return vldt, lb, x, β1, cost1plus2, oℶ1
    end
end
function master(R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2) # if o3 has ≥1 cut
    ø = JumpModel(0)
    @stage1feas_code()
    JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
    JuMP.@variable(ø, β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))]) # free, but TBD
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
        vldt = false
        (JuMP.set_lower_bound.(β1, -btBnd); JuMP.set_upper_bound.(β1, btBnd))
        @optimise()
        @assert status == JuMP.OPTIMAL
    else
        vldt = true
    end
    lb = JuMP.objective_value(ø)
    x, β1 = get_bin_var(x), jv(β1)
    cost1plus2, oℶ1 = JuMP.value(o1) + JuMP.value(o2), JuMP.value(o3)
    return vldt, lb, x, β1, cost1plus2, oℶ1
end
function eval_Δ1(Δ1, x, β1)::Float64 # t1, in termination criterion
    i_vec = findall(t -> t == x, Δ1["x"])
    isempty(i_vec) && return Inf
    R2 = length(i_vec)
    fV2 = Δ1["f"][i_vec]
    β1V2 = Δ1["β"][i_vec]
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, sum(β1V2[r] * λ[r] for r in 1:R2) .== β1)
    JuMP.@objective(ø, Min, ip(fV2, λ))
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        status == JuMP.INFEASIBLE && return Inf
        error(" in eval_Δ1(): $status ")
    end
    return JuMP.objective_value(ø)
end
function ub_psi(Δ2, x::BitMatrix, Y::Int)::Float64 # 
    i_vec = findall(t -> t == x, Δ2["x"]) ∩ findall(t -> t == Y, Δ2["Y"])
    isempty(i_vec) && return Inf
    R2 = length(i_vec)
    fV2 = Δ2["f"][i_vec] # evaluated at final stage, thus accurate φ2
    β2V2 = Δ2["β"][i_vec]
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@variable(ø, β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))])
    JuMP.@constraint(ø, sum(β2V2[r] * λ[r] for r in 1:R2) .== β2)
    JuMP.@objective(ø, Min, ip(MZ, β2) + ip(fV2, λ))
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        status == JuMP.INFEASIBLE && return Inf
        error(" in ub_psi(), status = $status ")
    end
    return JuMP.objective_value(ø)
end
ub_φ1(Δ2, x, β1, yM, i) = -ip(β1, yM[:, :, i]) + ub_psi(Δ2, x, i) # cf. eval_Δ1, the latter is Y-free
function tryPush_Δ1(Δ2, x, β1, yM, i)::Bool # in t2b 👍 use this directly
    f = ub_φ1(Δ2, x, β1, yM, i)
    if f < Inf
        push!(Δ1["f"], f)
        push!(Δ1["x"], x)
        push!(Δ1["β"], β1)
        return true
    end
    return false
end
function argmaxindY(Δ2, x, β1, yM)::Int # 
    (NY = size(yM, 3); fullVec = zeros(NY))
    for i in 1:NY
        v = ub_φ1(Δ2, x, β1, yM, i)
        v == Inf && return i
        fullVec[i] = v
    end
    return findmax(fullVec)[2]
end
function get_trial_β2_oℶ2(ℶ2, x, yM, iY) #  invoke next to argmaxY
    function readCut(ℶ)
        stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pY"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCut(ℶ2)
    ø = JumpModel(0)
    JuMP.@variable(ø, -btBnd <= β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))] <= btBnd) # 🧪 generate limited trial β2
    if R2 == 0
        JuMP.@objective(ø, Min, ip(MZ, β2))
    else
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
    end
    @optimise()
    @assert status == JuMP.OPTIMAL " in get_trial_β2(): $status "
    oℶ2 = R2 == 0 ? -Inf : JuMP.value(o2)
    β2 = jv(β2)
    return β2, oℶ2
end
function argmaxZ(u, v, x, Y, β2) # 💻 Feat
    ø = JumpModel(2)
    JuMP.@variable(ø, 0 <= Z[t = 1:T, l = 1:L] <= LM[l])
    @dualobj_code()
    JuMP.@objective(ø, Max, ip(Z, CL .- β2) + dualobj)
    @optimise() # JuMP.unset_silent(ø)
    @assert status == JuMP.OPTIMAL
    return jv(Z)
end
function argmaxZ(x, yM, i, β2)::Matrix{Float64}
    u, v = decode_uv_from_x(x)
    return argmaxZ(u, v, x, yM[:, :, i], β2)
end
function ψ_argmaxppo_master(oψ, u, v, x) #  🫖 arg is a trial point
    rsp(matrix) = reshape(matrix, (:,))
    function read(ℸ) # for Benders (or SB) cut
        oψV2, uV2, vV2, xV2 = ℸ["oψ"], ℸ["u"], ℸ["v"], ℸ["x"]
        R2 = length(oψV2)
        return R2, oψV2, uV2, vV2, xV2
    end
    R2, oψV2, uV2, vV2, xV2 = read(ℸ1)
    ϵ, HYPER_PARAM = 1e-5, 1.0 
    ø = JumpModel(0)
    JuMP.@variable(ø, po >= ϵ)
    @addMatVarViaCopy(pu, u)
    @addMatVarViaCopy(pv, v)
    @addMatVarViaCopy(px, x)
        JuMP.@expression(ø, pri,  vcat(rsp(u),  rsp(v),  rsp(x)))
        JuMP.@expression(ø, pai, vcat(rsp(pu), rsp(pv), rsp(px)))
        JuMP.@variable(ø, api[eachindex(pai)])
        JuMP.@constraint(ø, api .>=  pai)
        JuMP.@constraint(ø, api .>= -pai)
        JuMP.@constraint(ø, HYPER_PARAM * po + sum(api) <= 1.)
    JuMP.@expression(ø, o2, -ip(pai, pri))
    JuMP.@expression(ø, o3, -po * oψ)
    if R2 == 0   
        JuMP.@objective(ø, Max, o2 + o3)
    else
        JuMP.@variable(ø, o1)
        for r in 1:R2
            tmp = [(oψV2[r], po), (uV2[r], pu), (vV2[r], pv), (xV2[r], px)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp)) # anonymous because in a loop
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o1 <= cut_expr)
        end
        JuMP.@objective(ø, Max, o1 + o2 + o3)
    end
    @optimise() # Linear Program
    @assert status == JuMP.OPTIMAL
    po = JuMP.value(po)
    @assert po > ϵ/2 "Gurobi's err"
    dualBnd = R2 == 0 ? Inf : JuMP.objective_value(ø)
    primVal_2 = JuMP.value(o2) + JuMP.value(o3)
    pu = jv(pu)
    pv = jv(pv)
    px = jv(px)
    return dualBnd, primVal_2, po, pu, pv, px
end
function ψ_try_gen_vio_lag_cut(yM, iY, oψΓ, uΓ, vΓ, xΓ) 🥑
    function pushTrial(oψ, u, v, x)
        push!(ℸ1["oψ"], oψ)
        push!(ℸ1["u"], u)
        push!(ℸ1["v"], v)
        push!(ℸ1["x"], x)
    end
    [empty!(ℸ1[k]) for k in keys(ℸ1)] # ∵ param Y may vary
    if oψΓ == -Inf
        cn = ψ_lag_subproblem(yM, iY, 1., zero(uΓ), zero(vΓ), zero(xΓ))[1] 
        return cn, 1., zero(uΓ), zero(vΓ), zero(xΓ) # generate a horizontal cut, although cn might be -Inf
    end
    PATIENCE = 0.01
    while true
        dualBnd, primVal_2, po, pu, pv, px = ψ_argmaxppo_master(oψΓ, uΓ, vΓ, xΓ)
        dualBnd < cϵ && return -Inf, 1., zero(uΓ), zero(vΓ), zero(xΓ)
        cn, (oψ, u, v, x) = ψ_lag_subproblem(yM, iY, po, pu, pv, px)
        primVal = cn + primVal_2
        @assert primVal <= dualBnd + 5e-5 "weak dual error: pV=$primVal | $dualBnd=dB"
        primVal > PATIENCE * dualBnd && return cn, po, pu, pv, px
        pushTrial(oψ, u, v, x)
    end
end
function ψ_lag_subproblem(yM, iY, po, pu, pv, px) # 🩳 according to the incumbent ℶ2
    function readCut(ℶ)
        stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pY"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCut(ℶ2)
    ø = JumpModel(0)
    @stage1feas_code()
    tmp = [(u, pu), (v, pv), (x, px)]
    JuMP.@expression(ø, o1, mapreduce(t -> ip(t[1], t[2]), +, tmp))
    JuMP.drop_zeros!(o1)
    JuMP.@variable(ø, β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]) # must be free
    JuMP.@variable(ø, objℶ2)
    for r in 1:R2
        if stV2[r]   
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], yM[:, :, iY]), (pβ2V2[r], β2)] # Y is fixed as a param
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, objℶ2 >= cut_expr)
        end
    end
    JuMP.@expression(ø, oψ, ip(MZ, β2) + objℶ2) # oψ is the essential 2nd stage convex function, resembling ofv 
    JuMP.@objective(ø, Min, o1 + po * oψ)
    @optimise()
    if status == JuMP.OPTIMAL
        cn = JuMP.objective_value(ø)
        oψ = JuMP.value(oψ)
        u = jv(u)
        v = jv(v)
        x = jv(x)
        return cn, (oψ, u, v, x) # [1] RHS of lag cut [2] ℸ requisites
    elseif status == JuMP.INFEASIBLE_OR_UNBOUNDED
        JuMP.set_attribute(ø, "DualReductions", 0)
        @optimise()
    end
    @assert status == JuMP.DUAL_INFEASIBLE "$status"
    cn = oψ = -Inf
    u, v, x = zero(pu), zero(pv), zero(px)
    return cn, (oψ, u, v, x)
end
function gencut_ψ_uvx(yM, iY, oψ, u, v, x)
    cn, po, pu, pv, px = ψ_try_gen_vio_lag_cut(yM, iY, oψ, u, v, x) # lag cut
    cn =  cn / po
    pu = -pu / po
    pv = -pv / po
    px = -px / po
    return cn, pu, pv, px # Ben cut
end
function gencut_ℶ1(yM, iY, oψ, u, v, x)
    cn, pu, pv, px = gencut_ψ_uvx(yM, iY, oψ, u, v, x)
    pβ1 = -yM[:, :, iY]
    return cn, pu, pv, px, pβ1
end
function tryPush_ℶ1(yM, iY, oψ, u, v, x)
    cn, pu, pv, px, pβ1 = gencut_ℶ1(yM, iY, oψ, u, v, x)
    cn == -Inf && return false # no cut can be generated
    push!(ℶ1["st"], true)
    push!(ℶ1["x"], x)
    push!(ℶ1["rv"], iY)
    push!(ℶ1["cn"], cn)
    push!(ℶ1["pu"], pu)
    push!(ℶ1["pv"], pv)
    push!(ℶ1["px"], px)
    push!(ℶ1["pβ"], pβ1)
    return true
end
function tryPush_ℶ1(yM, iY, oℶ1, x, β1)::Bool # 👍 use this directly
    oψ = oℶ1 + ip(β1, yM[:, :, iY])
    u, v = decode_uv_from_x(x)
    return success_flag = tryPush_ℶ1(yM, iY, oψ, u, v, x)
end
function argmaxppo_master(ofv, u, v, x, Y) # 🫖 arg is a trial point
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
    JuMP.@expression(ø, o3, -po * ofv)
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
    dualBnd = vldt ? JuMP.objective_value(ø) : Inf
    primVal_2 = JuMP.value(o2) + JuMP.value(o3)
    pu = jv(pu)
    pv = jv(pv)
    px = jv(px)
    pY = jv(pY)
    return dualBnd, primVal_2, po, pu, pv, px, pY
end
function try_gen_vio_lag_cut(yM, Z, ofvΓ, uΓ, vΓ, xΓ, YΓ) # 🥑 use suffix "Γ" to avoid clash
    function pushTrial(ofv, u, v, x, Y)
        push!(ℸ2["ofv"], ofv)
        push!(ℸ2["u"], u)
        push!(ℸ2["v"], v)
        push!(ℸ2["x"], x)
        push!(ℸ2["Y"], Y)
    end
    [empty!(ℸ2[k]) for k in keys(ℸ2)] # ∵ param Z may vary
    if ofvΓ == -Inf
        cn = lag_subproblem(yM, Z, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ))[1]
        return cn, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ)
    end
    PATIENCE = 0.01
    while true
        dualBnd, primVal_2, po, pu, pv, px, pY = argmaxppo_master(ofvΓ, uΓ, vΓ, xΓ, YΓ)
        dualBnd < cϵ && return -Inf, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ)
        cn, (ofv, u, v, x, Y) = lag_subproblem(yM, Z, po, pu, pv, px, pY)
        primVal = cn + primVal_2
        @assert primVal <= dualBnd + 5e-5 "weak dual error: pV=$primVal | $dualBnd=dB"
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
    ofv = JuMP.value(primobj)
    u = jv(u)
    v = jv(v)
    x = jv(x)
    Y = jv(Y)
    return cn, (ofv, u, v, x, Y) # [1] RHS of lag cut [2] ℸ requisites  
end
function gencut_ofv_uvxY(yM, Z, ofv, u, v, x, Y) # lag's format to Bens'
    cn, po, pu, pv, px, pY = try_gen_vio_lag_cut(yM, Z, ofv, u, v, x, Y)
    cn =  cn / po
    pu = -pu / po
    pv = -pv / po
    px = -px / po
    pY = -pY / po
    return cn, pu, pv, px, pY
end
function gencut_f_uvxY(yM, Z, ofv, u, v, x, Y) # append ofc
    cn, pu, pv, px, pY = gencut_ofv_uvxY(yM, Z, ofv, u, v, x, Y)
    cn += ip(CL, Z)
    return cn, pu, pv, px, pY
end
function gencut_f_uvxY(yM, i, Z, ofv, x) # a sheer wrapper for input style
    u, v = decode_uv_from_x(x)
    Y = yM[:, :, i]
    return gencut_f_uvxY(yM, Z, ofv, u, v, x, Y)
end
function gencut_ℶ2(yM, i, Z, ofv, x)
    cn, pu, pv, px, pY = gencut_f_uvxY(yM, i, Z, ofv, x)
    pβ2 = -Z
    return cn, pu, pv, px, pY, pβ2
end
function tryPush_ℶ2(yM, i, Z, ofv, x)
    cn, pu, pv, px, pY, pβ2 = gencut_ℶ2(yM, i, Z, ofv, x)
    cn == -Inf && return false # no cut can be generated
    push!(ℶ2["st"], true)
    push!(ℶ2["cn"], cn)
    push!(ℶ2["pu"], pu)
    push!(ℶ2["pv"], pv)
    push!(ℶ2["px"], px)
    push!(ℶ2["pY"], pY)
    push!(ℶ2["pβ"], pβ2)
    return true
end
function tryPush_ℶ2(yM, i, Z, oℶ2, x, β2)::Bool # 👍 use this directly
    ofv = oℶ2 + ip(β2, Z) - ip(CL, Z)
    return success_flag = tryPush_ℶ2(yM, i, Z, ofv, x)
end
function f(x, yM, i, Z)
    ofc = ip(CL, Z)
    ofv = primobj_value(x, yM, i, Z)
    ofv2 = dualobj_value(x, yM, i, Z)
    @assert isapprox(ofv, ofv2; rtol = 1e-5) "ofv = $ofv | $ofv2 = ofv2" # to assure the validity of most hazardous Z
    return of = ofc + ofv
end
phi_2(β2, x, yM, i, Z) = -ip(β2, Z) + f(x, yM, i, Z)
function evalPush_Δ2(β2, x, yM, i, Z) # 👍 use this directly
    push!(Δ2["f"], phi_2(β2, x, yM, i, Z))
    push!(Δ2["x"], x)
    push!(Δ2["Y"], i)
    push!(Δ2["β"], β2)
end
# 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄
_, _, x, β1, _, oℶ1 = master(ℶ1)
iY = argmaxindY(Δ2, x, β1, yM)
β2, oℶ2 = get_trial_β2_oℶ2(ℶ2, x, yM, iY)
Z = argmaxZ(x, yM, iY, β2)
gCnt = [0]
termination_flag = falses(1)
lbV = Float64[] # to draw pictures
xV, β1V, oℶ1V = [x], [β1], [oℶ1]
iYV = [iY]
β2V, oℶ2V = [β2], [oℶ2]
ZV = [Z]
tV = [t1]
hint = falses(2)
function xbo1(is_premain::Bool)
    vldt, lb, x, β1, cost1plus2, oℶ1 = master(ℶ1)
    str = "lb = $lb, o12 = $cost1plus2, oℶ1 = $oℶ1,"
    hint[2] && (str *= " ℶ1")
    hint[1] && (str *= " Δ1")
    @info str
    if vldt
        @info "▶▶ master's vldt is true"
        return true
    end
    xV[1], β1V[1], oℶ1V[1] = x, β1, oℶ1
    tV[1] = t2f
    return false
end
function xbo1()
    vldt, lb, x, β1, cost1plus2, oℶ1 = master(ℶ1)
    φ1ub = eval_Δ1(Δ1, x, β1)
        ub = cost1plus2 + φ1ub
        gap = gap_lu(lb, ub)
        push!(lbV, lb)
        lb, ub = rd6(lb), rd6(ub)
        str = "t1:[g$(gCnt[1])]($vldt)lb $lb | $ub ub, gap $gap"
        hint[2] && (str *= " ℶ1")
        hint[1] && (str *= " Δ1")
        @info str
        if gap < 8/1000
            @info " 😊 gap < 8/1000, thus terminate at next t3 "
            termination_flag[1] = true
        end
    xV[1], β1V[1], oℶ1V[1] = x, β1, oℶ1
    tV[1] = t2f
end
function ybo2(yM)
    x, β1 = xV[1], β1V[1]
    iYV[1] = iY = argmaxindY(Δ2, x, β1, yM)
    β2V[1], oℶ2V[1] = get_trial_β2_oℶ2(ℶ2, x, yM, iY)
    tV[1] = t3
end
function zb2d2(yM)
    x, iY, β2, oℶ2 = xV[1], iYV[1], β2V[1], oℶ2V[1]
    ZV[1] = Z = argmaxZ(x, yM, iY, β2)
    termination_flag[1] && return true
    tryPush_ℶ2(yM, iY, Z, oℶ2, x, β2) || @warn "t3: ℶ2 saturation"
    evalPush_Δ2(β2, x, yM, iY, Z)
    gCnt[1] += 1
    tV[1] = t2b
    return false
end
function yb1d1(yM)
    x, β1, oℶ1 = xV[1], β1V[1], oℶ1V[1]
    iYV[1] = iY = argmaxindY(Δ2, x, β1, yM)
    hint[1] = tryPush_Δ1(Δ2, x, β1, yM, iY) # (x, β1) ∈ Fwd | Y ∈ Bwd
    hint[2] = gb = tryPush_ℶ1(yM, iY, oℶ1, x, β1) # x ∈ Fwd | Y ∈ Bwd
    tV[1] = gb ? t1 : t2f
end
function main(yM, is_premain::Bool)
    while true # 🌸 to make master has lower_bound even if β1 is free
        t = tV[1]
        if t == t1
            xbo1(is_premain) && break
        elseif t == t2f
            ybo2(yM)
        elseif t == t3
            zb2d2(yM)
        elseif t == t2b
            yb1d1(yM)
        end
    end
end
function main(yM)
    while true # 🌸 to make master has lower_bound even if β1 is free
        t = tV[1]
        if t == t1
            xbo1()
        elseif t == t2f
            ybo2(yM)
        elseif t == t3
            zb2d2(yM) && break
        elseif t == t2b
            yb1d1(yM)
        end
    end
end

main(yM, true)
main(yM)


if false########################################################################################################################
    function master(ℶ1, yM)
        function readCut(ℶ) # for Benders (or SB) cut
            stV2, cnV2, puV2, pvV2, pxV2, pβ1V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pβ"]
            R2 = length(cnV2)
            return R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2
        end
        function my_callback_function(cb_data, cb_where::Cint) # 🧪 the odds are that the FINAL optimal solution is not present in this callback_function
            vfun(x) = JuMP.callback_value(cb_data, x)
            cb_where == Gurobi.GRB_CB_MIPSOL || return
            Gurobi.load_callback_variable_primal(cb_data, cb_where)
            let # physical constraint
                uht = vfun.(u) # ⚠️ you must use DIFFERENT name for the numerical ret
                vht = vfun.(v) 
                key = findfirst(b -> b > 1.5, uht + vht)
                isnothing(key) || (JuMP.MOI.submit(ø, JuMP.MOI.LazyConstraint(cb_data), JuMP.@build_constraint(u[key] + v[key] <= 1.)); return)
            end
            lb, oℶ1, cost1plus2 = vfun(o123), vfun(o3), vfun(o1) + vfun(o2)
            @info "(cb): lb = $lb, o12 = $cost1plus2, oℶ1 = $oℶ1,"
            xht, βht = Bool.(round.(vfun.(x))), vfun.(β1)
            iY = argmaxindY(Δ2, xht, βht, yM)
            β2, oℶ2 = get_trial_β2_oℶ2(ℶ2, xht, yM, iY)
            Z = argmaxZ(xht, yM, iY, β2)
            tryPush_ℶ2(yM, iY, Z, oℶ2, xht, β2) || @warn "(cb): ℶ2 saturation"
            evalPush_Δ2(β2, xht, yM, iY, Z)
            iY = argmaxindY(Δ2, xht, βht, yM)
            tryPush_Δ1(Δ2, xht, βht, yM, iY)
            Y = yM[:, :, iY]
            oψ = oℶ1 + ip(βht, Y)
            uht, vht = decode_uv_from_x(xht)
            any(uht .& vht) && error("∃ u and v are 1 simultaneously\n u = $uht\n v = $vht")
            cn, pu, pv, px = gencut_ψ_uvx(yM, iY, oψ, uht, vht, xht)
            if cn == -Inf
                @warn "(cb): no more new cut for ℶ1"
                return
            end
            JuMP.MOI.submit(ø, JuMP.MOI.LazyConstraint(cb_data), JuMP.@build_constraint(o3 >= cn + ip(pu, u) + ip(pv, v) + ip(px, x) + ip(-Y, β1)))
        end
        R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2 = readCut(ℶ1)
        ø = JumpModel(2)
            JuMP.@variable(ø, u[t = 1:T, g = 1:G+1], Bin)
            JuMP.@variable(ø, v[t = 1:T, g = 1:G+1], Bin)
            JuMP.@variable(ø, x[t = 1:T, g = 1:G+1], Bin)
            JuMP.@expression(ø, xm1, vcat(transpose(ZS), x)[1:end-1, :])
            JuMP.@constraint(ø, x .- xm1 .== u .- v)
            JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
        JuMP.@variable(ø, β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))])
        JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, o2 >= ip(MY, β1))
        JuMP.@variable(ø, o3)
        for r in 1:R2
            if stV2[r]
                tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pβ1V2[r], β1)] # modify this line
                cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
                JuMP.drop_zeros!(cut_expr)
                JuMP.@constraint(ø, o3 >= cut_expr)
            end
        end
        JuMP.@expression(ø, o123, o1 + o2 + o3)
        JuMP.@objective(ø, Min, o123)
        JuMP.MOI.set(ø, JuMP.MOI.RawOptimizerAttribute("LazyConstraints"), 1)
        JuMP.MOI.set(ø, Gurobi.CallbackFunction(), my_callback_function)
        @optimise() # --> Go to callback
        @assert status == JuMP.OPTIMAL
        lb = JuMP.value(o123)
        x = get_bin_var(x)
        β1 = jv(β1)
        cost1plus2 = JuMP.value(o1) + JuMP.value(o2)
        oℶ1 = JuMP.value(o3) # 🌸 the trial point gened for ℶ1 is ((u, v, x, β1), o3)
        vldt = true
        return vldt, lb, x, β1, cost1plus2, oℶ1
    end
    master(ℶ1, yM) # working...  working...  working...  working...  
end########################################################################################################################



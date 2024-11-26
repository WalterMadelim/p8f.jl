# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import LinearAlgebra
import Distributions
import Statistics
import MosekTools
import Polyhedra
import Random
import Gurobi
import JuMP
using Logging

# T = 4 test okay
# 26/11/24

GRB_ENV = Gurobi.Env()
macro optimise() return esc(:((_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø)))) end
ip(x, y) = LinearAlgebra.dot(x, y)
norm1(x) = LinearAlgebra.norm(x, 1)
LT(P, c, x) = P \ (x .+ c) # from STD to Desired 💡 x can be both Vector and Matrix
function get_Pc(yM::Matrix)
    N, S = size(yM)
    ø = JumpModel(1) # 🌸 MOSEK
    JuMP.@variable(ø, P[1:N, 1:N])
    JuMP.@variable(ø, c[1:N])
    JuMP.@variable(ø, L[i = 1:N, j = 1:i]) # SparseAxisArray
    JuMP.@variable(ø, v[1:N])
    JuMP.@variable(ø, ae[1:S, 1:N]) # ancillary variable: abs(entry)
    JuMP.@expression(ø, e[s = 1:S, i = 1:N], ip(P[i, :], yM[:, s]) - c[i])
    JuMP.@expression(ø, Lmat, [(j > i ? 0. * L[i, i] : 1. * L[i, j]) for i in 1:N, j in 1:N])
    JuMP.@expression(ø, LDiagmat, [(j == i ? 1. * L[i, i] : 0. * L[i, i]) for i in 1:N, j in 1:N])
    JuMP.@constraint(ø, [s = 1:S], sum(ae[s, :]) <= 1.) # the ||⋅||₁ <= 1 constraint
    JuMP.@constraint(ø, [s = 1:S, i = 1:N],  e[s, i] <= ae[s, i])    
    JuMP.@constraint(ø, [s = 1:S, i = 1:N], -e[s, i] <= ae[s, i])    
    JuMP.@constraint(ø, [i = 1:N], [v[i], 1., L[i, i]] in JuMP.MOI.ExponentialCone()) # exp(v) <= L
    JuMP.@constraint(ø, [P Lmat; transpose(Lmat) LDiagmat] in JuMP.PSDCone())
    JuMP.@objective(ø, Max, sum(v))
    @optimise()
    if status != JuMP.OPTIMAL
        error(" $status ")
    end
    return P, c = JuMP.value.(P), JuMP.value.(c)
end
function is_in(y, yM)
    N, S = size(yM)
    ø = JumpModel(0)
    JuMP.@variable(ø, c[1:S] >= 0.)
    JuMP.@constraint(ø, sum(c) == 1.)
    JuMP.@constraint(ø, [i = 1:N], y[i] == ip(yM[i, :], c))
    @optimise()
    status == JuMP.OPTIMAL && return JuMP.value.(c)
    error(" in is_in: status = $status")
end
ds(ref, y) = LinearAlgebra.norm(ref .- y, 1)
ds(ref, y::Matrix) = [ds(ref, v) for v in eachcol(y)]
ds(y::Matrix, ::Any) = error(" malfunction ")
function ptsD0(N) return [zeros(N) one(ones(N, N))] end
function pts2N(N)
    a, b = LinearAlgebra.Diagonal(ones(N)), LinearAlgebra.Diagonal(-ones(N))
    [(c % 2 == 0) ? b[i, div(c, 2)] : a[i, div(c+1, 2)] for i in 1:N, c in 1:2N]
end
vov2m(vec) = [vec[c][r] for r in eachindex(vec[1]), c in eachindex(vec)]
m2vov(mat) = [Vector(c) for c in eachcol(mat)]
@enum State begin t1; t2f; t3; t2b end
(Δβ = 1.5; βnm1V = 0.0 : Δβ : 5e3) # hyper-param
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        # JuMP.set_attribute(ø, "QCPDual", 1)
        # vio = JuMP.get_attribute(ø, Gurobi.ModelAttribute("MaxVio")) 🍀 we suggest trial-and-error to decide if a constr is tight, rather than using this cmd
    elseif i == 1 
        ø = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 
        ø = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(ø) # JuMP.unset_silent(ø)
    return ø
    # (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    # if status != JuMP.OPTIMAL
    #     if status == JuMP.INFEASIBLE_OR_UNBOUNDED
    #         JuMP.set_attribute(ø, "DualReductions", 0)
    #         (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    #     end
    #     if status == JuMP.DUAL_INFEASIBLE
    #         @info "The program is unbounded"
    #         error()
    #     else
    #         error(" $status ")
    #     end
    # else
    #     return worstObj, worstZ = JuMP.objective_value(ø), JuMP.value.(Z)
    # end
end
macro stage_1_code()
    return esc(:(
        begin
            JuMP.@variable(ø, u[t = 1:T, g = 1:G+1], Bin)
            JuMP.@variable(ø, v[t = 1:T, g = 1:G+1], Bin)
            JuMP.@variable(ø, x[t = 1:T, g = 1:G+1], Bin)
            JuMP.@constraint(ø, [g = 1:G+1],          x[1, g] - ZS[g]     == u[1, g] - v[1, g])
            JuMP.@constraint(ø, [t = 2:T, g = 1:G+1], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
            JuMP.@expression(ø, o1, sum(CST[g] * u[t, g] + CSH[g] * v[t, g] for t in 1:T, g in 1:G+1))
        end
    ))
end
macro f_prim_code()
    return esc(:(
        begin
            JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])       
            JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
            JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
            JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 in place of ϱ[t, G+1]
            JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w]) 
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
            JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RC)             # 🍀
            JuMP.@constraint(ø, Du1[g = 1:G+1], RC >= p[1, g] - ZP[g])              # 🍀
            JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RC) # 🍀
            JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RC >= p[t, g] - p[t-1, g])  # 🍀
            JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
                sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
                + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
                - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
            )
            JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b])
            JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])
            JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
            JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
            JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
            JuMP.@expression(ø, gencost, sum(C1[g] * p[t, g] for t in 1:T, g in 1:G+1))
            JuMP.@expression(ø, primobj, lscost_2 + gccost_1 + gccost_2 + gencost)
        end
    ))
end
macro f_dual_code()
    return esc(:(
        begin
            JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.)
            JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.)
            JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
            JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.)
            JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.)
            JuMP.@constraint(ø, p1[g = 1:G], CG[g] + C1[g] + Dps[1, g] - Dpi[1, g] - Dvr[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0.) # 🍀
            JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], CG[g] + C1[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0.) # 🍀
            JuMP.@constraint(ø, pT[g = 1:G], CG[g] + C1[g] + Dps[T, g] - Dpi[T, g] - Dvr[T, g] + Du[T, g] - Dd[T, g] == 0.) # 🍀
            JuMP.@constraint(ø, psl1, CG[G+1] + C1[G+1] + Dps[1, G+1] - Dpi[1, G+1] - Dϱu[1] + Du1[G+1] - Dd1[G+1] + Dd[1+1, G+1] - Du[1+1, G+1] == 0.) # 🍀slack
            JuMP.@constraint(ø, psl2[t = 2:T-1], CG[G+1] + C1[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t] + Du[t, G+1] - Dd[t, G+1] + Dd[t+1, G+1] - Du[t+1, G+1] == 0.) # 🍀slack
            JuMP.@constraint(ø, pslT, CG[G+1] + C1[G+1] + Dps[T, G+1] - Dpi[T, G+1] - Dϱu[T] + Du[T, G+1] - Dd[T, G+1] == 0.) # 🍀slack
            JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
            JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
            JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
            JuMP.@expression(ø, dualobj, -ip(Y, Dvp) - ip(Z, Dzt) + ip(Dd1 .- Du1, ZP)
                - RC * (sum(Dd1) + sum(Du1) + sum(Dd) + sum(Du))
                + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
                - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
            )
        end
    ))
end
function load_UC_data(T)
    CST = [0.72, 0.60, 0.63]/5;
    CSH = [0.15, 0.15, 0.15]/5;
    @assert T in 1:8
    CL = [8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 16.0 13.776 14.443]/5;
    CL = CL[end-T+1:end, :]
    CG = [3.6, 3.4, 4.0]/5;
    C1 = [0.67, 0.41, 0.93]/5;
    PI = [0.45, 0.375, 0.5];
    PS = [5.5,  4,     4.5];
    LM = [4, 3.5, 3];
    ZS = [0, 0, 1.0]; # Binary
    ZP = [0, 0, 0.5];
    NG = [3, 2] # 🍀 since `G+1` generator is on the slack bus, it doesn't contribute to any power flow, we omit it
    NW = [2, 3]
    NL = [4, 5, 6]
    FM = let
        [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    end
    BC = 2.0 * [1.0043, 2.191, 1.3047, 0.6604, 1.7162, 0.6789, 1.0538, 1.1525, 1.3338, 0.4969, 0.7816]
    RC = 1.7 # ramping cap
    return CST, CSH, CL, CG, C1, PI, PS, LM, ZS, ZP, NG, NW, NL, FM, BC, RC
end

T, G, W, L, B = 4, 2, 2, 3, 11 # 🌸 G+1 is the size of (u, v, x)
CST, CSH, CL, CG, C1, PI, PS, LM, ZS, ZP, NG, NW, NL, FM, BC, RC = load_UC_data(T)

begin # build (MY, yM)
    N = T * W
    Random.seed!(23)
    ℙ = let
        ℙ = Distributions.Uniform(0.05, 0.5)
        lb = rand(ℙ, N)
        ℙ = Distributions.Uniform(3.3, 3.8)
        ub = rand(ℙ, N)
        Distributions.Product(Distributions.Uniform.(lb, ub)) # if you use other underlying distributions, they maybe inefficient to sample, i.e., many many time before you can sample an enclosing y
    end
    MY = Statistics.mean(ℙ)
    y_peripheral, dist_y_peri = let # collect a set of points that encloses MY in all directions
        a = [[2 for _ in 1:N]; N];
        y_peripheral = NaN * ones(a...);
        a = [2 for _ in 1:N];
        dist_y_peri = -Inf * ones(a...);
        y_peripheral, dist_y_peri
    end;
    for b in 1:150
        for s in 1:2^N
            y = rand(ℙ)
            Δ = y .- MY
            a, d = (Δ .> 0) .+ 1, norm1(Δ)
            if d > dist_y_peri[a...]
                y_peripheral[a..., :] .= y
                dist_y_peri[a...] = d
            end
        end
        mi, ma = minimum(dist_y_peri), maximum(dist_y_peri)
        print("b = $b, min_dist = $mi, MAX_dist = $ma  \r")
    end
    @assert all(-Inf .< [minimum(dist_y_peri), maximum(dist_y_peri)])
    y_peripheral = Matrix(transpose(reshape(y_peripheral, (:, N))));
    yM_ref, yM_eft, yM_ineft = let
        P, c = get_Pc(y_peripheral)
        @assert all(LinearAlgebra.eigen(P).values .> 0) " invalid transform P "
        yM_ref = LT(P, c, pts2N(N)); # 💡 extreme points
        @assert all( sum(yM_ref .< 0; dims = 2) .== ones(N) ) " abnormal yM_ref "
        yM_eft = yM_ref[:, [(c % 2 == 1) for c in 1:2N]] # valid points
        yM_ineft = yM_ref[:, [(c % 2 == 0) for c in 1:2N]] # for invalid negatives
        yM_ref, yM_eft, yM_ineft
    end
    y = rand(ℙ)
    min_dist = ds(y, yM_ineft)
    yM_delegate = vov2m([y for _ in eachindex(min_dist)])
    for b in 1:10000
        for s in 1:5000
            y = rand(ℙ)
            _, c = findmin(y)
            d = ds(y, yM_ineft[:, c])
            if d < min_dist[c]
                yM_delegate[:, c] .= y
                min_dist[c] = d
            end
        end
        print("b = $b, sumdist = $(sum(min_dist)) \r")
    end
    yM_center = let
        p = [yM_ineft[i, i] / (yM_ineft[i, i] - yM_delegate[i, i]) for i in 1:N]
        cm = vov2m([p[i] * yM_delegate[:, i] + (1 - p[i]) * yM_ineft[:, i] for i in 1:N])
        cm[one(ones(Bool, (N, N)))] .= 0 # to round small floating point errors
        cm
    end
    yM = [yM_eft yM_center] # a trial
    yM = let
        v = Polyhedra.vrep(m2vov(yM));
        v_rmed = Polyhedra.removevredundancy(v, Gurobi.Optimizer) # 💡 this procedure is efficient
        vov2m(v_rmed.points)
    end
    is_in(MY, yM) # 🌸 you must test
    @assert Polyhedra.ininterior(MY, Polyhedra.polyhedron(Polyhedra.vrep(m2vov(yM)))) # 🌸 test if possible
    # Polyhedra.volume(poly)
    # h = Polyhedra.hrep(poly)
end
Yvec2mat(vec) = reshape(vec, (T, W))
MY, yM = Yvec2mat(MY), cat([Matrix(Yvec2mat(v)) for v in eachcol(yM)]..., dims = 3)

Random.seed!(8544002554851888986)
MZ = let
    Dload = Distributions.Arcsine.(LM)
    vec = [rand.(Dload) for t in 1:T]
    [vec[t][l] for t in 1:T, l in 1:L]
end

u, v, x, Y, Z, x1, x2, β1, β2 = let
    u, v, x = 1. * rand(Bool, T, G+1), 1. * rand(Bool, T, G+1), 1. * rand(Bool, T, G+1)
    x1 = u, v, x
    β1, Y = rand(T, W), rand(T, W)
    x2 = x1, Y
    β2, Z = rand(T, L), rand(T, L)
    u, v, x, Y, Z, x1, x2, β1, β2
end
Δ1, Δ2, ℶ1, ℶ2, ℶu = let
    ℶu = Dict( # surrogate \underline_{h} (SP_lb_surrogate)
        "cn" => Float64[],
        "px" => typeof(x1)[]
    ) # each element cut is generated at (x_trial, Q_trial, P_trial)
    ℶ1 = Dict( 
        "st" => Bool[],
        "x" =>  typeof(x1)[],
        "rv" => Int[],
        "cn" => Float64[],
        "px" => typeof(x1)[],
        "pβ" => typeof(β1)[]
    )
    ℶ2 = Dict(
        "st" => Bool[],
        "x" =>  typeof(x2)[],
        "rv" => typeof(Z)[],
        "cn" => Float64[],
        "px" => typeof(x2)[],
        "pβ" => typeof(β2)[]
    )
    Δ1 = Dict(
        "f" => Float64[],
        "x" => typeof(x1)[],
        "β" => typeof(β1)[]
    )
    Δ2 = Dict(
        "f" => Float64[],
        "x" => typeof(x2)[],
        "β" => typeof(β2)[]
    )
    Δ1, Δ2, ℶ1, ℶ2, ℶu
end
function pushCut(D, x, rv, cn, px, pβ) return (push!(D["st"], true); push!(D["x"], x); push!(D["rv"], rv); push!(D["cn"], cn); push!(D["px"], px); push!(D["pβ"], pβ)) end
function pushSimplicial(D, f, x, β) return (push!(D["f"], f); push!(D["x"], x); push!(D["β"], β)) end
function readCut(ℶ)
    cnV2, pxV2, pβV2, stV2 = ℶ["cn"], ℶ["px"], ℶ["pβ"], ℶ["st"]
    R2 = length(cnV2)
    return R2, stV2, cnV2, pxV2, pβV2
end
function readSimplicial(Δ)
    fV, xV, βV = Δ["f"], Δ["x"], Δ["β"]
    R2 = length(fV)
    return R2, fV, xV, βV
end
macro master_o2_o3()
    return esc(quote
        JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, o2 >= ip(MY, β1))
        JuMP.@variable(ø, o3)
        for r in 1:R2
            stV2[r] && JuMP.@constraint(ø, o3 >= cnV2[r] + ip(px1V2[r], (u, v, x)) + ip(pβ1V2[r], β1))
        end
    end)
end
macro master_get_ret()
    return esc(quote
        lb, x1, β1, cost1plus2, o3 = let
            lb = JuMP.objective_value(ø)
            x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
            β1 = JuMP.value.(β1)
            cost1plus2 = JuMP.value(o1) + JuMP.value(o2)
            o3 = JuMP.value(o3)
            lb, x1, β1, cost1plus2, o3
        end
    end)
end
macro add_beta_nm1Bnd(b) # invoked in master() and psi()
    return esc(quote
        (bind = iCnt[1]; βnormBnd = βnm1V[bind])
        bind == length(βnm1V) && error(" enlarge the scale of βnm1V please. ")
        JuMP.@variable(ø, aβ[eachindex(eachrow($b)), eachindex(eachcol($b))])
        JuMP.@constraint(ø, aβ .>=  $b)
        JuMP.@constraint(ø, aβ .>= -$b)
        JuMP.@constraint(ø, sum(aβ) <= βnormBnd)
    end)
end
function master(masterIsMature, iCnt, R2, stV2, cnV2, px1V2, pβ1V2) # enforcing boundedness version
    ø = JumpModel(0)
    @stage_1_code()
    JuMP.@variable(ø, β1[t = 1:T, w = 1:W])
    @add_beta_nm1Bnd(β1)
    @master_o2_o3()
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    @optimise()
    @assert status == JuMP.OPTIMAL " in master1(), $status "
    @master_get_ret()
    (norm1(β1) > βnormBnd - Δβ/3) ? (iCnt[1] += 1) : (masterIsMature[1] = true)
    return lb, x1, β1, cost1plus2, o3
end
function master(R2, stV2, cnV2, px1V2, pβ1V2) # final version
    ø = JumpModel(0)
    @stage_1_code()
    JuMP.@variable(ø, β1[t = 1:T, w = 1:W])
    @master_o2_o3()
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    @optimise()
    @assert status == JuMP.OPTIMAL " in masterFinal(), $status "
    @master_get_ret()
    return lb, x1, β1, cost1plus2, o3
end
function master(masterIsMature, iCnt, ℶ1) # portal
    R2, stV2, cnV2, px1V2, pβ1V2 = readCut(ℶ1)
    if masterIsMature[1]
        lb, x1, β1, cost1plus2, o3 = master(R2, stV2, cnV2, px1V2, pβ1V2) # 3️⃣
        return true, lb, x1, β1, cost1plus2, o3
    else
        if R2 >= 1
            lb, x1, β1, cost1plus2, o3 = master(masterIsMature, iCnt, R2, stV2, cnV2, px1V2, pβ1V2) # 2️⃣
            return false, lb, x1, β1, cost1plus2, o3
        else
            x1, β1 = master() # 1️⃣
            return false, -Inf, x1, β1, NaN, -Inf
        end
    end
end
function master() # initialization version
    ø = JumpModel(0)
    @stage_1_code()
    JuMP.@variable(ø, -Δβ <= β1[t = 1:T, w = 1:W] <= Δβ)
    JuMP.@objective(ø, Min, o1 + ip(MY, β1))
    @optimise()
    @assert status == JuMP.OPTIMAL " in master(), $status "
    x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
    β1 = JuMP.value.(β1)
    return x1, β1
end
function psi(iCnt, x1, Y) # t2f, 
    x2 = x1, Y # 🍀 this is fixed
    R2, _, cnV2, px2V2, pβ2V2 = readCut(ℶ2)
    ø = JumpModel(0)
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
    @add_beta_nm1Bnd(β2)
    if R2 == 0
        JuMP.@objective(ø, Min, ip(MZ, β2))
    else
        JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, [r = 1:R2], o2 >= cnV2[r] + ip(px2V2[r], x2) + ip(pβ2V2[r], β2))
        JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    end
    @optimise()
    @assert status == JuMP.OPTIMAL " in psi(): $status "
    β2 = JuMP.value.(β2)
    let
        bnm = norm1(β2)
        if bnm > βnormBnd - Δβ/3
            iCnt[1] += 1
        elseif bnm < (βnormBnd - Δβ) - Δβ/3
            iCnt[1] -= 1
        end
    end
    return x2, β2
end
function gen_cut_for_ℶ1(x1, Y) # t2b
    function gen_cut_ψ_wrt_x1(x1Γ, Y)
        R2, _, cnV2, px2V2, pβ2V2 = readCut(ℶ2)
        @assert R2 >= 1
        ø = JumpModel(0)
        JuMP.@variable(ø, x1[i = 1:3, t = 1:T, g = 1:G+1]) # a part of x2
        JuMP.@constraint(ø, cp[i = 1:3, t = 1:T, g = 1:G+1], x1[i, t, g] == x1Γ[i][t, g])
        JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
        JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, [r = 1:R2], o2 >= cnV2[r] + ip(px2V2[r], ((x1[1, :, :], x1[2, :, :], x1[3, :, :]), Y)) + ip(pβ2V2[r], β2))
        JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
        @optimise()
        if status != JuMP.OPTIMAL
            if status == JuMP.INFEASIBLE_OR_UNBOUNDED
                JuMP.set_attribute(ø, "DualReductions", 0)
                @optimise()
            end
            status == JuMP.DUAL_INFEASIBLE && return -Inf
            error("in gen_cut_ψ_wrt_x1: $status")
        else
            tmp = JuMP.dual.(cp)
            px1 = tmp[1, :, :], tmp[2, :, :], tmp[3, :, :]
            cn = JuMP.objective_value(ø) - ip(px1, x1Γ)
            return cn, px1
        end
    end
    ret = gen_cut_ψ_wrt_x1(x1, Y)
    if length(ret) == 1
        return ret
    else
        cn, px1 = ret
        pβ1 = -Y # 💡 this is fixed, and irrespective of 'β1'
        return cn, px1, pβ1
    end
end
function maximize_φ2_over_Z(x2, β2)
    (u, v, x), Y = x2
    ø = JumpModel(2)
    JuMP.@variable(ø, 0. <= Z[t = 1:T, l = 1:L] <= LM[l])
    JuMP.@expression(ø, f_cn, ip(CL, Z))
    @f_dual_code()
    JuMP.@objective(ø, Max, -ip(β2, Z) + f_cn + dualobj)
    # JuMP.unset_silent(ø)
    @optimise()
    @assert status == JuMP.OPTIMAL " in maximize_φ2_over_Z: $status "
    return JuMP.value.(Z)
end
function eval_prim(u, v, x, Y, Z)
    ø = JumpModel(2)
    @f_prim_code()
    JuMP.@objective(ø, Min, primobj)
    @optimise()
    @assert status == JuMP.OPTIMAL " in eval_prim: $status "
    return JuMP.value(primobj)
end
function eval_dual(u, v, x, Y, Z)
    ø = JumpModel(2)
    @f_dual_code()
    JuMP.@objective(ø, Max, dualobj)
    @optimise()
    @assert status == JuMP.OPTIMAL " in eval_dual: $status "
    return JuMP.value(dualobj)
end
function eval_φ2(x2, β2, Z)
    (u, v, x), Y = x2
    primobj, dualobj = eval_prim(u, v, x, Y, Z), eval_dual(u, v, x, Y, Z)
    @assert primobj + 5e-6 >= dualobj "weak dual $dualobj, $primobj"
    @assert dualobj + 9e-6 >= primobj "strong dual $dualobj, $primobj"
    return φ2 = -ip(β2, Z) + ip(CL, Z) + primobj
end
function gen_cut_for_ℶ2(x2, Z)
    function gen_cut_f_wrt_x2(x2, Z) # this cut is always tight
        ø = JumpModel(0)
        JuMP.@variable(ø, u[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, v[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, x[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, Y[t = 1:T, w = 1:W])
        JuMP.@constraint(ø, cpu[t = 1:T, g = 1:G+1], u[t, g] == x2[1][1][t, g])
        JuMP.@constraint(ø, cpv[t = 1:T, g = 1:G+1], v[t, g] == x2[1][2][t, g])
        JuMP.@constraint(ø, cpx[t = 1:T, g = 1:G+1], x[t, g] == x2[1][3][t, g])
        JuMP.@constraint(ø, cpY[t = 1:T, w = 1:W], Y[t, w] == x2[2][t, w])
        @f_prim_code()
        JuMP.@objective(ø, Min, primobj) # 🍀 only the 2nd half of obj(f(x1, Y, Z))
        @optimise()
        @assert status == JuMP.OPTIMAL "in gen_cut_f_wrt_x2(): $status"
        px1 = JuMP.dual.(cpu), JuMP.dual.(cpv), JuMP.dual.(cpx)
        pY  = JuMP.dual.(cpY)
        px2 = px1, pY
        cn = JuMP.objective_value(ø) - ip(px2, x2)
        cn += ip(CL, Z) # 🍀 an additional const obj term of f(x1, Y, Z)
        return cn, px2
    end
    cn, px2 = gen_cut_f_wrt_x2(x2, Z)
    pβ2 = -Z # 💡 this is fixed, and irrespective of 'β2'
    return cn, px2, pβ2
end
function eval_Δ_at(Δ, x, β) # t1, in termination criterion
    isempty(Δ["f"]) && return Inf
    R2, fV, xV, βV = readSimplicial(Δ)
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G+1], sum(xV[r][i][t, g] * λ[r] for r in 1:R2) == x[i][t, g])
    JuMP.@constraint(ø, [t = 1:T, w = 1:W],            sum(βV[r][t, w]    * λ[r] for r in 1:R2) ==    β[t, w])
    JuMP.@objective(ø, Min, ip(fV, λ))
    @optimise()
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            @optimise()
        end
        status == JuMP.INFEASIBLE && return Inf
        error(" in eval_Δ_at(): $status ")
    end
    return JuMP.objective_value(ø)
end
function eval_φ1ub(x1, β1, yM) # t2: a max-min problem to choose worst Y
    function eval_xby(x1, β1, Y)
        R2, fV, x2V, β2V = readSimplicial(Δ2)
        @assert R2 >= 1
        const_obj = -ip(β1, Y)
        ø = JumpModel(0)
        JuMP.@variable(ø, λ[1:R2] >= 0.)
        JuMP.@constraint(ø, sum(λ) == 1.)
        JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
        JuMP.@constraint(ø, [t = 1:T, l = 1:L],            sum(β2V[r][t, l]       * λ[r] for r in 1:R2) ==    β2[t, l])
        # following 2 lines are minimum infeas. system ########################################################
        JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G+1], sum(x2V[r][1][i][t, g] * λ[r] for r in 1:R2) == x1[i][t, g])
        JuMP.@constraint(ø, [t = 1:T, w = 1:W],            sum(x2V[r][2][t, w]    * λ[r] for r in 1:R2) ==     Y[t, w])
        #######################################################################################################
        JuMP.@objective(ø, Min, ip(MZ, β2) + ip(fV, λ)) # problem \bar{ψ}(x1, Y)
        @optimise()
        if status != JuMP.OPTIMAL
            if status == JuMP.INFEASIBLE_OR_UNBOUNDED
                JuMP.set_attribute(ø, "DualReductions", 0)
                @optimise()
            end
            status == JuMP.INFEASIBLE && return Inf
            error(" in eval_xby(), status = $status ")
        end
        return const_obj + JuMP.objective_value(ø)
    end
    if isempty(Δ2["f"]) # only once
        return (φ1ub, index) = (Inf, 1)
    end
    NY = size(yM, 3)
    fullVec = zeros(NY)
    for i in 1:NY
        φ1x1b1y = eval_xby(x1, β1, yM[:, :, i])
        if φ1x1b1y == Inf
            return (φ1ub, index) = (Inf, i)
        else
            fullVec[i] = φ1x1b1y
        end
    end
    return (φ1ub, index) = findmax(fullVec)
end

masterCnt, psiCnt = [2], [2]
masterIsMature = falses(1)
x1V, β1V, x2V, β2V, ZV = let
    x1V, β1V = [x1], [β1]
    x2V, β2V = [x2], [β2] # 💡 (x1, β1) -> Y -> β2 is essential; 'x2' is NOT essential
    ZV = [Z]
    x1V, β1V, x2V, β2V, ZV
end
tV, termination_flag = [t1], falses(1)
while true
    ST = tV[1]
    if ST == t1
        vldt, lb, x1, β1, cost1plus2, _ = master(masterIsMature, masterCnt, ℶ1)
        let
            φ1ub = eval_Δ_at(Δ1, x1, β1)
            ub = cost1plus2 + φ1ub # 🍀 ub(Δ1) is a valid upper bound of υ_MSDRO
            gap = abs(ub - lb) / max( abs(lb), abs(ub) )
            @info " t1: masterCnt[$(masterCnt[1])] ($vldt)lb = $lb | $ub = ub, gap = $gap"
            if gap < 0.0001
                @info " 😊 gap < 0.01%, thus terminate "
                termination_flag[1] = true
            end
        end
        x1V[1], β1V[1] = x1, β1
        tV[1] = t2f
    elseif ST == t2f
            x1, β1 = x1V[1], β1V[1]
            _, index = eval_φ1ub(x1, β1, yM)
        x2, β2 = psi(psiCnt, x1, yM[:, :, index]) # 2️⃣ only a trial (x2, β2) is needed here
        x2V[1], β2V[1] = x2, β2
        tV[1] = t3
    elseif ST == t3
        x2, β2 = x2V[1], β2V[1]
        ZV[1] = Z = maximize_φ2_over_Z(x2, β2) # 🌸 the 🍾 bottleneck
        termination_flag[1] && break # 🌿🌿🌿🌿
        φ2 = eval_φ2(x2, β2, Z) # 🌸 strong duality confirmation inside
        cn, px2, pβ2 = gen_cut_for_ℶ2(x2, Z)
        pushSimplicial(Δ2, φ2, x2, β2) # ✅ Δ2 is a precise evaluation of φ2(x2, β2)
        pushCut(ℶ2, x2, Z, cn, px2, pβ2) # ✅ ℶ2 is a valid underestimator of    φ2(x2, β2)
        tV[1] = t2b
    elseif ST == t2b
            x1, β1 = x1V[1], β1V[1]
            φ1ub, index = eval_φ1ub(x1, β1, yM)
        ret = gen_cut_for_ℶ1(x1, yM[:, :, index])
        if length(ret) == 1
            tV[1] = t2f
        else
            pushCut(ℶ1, x1, index, ret...) # ✅ ℶ1(ℶ2) is a valid underestimator of φ1(x1, β1)
            φ1ub < Inf && pushSimplicial(Δ1, φ1ub, x1, β1) # ⚠️ conditional update ✅ Δ1(Δ2) is a valid overestimator of φ1(x1, β1)
            tV[1] = t1
        end
    end
end



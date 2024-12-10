import LinearAlgebra
import Distributions
import Statistics
import MosekTools
import Polyhedra
import Random
import Gurobi
import JuMP
using Logging
GRB_ENV = Gurobi.Env()
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
macro optimise() return esc(:((_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø)))) end
is_finite(r) = -Inf < r < Inf
rgapsimple(l, u) = abs(u - l) / max(abs(l), abs(u))
function rgap(l, u)
    is_finite(l) && is_finite(u) && return rgapsimple(l, u)
    return Inf
end
ip(x, y)             = LinearAlgebra.dot(x, y)
norm1(x)             = LinearAlgebra.norm(x, 1)
LT(P, c, x)          = P \ (x .+ c) # from STD to Desired 💡 x can be both Vector and Matrix
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
    JuMP.unset_silent(ø)
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
    JuMP.@constraint(ø, [i = 1:N], y[i] == ip(yM[i, :], c)) # ith entry
    @optimise()
    status == JuMP.OPTIMAL && return JuMP.value.(c)
    error(" in is_in: status = $status")
end
ds(ref, y)           = norm1(ref .- y)
ds(ref, y::Matrix)   = [ds(ref, v) for v in eachcol(y)]
ds(y::Matrix, ::Any) = error(" malfunction ")
function pts2N(N)
    # function ptsD0(N) return [zeros(N) one(ones(N, N))] end
    a, b = LinearAlgebra.Diagonal(ones(N)), LinearAlgebra.Diagonal(-ones(N))
    [(c % 2 == 0) ? b[i, div(c, 2)] : a[i, div(c+1, 2)] for i in 1:N, c in 1:2N]
end
vov2m(vec) = [vec[c][r] for r in eachindex(vec[1]), c in eachindex(vec)]
m2vov(mat) = [Vector(c) for c in eachcol(mat)]

T, G, W, L, B = 4, 2, 2, 3, 11 # 🌸 G+1 is the size of (u, v, x)

begin # build (MY, yM)
    N = T * W # cardinality of Y
    Random.seed!(23)
    ℙ = let
        (ℙ = Distributions.Uniform(0.05, 0.50); lb = rand(ℙ, N))
        (ℙ = Distributions.Uniform(3.30, 3.80); ub = rand(ℙ, N))
        Distributions.Product(Distributions.Uniform.(lb, ub)) # if you use other underlying distributions, they maybe inefficient to sample, i.e., many many time before you can sample an enclosing y
    end
    MY = Statistics.mean(ℙ)
    (a = [[2 for _ in 1:N]; N]; y_peripheral = NaN * ones(a...));
    (a = [2 for _ in 1:N];      dist_y_peri = -Inf * ones(a...));
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
    @assert -Inf < minimum(dist_y_peri) "you want more batchs"
    y_peripheral = Matrix(transpose(reshape(y_peripheral, (:, N))));
    P, c = get_Pc(y_peripheral)
    @assert all(LinearAlgebra.eigen(P).values .> 0) " invalid transform P "
    yM_ref = LT(P, c, pts2N(N)); # 💡 extreme points
    @assert all( sum(yM_ref .< 0; dims = 2) .== ones(N) ) " abnormal yM_ref "
    yM_eft = yM_ref[:, [(c % 2 == 1) for c in 1:2N]] # valid points
    yM_ineft = yM_ref[:, [(c % 2 == 0) for c in 1:2N]] # for invalid negatives
    y = rand(ℙ);
    min_dist = ds(y, yM_ineft);
    yM_delegate = vov2m([y for _ in eachindex(min_dist)]);
    for b in 1:4000
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
    println("testing is_in_interior...")
    @assert Polyhedra.ininterior(MY, Polyhedra.polyhedron(Polyhedra.vrep(m2vov(yM)))) # 🌸 test if possible
    println("testing finished.")
    # Polyhedra.volume(poly)
    # h = Polyhedra.hrep(poly)
    Yvec2mat(vec) = reshape(vec, (T, W))
    MY, yM = Yvec2mat(MY), cat([Matrix(Yvec2mat(v)) for v in eachcol(yM)]..., dims = 3)
end
begin # build (MZ, zM)
    N = T * L # cardinality of Z
    Random.seed!(89)
    ℙ = let
        (ℙ = Distributions.Uniform(0.05, 0.50); lb = rand(ℙ, N))
        (ℙ = Distributions.Uniform(3.30, 3.80); ub = rand(ℙ, N))
        Distributions.Product(Distributions.Uniform.(lb, ub)) # if you use other underlying distributions, they maybe inefficient to sample, i.e., many many time before you can sample an enclosing y
    end
    MZ = Statistics.mean(ℙ)
    (a = [[2 for _ in 1:N]; N]; z_peripheral = NaN * ones(a...));
    (a = [2 for _ in 1:N];      dist_z_peri = -Inf * ones(a...));
    for b in 1:150
        for s in 1:2^N
            y = rand(ℙ)
            Δ = y .- MZ
            a, d = (Δ .> 0) .+ 1, norm1(Δ)
            if d > dist_z_peri[a...]
                z_peripheral[a..., :] .= y
                dist_z_peri[a...] = d
            end
        end
        mi, ma = minimum(dist_z_peri), maximum(dist_z_peri)
        print("b = $b, min_dist = $mi, MAX_dist = $ma  \r")
    end
    @assert -Inf < minimum(dist_z_peri) "you want more batchs"
    z_peripheral = Matrix(transpose(reshape(z_peripheral, (:, N))));
    P, c = get_Pc(z_peripheral)
    @assert all(LinearAlgebra.eigen(P).values .> 0) " invalid transform P "
    zM_ref = LT(P, c, pts2N(N)); # 💡 extreme points
    @assert all( sum(zM_ref .< 0; dims = 2) .== ones(N) ) " abnormal zM_ref "
    zM_eft = zM_ref[:, [(c % 2 == 1) for c in 1:2N]] # valid points
    zM_ineft = zM_ref[:, [(c % 2 == 0) for c in 1:2N]] # for invalid negatives
    z = rand(ℙ);
    min_dist = ds(z, zM_ineft);
    zM_delegate = vov2m([z for _ in eachindex(min_dist)]);
    for b in 1:4000
        for s in 1:5000
            z = rand(ℙ)
            _, c = findmin(z)
            d = ds(z, zM_ineft[:, c])
            if d < min_dist[c]
                zM_delegate[:, c] .= z
                min_dist[c] = d
            end
        end
        print("b = $b, sumdist = $(sum(min_dist)) \r")
    end
    zM_center = let
        p = [zM_ineft[i, i] / (zM_ineft[i, i] - zM_delegate[i, i]) for i in 1:N]
        cm = vov2m([p[i] * zM_delegate[:, i] + (1 - p[i]) * zM_ineft[:, i] for i in 1:N])
        cm[one(ones(Bool, (N, N)))] .= 0 # to round small floating point errors
        cm
    end
    zM = [zM_eft zM_center] # a trial
    zM = let
        v = Polyhedra.vrep(m2vov(zM));
        v_rmed = Polyhedra.removevredundancy(v, Gurobi.Optimizer) # 💡 this procedure is efficient
        vov2m(v_rmed.points)
    end
    is_in(MZ, zM) # 🌸 you must test
    println("testing is_in_interior...")
    @assert Polyhedra.ininterior(MZ, Polyhedra.polyhedron(Polyhedra.vrep(m2vov(zM)))) # 🌸 test if possible
    println("testing finished.")
    # Polyhedra.volume(poly)
    # h = Polyhedra.hrep(poly)
    Zvec2mat(vec) = reshape(vec, (T, L))
    MZ, zM = Zvec2mat(MZ), cat([Matrix(Zvec2mat(v)) for v in eachcol(zM)]..., dims = 3)
end



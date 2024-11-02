import LinearAlgebra
import Distributions
import Statistics
import Random
import JuMP
import Gurobi
import Polyhedra
import MosekTools

# A heuristic procedure to generate a support for a random variable Y in ℝᴺ
# (If domain-specific knowledge is available, e.g. certain hazardous scenarios, Just merge them)
# This support comprises a tractable finite num of Ext Points that can be employed in the vertex enum procedure of a convex maximum problem
# We can use 'Polyhedra.removevredundancy(v, Gurobi.Optimizer)' to ensure the irredundancy of that set of points
# The support should contain 𝔼[Y] in its interior (a fortiori, the support should be solid in ℝᴺ)
# 2/11/24

GRB_ENV = Gurobi.Env()
column(x::JuMP.VariableRef) = Gurobi.c_column(JuMP.backend(JuMP.owner_model(x)), JuMP.index(x))
function ip(x, y) return LinearAlgebra.dot(x, y) end
function ds(x, y::Vector) return LinearAlgebra.norm(x .- y, 1) end # 1-norm distance
function ds(x, y::Matrix) return [ds(x, y[:, c]) for c in 1:size(y, 2)] end
function ptsD0(N) return [zeros(N) one(ones(N, N))] end
function pts2N(N)
    a, b = LinearAlgebra.Diagonal(ones(N)), LinearAlgebra.Diagonal(-ones(N))
    [(c % 2 == 0) ? b[i, div(c, 2)] : a[i, div(c+1, 2)] for i in 1:N, c in 1:2N]
end
function vov2m(vec) return [vec[c][r] for r in eachindex(vec[1]), c in eachindex(vec)] end
function m2vov(mat) return [Vector(c) for c in eachcol(mat)] end
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        # JuMP.set_attribute(ø, "QCPDual", 1)
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
function LT(P, c, x) return P \ (x .+ c) end # from STD to Desired 💡 x can be both Vector and Matrix
function get_Pc(yM::Matrix)
    N, S = size(yM)
    ø = JumpModel(1)
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
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" $status ")
    else
        P, c = JuMP.value.(P), JuMP.value.(c)
        return P, c
    end
end
function is_in(y, yM)
    N, S = size(yM)
    ø = JumpModel(0)
    JuMP.@variable(ø, c[1:S] >= 0.)
    JuMP.@constraint(ø, sum(c) == 1.)
    JuMP.@constraint(ø, [i = 1:N], y[i] == ip(yM[i, :], c))
    JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status == JuMP.OPTIMAL
        return JuMP.value.(c)
    end
end

N = 16
ℙ = let
    lb = [0.412, 0.392, 0.158, 0.031, 0.155, 0.425, 0.422, 0.058, 0.271, 0.046, 0.385, 0.381, 0.162, 0.087, 0.474, 0.19]
    ub = [4.01588453677511, 4.10481644003798, 3.994238499688115, 4.170218035327311, 4.16498217856943, 4.063888305117489, 4.008386349019764, 4.100947227321259, 3.8529151075562083, 3.8074367833089164, 3.804480877140664, 3.93061964472342, 3.7961731703688204, 3.864105964009756, 3.860208040329274, 3.829591856493192]
    Distributions.Product(Distributions.Uniform.(lb, ub)) # if you use other underlying distributions, they maybe inefficient to sample, i.e., many many time before you can sample an enclosing y
end;
MY = Statistics.mean(ℙ)
Random.seed!(23)

y_peripheral, dist_y_peri = let # collect a set of points that encloses MY in all directions
    a = [[2 for _ in 1:N]; N];
    y_peripheral = NaN * ones(a...);
    a = [2 for _ in 1:N];
    dist_y_peri = -Inf * ones(a...);
    y_peripheral, dist_y_peri
end;
for b in 1:100
    for s in 1:2^N
        y = rand(ℙ)
        Δ = y .- MY
        a, d = (Δ .> 0) .+ 1, LinearAlgebra.norm(Δ, 1) 
        if d > dist_y_peri[a...]
            y_peripheral[a..., :] .= y
            dist_y_peri[a...] = d
        end
    end
    mi, ma = minimum(dist_y_peri), maximum(dist_y_peri)
    print("b = $b, min_dist = $mi, MAX_dist = $ma  \r")
end

y_peripheral = Matrix(transpose(reshape(y_peripheral, (:, N))));
P, c = get_Pc(y_peripheral) # 📚 Mosek's SDP
@assert all(LinearAlgebra.eigen(P).values .> 0) " invalid transform P "
yM_ref = LT(P, c, pts2N(N)); # 💡 extreme points
@assert all( sum(yM_ref .< 0; dims = 2) .== ones(N) ) " abnormal yM_ref "
yM_eft = yM_ref[:, [(c % 2 == 1) for c in 1:2N]] # valid points

# # 1️⃣ the first lazy method, we have N+1 extreme points
# yM_eft = [zeros(N) yM_eft]
# yM = yM_eft
# v = Polyhedra.vrep(m2vov(yM));
# poly = Polyhedra.polyhedron(v);
# Polyhedra.ininterior(MY, poly) # yes
# Polyhedra.volume(poly) # 1.1019052393581734e9
# h = Polyhedra.hrep(poly) # tractable

# 2️⃣ the second method, we have 2N extreme points 
yM_ineft = yM_ref[:, [(c % 2 == 0) for c in 1:2N]] # for invalid negatives
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
poly = Polyhedra.polyhedron(v);
Polyhedra.ininterior(MY, poly) # this is true
Polyhedra.volume(poly) # if possible
h = Polyhedra.hrep(poly) # if possible

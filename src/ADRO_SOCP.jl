import JuMP
import MosekTools
import Optim
import Random
import Distributions
import LinearAlgebra
import Polyhedra
using Logging

# newsvendor fixed recourse ADRO problem (SOCP)
# We have 3 methods that is descending in accuracy
# SDP (full moment constr) -> DRO2019 (only J q's, K-piece convex obj) -> ELDR (introduce y(z, u))
# 27/8/24

function JumpModel(i)
    if i == 0 # the most frequently used
        m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    elseif i == 1 # generic convex conic program
        m = JuMP.Model(MosekTools.Optimizer)
    elseif i == 2 # if you need Gurobi callback
        m = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(m) # use JuMP.unset_silent(m) to debug when it is needed
    return m
end
function ip(x, y) LinearAlgebra.dot(x, y) end
function dirvec(x) x / LinearAlgebra.norm(x) end
function vecofvec2mat(vec) return [vec[c][r] for r in eachindex(vec[1]), c in eachindex(vec)] end
function masspoints(ξ, η) return [ξ[r, c] / η[c] for r in eachindex(eachrow(ξ)), c in eachindex(η)] end
function genExtMat(B, d)
    m = JuMP.Model()
    JuMP.@variable(m, p[eachindex(eachrow(B))] >= 0.)
    JuMP.@constraint(m, [i in eachindex(d)], ip(B[:, i], p) == d[i])
    vo = Polyhedra.vrep(Polyhedra.polyhedron(m))
    @assert isempty(Polyhedra.rays(vo))
    @assert isempty(Polyhedra.lines(vo))
    vecofvec2mat(collect(Polyhedra.points(vo)))
end

global_logger(ConsoleLogger(Info))
Random.seed!(86)

if true # data gen
    function h(q) LinearAlgebra.dot(q, Σ, q) end
    ℙ = Distributions.Uniform(-5., 5.) # for q0
    VIO_BRINK = -1e-6
    I = 3 # len( y / x / d / Z )
    M = 2 * I # num of rows in A, B, or b
    μ = rand(Distributions.Uniform(5., 100.), I)
    σ = [rand(Distributions.Uniform(μ[i], 5. * μ[i])) for i in 1:I]
    Σ = let
        Δ = .5
        upsilon = rand(Distributions.Uniform(Δ, 1.), I, I)
        V = upsilon' * upsilon
        w = [1/sqrt(V[i, i]) for i in 1:I]
        t = LinearAlgebra.Diagonal(w)
        t * V * t
    end
    c, c1, d, B = let # c: producing price; c1 is the 1-stage cost; d and B are along with A(i) and b(i)
        v = rand(Distributions.Uniform(5., 10.), I)
        g, b = v/10, v/4
        c = [rand(Distributions.Uniform( .5 * v[i], .6 * v[i] )) for i in 1:I]
        c, c - v - b, v + b - g, [one(ones(I,I)); one(ones(I,I))]
    end
    SUV, XUV = μ + 3 * σ, ip(μ + σ, c)
    function A(i)
        if i in 1:I
            return zeros(M, I)
        elseif i == 0
            return [-one(ones(I,I)); zeros(I,I)]
        else
            error()
        end
    end
    function b(i)
        tmp = zeros(M)
        if i == 0
            return tmp
        elseif i in 1:I
            tmp[i] = -1.
            return tmp
        else
            error()
        end
    end
    if true # utilize the K-piece convex PWL objective
        extPoints, K = genExtMat(B,d), 2^I
        @assert size(extPoints)[2] == K
        function b(k, x) -LinearAlgebra.dot(extPoints[:, k], A(0), x) end # this b_k(x) is in the objective, as the k-th piece
        function a(k, x) return [ip(extPoints[:, k], b(i)) for i in 1:I] end # ✏️ arg `x` is unused
    end
    if true # global containers
        qs = one(ones(I, I))[:, 1:2]
        x_incumbent = NaN * ones(I) # the global xt (the incumbent)
    end
end

function DRO19(qs)
    J = size(qs)[2]
    ι = JumpModel(1)
    JuMP.@variable(ι, x[1:I] >= 0.)
    JuMP.@constraint(ι, sum(x) <= XUV)
    if true # sub block of constrs and variables
        JuMP.@variable(ι, α)
        JuMP.@variable(ι, β[1:I])
        JuMP.@variable(ι, γ[1:J] >= 0.)
        JuMP.@variable(ι, r[1:K, 1:I])
        JuMP.@variable(ι, t[1:K])
        JuMP.@constraint(ι, η[k = 1:K], α - b(k, x) >= t[k])
        JuMP.@constraint(ι, ξ[i = 1:I, k = 1:K], β[i] == a(k, NaN)[i] + r[k, i])
        if true # dual cone of K(\bar{W})
            JuMP.@variable(ι, w[1:K, 1:I] >= 0.)
            JuMP.@variable(ι, v1[1:K, 1:J])
            JuMP.@variable(ι, v2[1:K, 1:J])
            JuMP.@variable(ι, v3[1:K, 1:J])
            JuMP.@constraint(ι, [k = 1:K, j = 1:J], [v1[k, j], v2[k, j], v3[k, j]] in JuMP.SecondOrderCone())
            JuMP.@constraint(ι, [k = 1:K, j = 1:J], γ[j] == v1[k, j] + v2[k, j])
            JuMP.@constraint(ι, [k = 1:K, i = 1:I], w[k, i] + r[k, i] >= 2 * sum(v3[k, j] * qs[i, j] for j in 1:J))
            JuMP.@constraint(ι, [k = 1:K], t[k] >= ip(w[k, :], SUV) + sum(v1[k, :]) - sum(v2[k, :]) - 2 * sum(v3[k, j] * ip(qs[:, j], μ) for j in 1:J))
        end
    end
    obj2 = α + ip(μ, β) + sum(γ[j] * h(qs[:, j]) for j in 1:J)
    obj1 = ip(c1, x)
    JuMP.@objective(ι, Min, obj1 + obj2)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    JuMP.objective_value(ι), JuMP.value.([obj1, obj2]), JuMP.value.(x)
end

function DRO19_subproblem(x, qs)
    J = size(qs)[2]
    ι = JumpModel(1)
    if true # sub block of constrs and variables
        JuMP.@variable(ι, α)
        JuMP.@variable(ι, β[1:I])
        JuMP.@variable(ι, γ[1:J] >= 0.)
        JuMP.@variable(ι, r[1:K, 1:I])
        JuMP.@variable(ι, t[1:K])
        JuMP.@constraint(ι, η[k = 1:K], α - b(k, x) >= t[k])
        JuMP.@constraint(ι, ξ[i = 1:I, k = 1:K], β[i] == a(k, NaN)[i] + r[k, i])
        if true # dual cone of K(\bar{W})
            JuMP.@variable(ι, w[1:K, 1:I] >= 0.)
            JuMP.@variable(ι, v1[1:K, 1:J])
            JuMP.@variable(ι, v2[1:K, 1:J])
            JuMP.@variable(ι, v3[1:K, 1:J])
            JuMP.@constraint(ι, [k = 1:K, j = 1:J], [v1[k, j], v2[k, j], v3[k, j]] in JuMP.SecondOrderCone())
            JuMP.@constraint(ι, [k = 1:K, j = 1:J], γ[j] == v1[k, j] + v2[k, j])
            JuMP.@constraint(ι, [k = 1:K, i = 1:I], w[k, i] + r[k, i] >= 2 * sum(v3[k, j] * qs[i, j] for j in 1:J))
            JuMP.@constraint(ι, [k = 1:K], t[k] >= ip(w[k, :], SUV) + sum(v1[k, :]) - sum(v2[k, :]) - 2 * sum(v3[k, j] * ip(qs[:, j], μ) for j in 1:J))
        end
    end
    obj2 = α + ip(μ, β) + sum(γ[j] * h(qs[:, j]) for j in 1:J)
    JuMP.@objective(ι, Min, obj2)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    JuMP.value(obj2), JuMP.dual.(η), JuMP.dual.(ξ)
end

function ELDR(qs)
    L = I
    J = size(qs)[2]
    ι = JumpModel(1)
    JuMP.@variable(ι, x[1:I] >= 0.)
    JuMP.@constraint(ι, sum(x) <= XUV)
    if true # sub block of constrs and variables
        JuMP.@variable(ι, y0[1:L])
        JuMP.@variable(ι, y1[1:I, 1:L])
        JuMP.@variable(ι, y2[1:J, 1:L])
        JuMP.@variable(ι, t[1:M])
        JuMP.@variable(ι, r[1:M, 1:I])
        JuMP.@expression(ι, s[m = 1:M, j = 1:J], ip(B[m, :], y2[j, :])) # we don't need to know ζ, thus use expression instead
        JuMP.@constraint(ι, ξ[i = 1:I, m = 1:M], ip(B[m, :], y1[i, :]) - b(i)[m] == r[m, i]) # A(i) = zeros
        JuMP.@constraint(ι, η[m = 1:M], ip(A(0)[m, :], x) + ip(B[m, :], y0) >= t[m]) # b(0) = zeros
        if true # dual cone of K(\bar W)
            JuMP.@variable(ι, w[1:M, 1:I] >= 0.)
            JuMP.@variable(ι, v1[1:M, 1:J])
            JuMP.@variable(ι, v2[1:M, 1:J])
            JuMP.@variable(ι, v3[1:M, 1:J])
            JuMP.@constraint(ι, [m = 1:M, j = 1:J], [v1[m, j], v2[m, j], v3[m, j]] in JuMP.SecondOrderCone())
            JuMP.@constraint(ι, [m = 1:M, j = 1:J], s[m, j] == v1[m, j] + v2[m, j])
            JuMP.@constraint(ι, [m = 1:M, i = 1:I], w[m, i] + r[m, i] >= 2 * sum(v3[m, j] * qs[i, j] for j in 1:J))
            JuMP.@constraint(ι, [m = 1:M], t[m] >= ip(w[m, :], SUV) + sum(v1[m, :]) - sum(v2[m, :]) - 2 * sum(v3[m, j] * ip(qs[:, j], μ) for j in 1:J))
        end
    end
    obj2 = ip(d, y0) + sum(μ[i] * ip(d, y1[i, :]) for i in 1:I) + sum(h(qs[:, j]) * ip(d, y2[j, :]) for j in 1:J)
    obj1 = ip(c1, x)
    JuMP.@objective(ι, Min, obj1 + obj2)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    JuMP.objective_value(ι), JuMP.value.([obj1, obj2]), JuMP.value.(x)
end

function ELDR_subproblem(x, qs)
    L = I
    J = size(qs)[2]
    ι = JumpModel(1)
    if true # sub block of constrs and variables
        JuMP.@variable(ι, y0[1:L])
        JuMP.@variable(ι, y1[1:I, 1:L])
        JuMP.@variable(ι, y2[1:J, 1:L])
        JuMP.@variable(ι, t[1:M])
        JuMP.@variable(ι, r[1:M, 1:I])
        JuMP.@expression(ι, s[m = 1:M, j = 1:J], ip(B[m, :], y2[j, :])) # we don't need to know ζ, thus use expression instead
        JuMP.@constraint(ι, ξ[i = 1:I, m = 1:M], ip(B[m, :], y1[i, :]) - b(i)[m] == r[m, i]) # A(i) = zeros
        JuMP.@constraint(ι, η[m = 1:M], ip(A(0)[m, :], x) + ip(B[m, :], y0) >= t[m]) # b(0) = zeros
        if true # dual cone of K(\bar W)
            JuMP.@variable(ι, w[1:M, 1:I] >= 0.)
            JuMP.@variable(ι, v1[1:M, 1:J])
            JuMP.@variable(ι, v2[1:M, 1:J])
            JuMP.@variable(ι, v3[1:M, 1:J])
            JuMP.@constraint(ι, [m = 1:M, j = 1:J], [v1[m, j], v2[m, j], v3[m, j]] in JuMP.SecondOrderCone())
            JuMP.@constraint(ι, [m = 1:M, j = 1:J], s[m, j] == v1[m, j] + v2[m, j])
            JuMP.@constraint(ι, [m = 1:M, i = 1:I], w[m, i] + r[m, i] >= 2 * sum(v3[m, j] * qs[i, j] for j in 1:J))
            JuMP.@constraint(ι, [m = 1:M], t[m] >= ip(w[m, :], SUV) + sum(v1[m, :]) - sum(v2[m, :]) - 2 * sum(v3[m, j] * ip(qs[:, j], μ) for j in 1:J))
        end
    end
    JuMP.@objective(ι, Min, ip(d, y0) + sum(μ[i] * ip(d, y1[i, :]) for i in 1:I) + sum(h(qs[:, j]) * ip(d, y2[j, :]) for j in 1:J))
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    JuMP.objective_value(ι), JuMP.dual.(η), JuMP.dual.(ξ) # ⚠️⚠️⚠️ you need to test manually the SIGN of the dual variable
end

if true # functions defined for separation problem
    function θ(p, η, m, q) # a public function, p and η are data, m is index, q is variable
        z, w = p[:, m], η[m] # pick the m-th column
        w * ip(q, z - μ)^2
    end
    if true # ELDR method separation pertinent
        function Mset(l) return [l, l+I] end
        function Φ(p, η, q, l) return Phi_first(q, l) - Phi_second(p, η, q, l) end
        function Phi_first(q, l) return d[l] * h(q) end
        function Phi_second(p, η, q, l) return sum(θ(p, η, m, q) for m in Mset(l)) end
    end
    if true # DRO19 method separation pertinent
        function Φ(p, η, q) return Phi_first(q) - Phi_second(p, η, q) end
        function Phi_first(q) return h(q) end
        function Phi_second(p, η, q) return sum(θ(p, η, k, q) for k in eachindex(η)) end
    end
end

useDRO19 = false # ✏️ switch between 2 modes
for mainIte in 1:typemax(Int) # the GIP algorithm
    if useDRO19
        ub, costvec, x = DRO19(qs)
    else
        ub, costvec, x = ELDR(qs)
    end
    xtmp = round.(x; digits = 4)
    @info " mainIte=$mainIte, x=$xtmp, cost=$costvec, ub=$ub"
    x_incumbent .= x
    if useDRO19
        _, η, ξ = DRO19_subproblem(x, qs)
    else
        _, η, ξ = ELDR_subproblem(x, qs)
    end
    if true # data process
        tmp = minimum(η)
        @assert tmp > 0 " η has a <= 0 entry "
        tmp < 5e-4 && @warn " η has an entry that is tiny, $(tmp) "
        p = masspoints(ξ, η) # magnitude of each col of p should resembles μ
    end
    find_already, vio_q = [false], Inf * ones(I)
    if useDRO19
        gen_res_at = q0 -> Optim.optimize(q -> Φ(p, η, q), q0, Optim.NewtonTrustRegion(); autodiff = :forward)
        for ite in 1:100
            q0 = rand(ℙ, I)
            # Phi_first(q0, l)
            # Phi_second(q0, l)
            ℝ = gen_res_at(q0)
            qt, vt = Optim.minimizer(ℝ), Optim.minimum(ℝ)
            dir = dirvec(qt)
            val = Φ(p, η, dir)
            if val < VIO_BRINK
                vio_q .= dir
                find_already[1] = true
                break
            end
        end
    else # the ELDR method
        for l in 1:I # I = L, this loop is for ELDR approach
            gen_res_at = q0 -> Optim.optimize(q -> Φ(p, η, q, l), q0, Optim.NewtonTrustRegion(); autodiff = :forward)
            for ite in 1:100
                q0 = rand(ℙ, I)
                # Phi_first(q0, l)
                # Phi_second(q0, l)
                ℝ = gen_res_at(q0)
                qt, vt = Optim.minimizer(ℝ), Optim.minimum(ℝ)
                dir = dirvec(qt)
                val = Φ(p, η, dir, l)
                if val < VIO_BRINK
                    vio_q .= dir
                    find_already[1] = true
                    break
                end
            end
            if find_already[1] == true
                break
            end
        end
    end
    if find_already[1] == false
        @info " 🫠 [End-of-GIP] we can't find a vio_q at this stage of GIP algorithm, thus return "
        return x_incumbent
    else
        qs = [qs vio_q]
        qtmp = round.(vio_q; digits = 4)
        @info "add vio_q = $qtmp"
    end
end

# -------------------------- SDP is an exact benchmark --------------------------
function SDP()
    ι = JumpModel(1)
    JuMP.@variable(ι, x[1:I] >= 0.)
    JuMP.@constraint(ι, sum(x) <= XUV)
    if true # sub block of constrs and variables
        JuMP.@variable(ι, α)
        JuMP.@variable(ι, β[1:I])
        JuMP.@variable(ι, δ[1:K])
        JuMP.@variable(ι, χ[1:K, 1:I])
        JuMP.@variable(ι, Γ[1:I, 1:I])
        JuMP.@constraint(ι, [k = 1:K], [δ[k] χ[k, :]' ; χ[k, :] Γ] in JuMP.PSDCone()) # b(0) = zeros
        JuMP.@variable(ι, t[1:K])
        JuMP.@variable(ι, r[1:K, 1:I])
        JuMP.@variable(ι, w[1:K, 1:I] >= 0.) # ancillary 
        JuMP.@constraint(ι, [k = 1:K, i = 1:I], r[k, i] + w[k, i] >= 0.)
        JuMP.@constraint(ι, [k = 1:K], t[k] >= ip(w[k, :], SUV))
        JuMP.@constraint(ι, [k = 1:K], α - b(k, x) + 2 * ip(μ, χ[k, :]) >= δ[k] + t[k])
        JuMP.@constraint(ι, [k = 1:K, i = 1:I], β[i] == a(k, NaN)[i] + r[k, i] + 2 * χ[k, i])
    end
    obj2 = α + ip(μ, β) + ip(Σ, Γ)
    obj1 = ip(c1, x)
    JuMP.@objective(ι, Min, obj1 + obj2)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    JuMP.objective_value(ι), JuMP.value.([obj1, obj2]), JuMP.value.(x)
end
val, valdetail, xt = SDP() 



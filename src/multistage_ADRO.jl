import JuMP
import MosekTools
import Optim
import Random
import Distributions
import LinearAlgebra
import Polyhedra
using Logging

# Multistage-ADRO with infinitely many moment constraints, the inventory case study
# use JuMP.fix to add NA constraints
# 30/8/24

# result:
# [ Info: mainIte = 231, ub = 143.95666806105396, x1 = 259.99999998563817
# [ Info: we can't find any more violating q's

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
function eye(n) return LinearAlgebra.Diagonal(ones(n)) end
function unitVec(n, c) return eye(n)[:, c] end
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
function qsize(qs)
    card_q_at = div.(length.(qs), T)
    part_sum_qs = [sum(card_q_at[begin:i]) for i in eachindex(card_q_at)]
    J = part_sum_qs[end]
    return J, card_q_at, part_sum_qs
end
function j2q(j, partsumv, qs)
    v = partsumv .- j
    t = findfirst(e -> e>=0, v)
    n = (t >= 2 ? -v[t-1] : j)
    qs[t][:, n]
end

global_logger(ConsoleLogger(Info))
Random.seed!(86)

if true # data gen
    T = I = 5
    L = 2 * T - 1 # [x2, ..., xT; y1, ..., yT] |> len
    # x1 is the sole chief decision
    # dependency of Adjustable decisions
    # x2(z1), x3(z1, z2), xT(z1, ..., zT-1)
    # y1(z1), y2(z1, z2), yT(z1, ..., zT)
    xUB = 260. # 0 <= x <= xUB
    zBND = 40. # symmetric bound
    function h(q) return zBND^2/3 * ip(q, q) end # i.e. q'Σq
    function phi(s)
        tmp = zBND^4
        return sum(tmp/18 * s[t]^4 + 6 * sum( tmp/9 * (s[r] * s[t])^2 for r in t+1:T) for t in 1:T)
    end
    ccost = 0.1
    hcost, bh_ratio = 0.02, 10
    bcost = let
        b = hcost * bh_ratio
        b = b * ones(T)
        b[end] = 10 * b[end-1]
        b
    end
    ℙ = Distributions.Uniform(-5., 5.) # for q0
    VIO_BRINK = -1e-6
    Num_q0_trial = 50
    function genB()
        function genB1()
            C = T-1
            E = eye(C)
            R = 2 * C
            f = (r, c) -> (isodd(r) ? E[div(r+1,2), c] : -E[div(r,2), c])
            [f(r, c) for r in 1:R, c in 1:C]
        end
        function genB3()
            C = T
            E = eye(C)
            R = 2 * C
            f = (r, c) -> (isodd(r) ? E[div(r+1,2), c] : E[div(r,2), c])
            [f(r, c) for r in 1:R, c in 1:C]
        end
        function genB2()
            function genB2b()
                R = T
                C = T - 1
                f = (r, c) -> ((c >= r) ? 0 : bcost[r])
                [f(r, c) for r in 1:R, c in 1:C]
            end
            function genB2h()
                R = T
                C = T - 1
                f = (r, c) -> ((c >= r) ? 0 : -hcost)
                [f(r, c) for r in 1:R, c in 1:C]
            end
            Bb = genB2b()
            Bh = genB2h()
            R = 2 * T
            C = T - 1
            f = (r, c) -> (isodd(r) ? Bb[div(r+1,2), c] : Bh[div(r,2), c])
            [f(r, c) for r in 1:R, c in 1:C]
        end
        Float64[genB1() zeros(2 * (T-1), T); genB2() genB3()]
    end
    d, B, A0 = let # 🏹 These are all non-rand parameters, in which A has only 1 column (i.e. x1)
        a_row_align_B2 = [(isodd(i) ? bcost[div(i+1,2)] : -hcost) for i in 1:2*T]
        a_row_aligh_B1 = zeros(2*(T-1))
        [ccost * ones(T-1); ones(T)], genB(), [a_row_aligh_B1; a_row_align_B2]
    end # 🏹 there are T extreme rays about B, whose form are specified in the paper
    # 📚 Ai = 0 for i > 0
    M = size(B)[1]
    Dslope, Dconst = 3/4, 200.
    if true # generate b0, b1, ..., bI, which are random vectors, in contrast to A0
        function D(t, z::Vector) # demand process controled by z[t]
            slope = Dslope * ones(t)
            slope[end] = 1.
            return ip(z[1:t], slope) + Dconst
        end
        function b(z::Vector{Float64})
            Dvec = [D(t, z) for t in 1:T]
            Dpartsum = [sum(Dvec[1:t]) for t in 1:T]
            b_row_align_B2 = [(isodd(i) ? bcost[div(i+1,2)] * Dpartsum[div(i+1,2)] : -hcost * Dpartsum[div(i,2)]) for i in 1:2*T]
            b_row_align_B1 = [(isodd(i) ? 0 : -xUB ) for i in 1:2*(T-1)] # this part is non-rand
            return [b_row_align_B1; b_row_align_B2]
        end
        function linb(z) return b(z) .- b(0) end
        function b(i::Int)
            vec = zeros(M)
            if i == 0
                vec .= b(zeros(I))
            elseif i in 1:I
                vec .= linb(unitVec(I, i))
            else
                error("index error in b(::Int)")
            end
            round.(vec; digits = 6)
        end
    end
    qs = let # we don't need to embellish qs[1]
        qS = one(ones(I, I)) # T = I
        qS = [qS[:, t:t] for t in 1:T]
        qS[1][1, 1] *= -1
        qS[1] = [qS[1] unitVec(T, 1)]
        qS
    end
end

function ELDR_subproblem(x1, qs)
    # x1 is the sole chief decision
    # dependency of Adjustable decisions
    # x2(z1), x3(z1, z2), xT(z1, ..., zT)
    # y1(z1), y2(z1, z2), yT(z1, ..., zT)
    J, card_q_at, part_sum_qs = qsize(qs)
    q = j -> j2q(j, part_sum_qs, qs)
    ι = JumpModel(1)
    JuMP.@variable(ι, y0[1:L])
    JuMP.@variable(ι, y1[1:L, 1:I])
    JuMP.@variable(ι, y2[1:L, 1:J])
    if true # NA constrs
        Bind = T-1
        # l is the first index of y0, y1, y2 
        # (l, y) = (1, x2), (2, x3), ..., (Bind, xT)
        # (l, y) = (Bind + 1, y1), (Bind + 2, y2), ..., (Bind + T, yT)
        for l in 1:T-1
            JuMP.fix.(y1[l, l                     + 1:end], 0.) # NA for x2 ~ xT, w.r.t. z's
            JuMP.fix.(y1[Bind + l, l              + 1:end], 0.) # NA for y1 ~ yT-1, w.r.t. z's
            JuMP.fix.(y2[l, part_sum_qs[l]        + 1:end], 0.) # NA for x2 ~ xT, w.r.t. u's
            JuMP.fix.(y2[Bind + l, part_sum_qs[l] + 1:end], 0.) # NA for y1 ~ yT-1, w.r.t. u's
        end
    end
    if true # 🔭 sub block of constrs and variables
        JuMP.@variable(ι, t[1:M])
        JuMP.@variable(ι, r[1:M, 1:I])
        JuMP.@expression(ι, s[m = 1:M, j = 1:J], ip(B[m, :], y2[:, j]))
        JuMP.@constraint(ι, η[m = 1:M], A0[m] * x1 - b(0)[m] + ip(B[m, :], y0) >= t[m])
        JuMP.@constraint(ι, ξ[i = 1:I, m = 1:M], ip(B[m, :], y1[:, i]) - b(i)[m] == r[m, i])
        if true # 🔭 dual cone of K(\bar W)
            JuMP.@variable(ι, w1[1:M, 1:I] >= 0.)
            JuMP.@variable(ι, w2[1:M, 1:I] >= 0.)
            JuMP.@variable(ι, v1[1:M, 1:J])
            JuMP.@variable(ι, v2[1:M, 1:J])
            JuMP.@variable(ι, v3[1:M, 1:J])
            JuMP.@constraint(ι, [m = 1:M, j = 1:J], [v1[m, j], v2[m, j], v3[m, j]] in JuMP.SecondOrderCone())
            JuMP.@constraint(ι, lambda_positive[m = 1:M], t[m] + sum(v2[m, :]) >= sum(v1[m, :]) + zBND * (sum(w1[m, :]) + sum(w2[m, :])))
            JuMP.@constraint(ι, u_free[m = 1:M, j = 1:J], s[m, j] == v1[m, j] + v2[m, j])
            JuMP.@constraint(ι, z_free[m = 1:M, i = 1:I], r[m, i] + w1[m, i] == w2[m, i] + sum(2 * v3[m, j] * q(j)[i] for j in 1:J))
        end
    end
    obj2 = ip(d, y0) + sum(h(q(j)) * ip(d, y2[:, j]) for j in 1:J)
    JuMP.@objective(ι, Min, obj2)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " terminates with JuMP.SLOW_PROGRESS "
    elseif status != JuMP.OPTIMAL
        error(" terminate with $(status) ")
    end
    eta = JuMP.dual.(η)
    @assert all(eta .>= 0)
    JuMP.objective_value(ι), eta, JuMP.dual.(ξ)
end

function ELDR(qs)
    J, card_q_at, part_sum_qs = qsize(qs)
    q = j -> j2q(j, part_sum_qs, qs)
    ι = JumpModel(1)
    JuMP.@variable(ι, 0. <= x1 <= xUB)
    JuMP.@variable(ι, y0[1:L])
    JuMP.@variable(ι, y1[1:L, 1:I])
    JuMP.@variable(ι, y2[1:L, 1:J])
    if true # NA constrs
        Bind = T-1
        # l is the first index of y0, y1, y2 
        # (l, y) = (1, x2), (2, x3), ..., (Bind, xT)
        # (l, y) = (Bind + 1, y1), (Bind + 2, y2), ..., (Bind + T, yT)
        for l in 1:T-1
            JuMP.fix.(y1[l, l                     + 1:end], 0.) # NA for x2 ~ xT, w.r.t. z's
            JuMP.fix.(y1[Bind + l, l              + 1:end], 0.) # NA for y1 ~ yT-1, w.r.t. z's
            JuMP.fix.(y2[l, part_sum_qs[l]        + 1:end], 0.) # NA for x2 ~ xT, w.r.t. u's
            JuMP.fix.(y2[Bind + l, part_sum_qs[l] + 1:end], 0.) # NA for y1 ~ yT-1, w.r.t. u's
        end
    end
    if true # 🔭 sub block of constrs and variables
        JuMP.@variable(ι, t[1:M])
        JuMP.@variable(ι, r[1:M, 1:I])
        JuMP.@expression(ι, s[m = 1:M, j = 1:J], ip(B[m, :], y2[:, j]))
        JuMP.@constraint(ι, η[m = 1:M], A0[m] * x1 - b(0)[m] + ip(B[m, :], y0) >= t[m])
        JuMP.@constraint(ι, ξ[i = 1:I, m = 1:M], ip(B[m, :], y1[:, i]) - b(i)[m] == r[m, i])
        if true # 🔭 dual cone of K(\bar W)
            JuMP.@variable(ι, w1[1:M, 1:I] >= 0.)
            JuMP.@variable(ι, w2[1:M, 1:I] >= 0.)
            JuMP.@variable(ι, v1[1:M, 1:J])
            JuMP.@variable(ι, v2[1:M, 1:J])
            JuMP.@variable(ι, v3[1:M, 1:J])
            JuMP.@constraint(ι, [m = 1:M, j = 1:J], [v1[m, j], v2[m, j], v3[m, j]] in JuMP.SecondOrderCone())
            JuMP.@constraint(ι, lambda_positive[m = 1:M], t[m] + sum(v2[m, :]) >= sum(v1[m, :]) + zBND * (sum(w1[m, :]) + sum(w2[m, :])))
            JuMP.@constraint(ι, u_free[m = 1:M, j = 1:J], s[m, j] == v1[m, j] + v2[m, j])
            JuMP.@constraint(ι, z_free[m = 1:M, i = 1:I], r[m, i] + w1[m, i] == w2[m, i] + sum(2 * v3[m, j] * q(j)[i] for j in 1:J))
        end
    end
    obj2 = ip(d, y0) + sum(h(q(j)) * ip(d, y2[:, j]) for j in 1:J)
    obj1 = ccost * x1
    JuMP.@objective(ι, Min, obj1 + obj2)
    JuMP.optimize!(ι)
    status = JuMP.termination_status(ι)
    if status == JuMP.SLOW_PROGRESS
        @warn " oneshot_inf_program terminates with JuMP.SLOW_PROGRESS"
    elseif status != JuMP.OPTIMAL
        error(" oneshot_inf_program terminate with $(status)")
    end
    JuMP.objective_value(ι), JuMP.value.([obj1, obj2]), JuMP.value.(x1)
end

for mainIte in 1:typemax(Int)
    ub, obj12, x1 = ELDR(qs)
    @info "mainIte = $mainIte, ub = $ub, x1 = $x1"
    obj2, eta, xi = ELDR_subproblem(x1, qs)
    bitvec = eta .> 5e-5
    etac, xic, Bc = eta[bitvec], xi[:, bitvec], B[bitvec, :] # 🏹 c: cut (to remove 0's) 
    xics = [xic[r, c] / etac[c] for r in eachindex(eachrow(xic)), c in eachindex(etac)] # 🏹 s: scaling
    # 🏹 cc: both rows (0's) and cols (NA) are cut
    find_already, vio_q = falses(1), zeros(I)
    # for Q2
    qlen = I-3 # 🏹
    Bcc, dc = Bc[:, end-3:end], d[end-3:end] # 🏹 Q2 -> [x3, x4, x5; y2, y3, y4, y5] Lstar = 7
    θ = (col, q, xics, etac) -> (etac[col] * ip(q, xics[1:qlen, col])^2)
    for lambda in eachcol(eye(length(dc)))
        Φ = (q, xics, etac, Bcc, dc, lambda) -> (h(q) * ip(dc, lambda) - sum(θ(c, q, xics, etac) * ip(Bcc[c, :], lambda) for c in eachindex(etac)))
        gen_res_at = q0 -> Optim.optimize(q -> Φ(q, xics, etac, Bcc, dc, lambda), q0, Optim.NewtonTrustRegion(); autodiff = :forward)
        for ite in 1:Num_q0_trial
            q0 = rand(ℙ, qlen)
            ℝ = gen_res_at(q0)
            qt, vt = Optim.minimizer(ℝ), Optim.minimum(ℝ)
            dir = dirvec(qt)
            val = Φ(dir, xics, etac, Bcc, dc, lambda)
            if val < VIO_BRINK
                vio_q[1:qlen] .= dir
                find_already[1] = true
                break
            end
        end
        find_already[1] && break
    end
    if find_already[1]
        @info "find $(vio_q) in Q2"
        qs[2] = [qs[2] vio_q] # 🏹
        continue
    end

    # for Q3
    qlen = I-2 # 🏹
    Bcc, dc = Bc[:, end-2:end], d[end-2:end] # 🏹 Q3 -> [x4, x5; y3, y4, y5] Lstar = 5
    θ = (col, q, xics, etac) -> (etac[col] * ip(q, xics[1:qlen, col])^2)
    for lambda in eachcol(eye(length(dc)))
        Φ = (q, xics, etac, Bcc, dc, lambda) -> (h(q) * ip(dc, lambda) - sum(θ(c, q, xics, etac) * ip(Bcc[c, :], lambda) for c in eachindex(etac)))
        gen_res_at = q0 -> Optim.optimize(q -> Φ(q, xics, etac, Bcc, dc, lambda), q0, Optim.NewtonTrustRegion(); autodiff = :forward)
        for ite in 1:Num_q0_trial
            q0 = rand(ℙ, qlen)
            ℝ = gen_res_at(q0)
            qt, vt = Optim.minimizer(ℝ), Optim.minimum(ℝ)
            dir = dirvec(qt)
            val = Φ(dir, xics, etac, Bcc, dc, lambda)
            if val < VIO_BRINK
                vio_q[1:qlen] .= dir
                find_already[1] = true
                break
            end
        end
        find_already[1] && break
    end
    if find_already[1]
        @info "find $(vio_q) in Q3"
        qs[3] = [qs[3] vio_q] # 🏹
        continue
    end

    # for Q4
    qlen = I-1 # 🏹
    Bcc, dc = Bc[:, end-1:end], d[end-1:end] # 🏹 Q4 -> [x5; y4, y5] Lstar = 3
    θ = (col, q, xics, etac) -> (etac[col] * ip(q, xics[1:qlen, col])^2)
    for lambda in eachcol(eye(length(dc)))
        Φ = (q, xics, etac, Bcc, dc, lambda) -> (h(q) * ip(dc, lambda) - sum(θ(c, q, xics, etac) * ip(Bcc[c, :], lambda) for c in eachindex(etac)))
        gen_res_at = q0 -> Optim.optimize(q -> Φ(q, xics, etac, Bcc, dc, lambda), q0, Optim.NewtonTrustRegion(); autodiff = :forward)
        for ite in 1:Num_q0_trial
            q0 = rand(ℙ, qlen)
            ℝ = gen_res_at(q0)
            qt, vt = Optim.minimizer(ℝ), Optim.minimum(ℝ)
            dir = dirvec(qt)
            val = Φ(dir, xics, etac, Bcc, dc, lambda)
            if val < VIO_BRINK
                vio_q[1:qlen] .= dir
                find_already[1] = true
                break
            end
        end
        find_already[1] && break
    end
    if find_already[1]
        @info "find $(vio_q) in Q4"
        qs[4] = [qs[4] vio_q] # 🏹
        continue
    end

    # for Q5
    Bcc, dc = Bc[:, end], d[end] # Q5 -> [y5] Lstar = 1
    θ = (col, q, xics, etac) -> (etac[col] * ip(q, xics[:, col])^2)
    Φ = (q, xics, etac, Bcc, dc) -> (h(q) * dc - sum(Bcc[c] * θ(c, q, xics, etac) for c in eachindex(etac)))
    gen_res_at = q0 -> Optim.optimize(q -> Φ(q, xics, etac, Bcc, dc), q0, Optim.NewtonTrustRegion(); autodiff = :forward)
    for ite in 1:Num_q0_trial
        q0 = rand(ℙ, I)
        ℝ = gen_res_at(q0)
        qt, vt = Optim.minimizer(ℝ), Optim.minimum(ℝ)
        dir = dirvec(qt)
        val = Φ(dir, xics, etac, Bcc, dc)
        if val < VIO_BRINK
            vio_q .= dir
            find_already[1] = true
            break
        end
    end
    if find_already[1]
        @info "find $(vio_q) in Q5"
        qs[5] = [qs[5] vio_q] # 🏹
    else
        @info "we can't find any more violating q's"
        return
    end
end





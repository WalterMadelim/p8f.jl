import JuMP, Gurobi; GRB_ENV = Gurobi.Env();
import JuMP.value as ı
import LinearAlgebra.⋅ as ⋅
import LinearAlgebra.norm as norm
import Random

function is_binary_bold(x) return abs(.5 - x) > (.5 - 5e-12) end; # a bold version of `is_binary`
function prev(t, d, T)
    n = t - d
    return n < 1 ? T+n : n
end; function gen_a_D_house(ID, DˈUB) return (
    ID = ID,
    D = rand(0:DˈUB, T), # base demand
    P_BUS = rand(12:24), # unidirectional due to no renewables
    P_EV = [2, rand(3:7)], # MIN and MAX charging power
    E_EV = rand(10:39),
    DW = rand(1:4, rand(2:5)), # cycle vector   
) end; function gen_a_G_house(ID) return (
    ID = ID,
    G = rand(0:15, T), # 🌞 renewables generation
    D = rand(2:9, T), # base demand
    P_BUS = rand(12:14), # bidirectional due to 🌞
    P_EV = [2, rand(3:10)], # MIN and MAX charging power
    E_EV = rand(10:39),
    DW = rand(1:7, rand(2:7)), # cycle vector
) end; function find_a_frac_block(χ)
    for j = Random.shuffle(1:J)
        ks = haskey(χ[j], :bLent) ? [:bLent, :bEV, :bDW] : [:bEV, :bDW]
        for k = ks, x = χ[j][k]
            is_binary_bold(ı(x)) || return j # [early return]
        end
    end; return 0 # all solutions are now agreeably integer
end; function solve_restricted_primal(B; mip = false)
    model = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)); # restricted primal
    if mip
        JuMP.set_attribute(model, "MIPGap", 0)
        JuMP.set_attribute(model, "MIPGapAbs", 0)
    else
        JuMP.set_silent(model) # This is an LP, used to generate fractional trial solutions
    end
    JuMP.@variable(model, λ[j = 1:J, v = 1:length(B[j][:p])] ≥ 0); JuMP.@constraint(model, [j = 1:J], sum(λ[j, :]) == 1);
    χ = NamedTuple[(; (k => similar(v, JuMP.VariableRef) for (k, v) = pairs(first(keys(B[j][:p]))))...) for j = 1:J]; # The "mean" vector
    for j = 1:J, k = keys(χ[j]), i = eachindex(χ[j][k])
        χ[j][k][i] = scalar_mean = JuMP.@variable(model)
        JuMP.@constraint(model, scalar_mean == sum(λ[j, v]nt[k][i] for (v, nt) = enumerate(keys(B[j][:p]))))
        mip || continue
        k ∈ [:bLent, :bEV, :bDW] && JuMP.set_integer(scalar_mean)
    end
    JuMP.@expression(model, pAgr[t = 1:T], sum(haskey(χ[j], :pBus) ? χ[j][:pBus][t] : χ[j][:pBus1][t]+χ[j][:pBus2][t] for j = 1:J));
    JuMP.@constraint(model, β[t = 1:T], P_AGR ≥ pAgr[t]); # 🟡 linking constr
    # ⚠️ we do not explicitly enforce the "no reverse flow" constraint, but will test it post-optimizing
    JuMP.@objective(model, Min, MarketPrice ⋅ pAgr);
    JuMP.optimize!(model); JuMP.assert_is_solved_and_feasible(model; allow_local = false)
    return χ
end; function initialize_inn(D)
    inn = Vector{JuMP.Model}(undef, J); # the intrablock MIP subproblems
    for (j, block) = enumerate(D)
        model = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # MIP of the j-th block
        JuMP.set_silent(model)
        JuMP.set_attribute(model, "Threads", 1)
        JuMP.set_attribute(model, "TimeLimit", 3)
        JuMP.set_attribute(model, "MIPGap", 0)
        JuMP.set_attribute(model, "MIPGapAbs", 0)
        if length(block) == 2 # [pBus1; pBus2; pLent; vec(pEV); bLent; vec(bEV); vec(bDW)]
            h1, h2 = block
            JuMP.@variable(model, bLent[1:T], Bin) # 0 is lending
            JuMP.@variable(model, pLent[1:T])
            JuMP.@constraint(model, [t = 1:T], pLent[t] ≤ (1 - bLent[t])h1[:P_EV][end])
            JuMP.@constraint(model, [t = 1:T], (1 - bLent[t])h1[:P_EV][begin] ≤ pLent[t])
            JuMP.@variable(model, bEV[1:T, 1:2], Bin)
            JuMP.@variable(model, pEV[1:T, 1:2])
            JuMP.@constraint(model, [t = 1:T, h = 1:2], pEV[t, h] ≤ bEV[t, h]block[h][:P_EV][end])
            JuMP.@constraint(model, [t = 1:T, h = 1:2], bEV[t, h]block[h][:P_EV][begin] ≤ pEV[t, h])
            JuMP.@constraint(model, [t = 1:T, h = 1:2], bEV[t, h] ≤ bLent[t])
            JuMP.@constraint(model, sum(        view(pEV, :, 1)) ≥ h1[:E_EV])
            JuMP.@constraint(model, sum(pLent + view(pEV, :, 2)) ≥ h2[:E_EV])
            JuMP.@variable(model, bDW[1:T, 1:2], Bin)
            JuMP.@expression(model, pDW[t = 1:T, h = 1:2], sum(bDW[prev(t, φ-1, T), h]P for (φ, P) = enumerate(block[h][:DW])))
            JuMP.@constraint(model, [h = 1:2], sum(view(bDW, :, h)) ≥ true)
            JuMP.@variable(model, -h1[:P_BUS] ≤ pBus1[1:T] ≤ h1[:P_BUS]) # 🟡
            JuMP.@variable(model, false ≤ pBus2[1:T] ≤ h2[:P_BUS]) # 🟡
            JuMP.@constraint(model, pBus1 + h1[:G] ≥ pLent + h1[:D] + view(pEV, :, 1) + view(pDW, :, 1))
            JuMP.@constraint(model, pBus2          ≥         h2[:D] + view(pEV, :, 2) + view(pDW, :, 2))
            JuMP.@expression(model, prim_obj, MarketPrice ⋅ (pBus1 + pBus2)) # the additional penalty is merely β-coefficiented
        elseif length(block) == 1 # [pBus; pEV; bEV; bDW]
            h = block[1]
            JuMP.@variable(model, bEV[1:T], Bin)
            JuMP.@variable(model, pEV[1:T])
            JuMP.@constraint(model, [t = 1:T], pEV[t] ≤ bEV[t]h[:P_EV][end])
            JuMP.@constraint(model, [t = 1:T], bEV[t]h[:P_EV][begin] ≤ pEV[t])
            JuMP.@constraint(model, sum(pEV) ≥ h[:E_EV])
            JuMP.@variable(model, bDW[1:T], Bin)
            JuMP.@expression(model, pDW[t = 1:T], sum(bDW[prev(t, φ-1, T)]P for (φ, P) = enumerate(h[:DW])))
            JuMP.@constraint(model, sum(bDW) ≥ true)
            JuMP.@variable(model, false ≤ pBus[1:T] ≤ h[:P_BUS]) # 🟡
            JuMP.@constraint(model, pBus ≥ h[:D] + pEV + pDW)
            JuMP.@expression(model, prim_obj, MarketPrice ⋅ pBus) # the additional penalty is merely β-coefficiented
        end; inn[j] = model # save
    end # not include: penalty objective, non-block linking constr
    return inn
end; function initialize_out(an_UB) # an_UB ⚠️ should be valid
    model = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # outer in DantzigWolfe decomposition
    JuMP.set_silent(model)
    JuMP.@variable(model, β[1:T] ≥ 0) # multiplier vector of the 🟡 linking constr
    JuMP.@variable(model, θ[1:J]) # This is a concise formulation---we omit the `pi` variable
    JuMP.@expression(model, common, -sum(β)P_AGR)
    JuMP.@expression(model, out_obj_tbMax, common + sum(θ))
    JuMP.@objective(model, Max, out_obj_tbMax)
    JuMP.@constraint(model, out_obj_tbMax ≤ an_UB)
    return model, θ, β
end; function reset_mj_obj!(mj, Β)
    ( haskey(mj, :pBus) ? 
      JuMP.@objective(mj, Min, mj[:prim_obj] + Β ⋅ mj[:pBus]) :
      JuMP.@objective(mj, Min, mj[:prim_obj] + Β ⋅ (mj[:pBus1] + mj[:pBus2])) )
end; function build_con(j, mj, θ, β)
    haskey(mj, :pBus) && return JuMP.@build_constraint(θ[j] ≤ ı(mj[:prim_obj]) + β ⋅ ı.(mj[:pBus]))
    return JuMP.@build_constraint(θ[j] ≤ ı(mj[:prim_obj]) + β ⋅ (ı.(mj[:pBus1]) + ı.(mj[:pBus2])))
end; function get_full_vec(nt)
    haskey(nt, :pBus) && return [nt[:pBus]; nt[:pEV]; nt[:bEV]; nt[:bDW]]
    return [nt[:pBus1]; nt[:pBus2]; nt[:pLent]; vec(nt[:pEV]); nt[:bLent]; vec(nt[:bEV]); vec(nt[:bDW])]
end; function get_bool_part_vec(nt)
    haskey(nt, :bLent) && return [nt[:bLent]; vec(nt[:bEV]); vec(nt[:bDW])]
    return [nt[:bEV]; nt[:bDW]]
end; function get_full_num_tuple(nt)
    haskey(nt, :pBus) && return (
        pBus = ı.(nt[:pBus]),
        pEV = ı.(nt[:pEV]),
        bEV = round.(Bool, ı.(nt[:bEV])),
        bDW = round.(Bool, ı.(nt[:bDW])) )
    return (
        pBus1 = ı.(nt[:pBus1]),
        pBus2 = ı.(nt[:pBus2]),
        pLent = ı.(nt[:pLent]),
        pEV = ı.(nt[:pEV]),
        bLent = round.(Bool, ı.(nt[:bLent])),
        bEV = round.(Bool, ı.(nt[:bEV])),
        bDW = round.(Bool, ı.(nt[:bDW]))
    )
end; function parallel_CG!(B, model, θ, β)
    for ite = 1:typemax(Int)
        JuMP.optimize!(model); JuMP.assert_is_solved_and_feasible(model; allow_local = false, dual = true)
        @info "ite = $ite, bound is $(JuMP.objective_bound(model))"
        Β = ı.(β) # ✂️
        out_upd_vec .= false
        Threads.@threads for j = 1:J
            Θ_j = ı(θ[j]); mj = inn[j] # ✂️
            reset_mj_obj!(mj, Β)
            JuMP.optimize!(mj); JuMP.assert_is_solved_and_feasible(mj; allow_local = false)
            Δ = Θ_j - JuMP.objective_value(mj)
            if Δ > 1e-11
                @info "Δ = $Δ, at j = $j"
                local intFullvertex = get_full_num_tuple(mj)
                if !haskey(B[j][:p], intFullvertex)
                    local con = build_con(j, mj, θ, β)
                    local cut = @lock my_lock JuMP.add_constraint(model, con)
                    B[j][:p][intFullvertex] = cut
                    B[j][:d][cut] = intFullvertex
                    out_upd_vec[j] = true
                end
            end
        end
        any(out_upd_vec) && continue
        break
    end
end; function condense!(B, model) # ⚠️🔴 This post-procedure could be harmful
    # e.g. lose the chance to recover a integer-feasible primal solution
    # or, even if you can recover one, its quality is inferior than the one
    # recovered were this post-procedure unexecuted
    for j = 1:J, (cut, vertex) = collect(B[j][:d]) # 🚰 condensing
        JuMP.dual(cut) == 0 || continue
        pop!(B[j][:p], vertex)
        pop!(B[j][:d], cut)
        JuMP.delete(model, cut) # delete one-by-one is already fast, so we don't bother reoptimizing
        JuMP.optimize!(model); JuMP.assert_is_solved_and_feasible(model; allow_local = false, dual = true)
    end; @info "Lagrangian bound is $(JuMP.objective_bound(model)) after condensing"
end; function restrict_a_frac_block!(B, model, inn)
    χ = solve_restricted_primal(B)
    j = find_a_frac_block(χ)
    if j == 0
        @info "▶▶▶ Now all fractional solutions has become integer"
        return true # you need to stop
    else
        @info "▶ eliminating fractionality in block $j"
    end
    mj = inn[j] # the block that we want to make 🌳 branching decision
    ⅁ = ı.(get_full_vec(χ[j])) # the concatenated fractional trial solution 
    v = get_full_vec(mj) # create a vec of decisions aligning with it
    o = JuMP.@variable(mj) # temporary epi-variable
    c1 = JuMP.@constraint(mj, v - ⅁ .≤ o) 
    c2 = JuMP.@constraint(mj, ⅁ - v .≤ o) # vector of constrs
    JuMP.@objective(mj, Min, o) # set-and-solve routine
    JuMP.optimize!(mj); JuMP.assert_is_solved_and_feasible(mj; allow_local = false) # the small MIP
    iv = get_full_num_tuple(mj) # integer-feasible vertex in NamedTuple format
    JuMP.delete.(mj, c1);
    JuMP.delete.(mj, c2);
    JuMP.delete(mj, o);
    iA = get_bool_part_vec(iv); # the Bool Anchor vector
    JuMP.fix.(get_bool_part_vec(mj), iA; force = true) # 🌳 The branching (= fixing) decision
    for (cut, vertex) = collect(B[j][:d])
        get_bool_part_vec(vertex) == iA && continue # keep the compatible ones previously gened
        pop!(B[j][:p], vertex)
        pop!(B[j][:d], cut)
        JuMP.delete(model, cut)
    end
    return false # don't need to stop
end; 

begin # Data acquisition (D)
    seed = 6787495924352825842 # [interesting case] IP_OPT_VALUE > LAG_DUAL_BOUND, but diving can recover IP_OPT_VALUE!
    # seed = rand(Int);
    Random.seed!(seed);
    T = 24; # global
    MarketPrice = [2, 1, 1, 1, 1, 1, 3, 5, 6, 7, 8, 9, 9, 8, 8, 9, 10, 11, 9, 6, 5, 3, 2, 2];
    P_AGR = 47; # pAgr should be in [0, P_AGR]
    D = Vector{Vector{NamedTuple}}(undef, 0); # the global data
    ID = 0; # each house has a unique ID
    while ID < 4 # build house pairs
        ID += 1; h1 = gen_a_G_house(ID)
        ID += 1; h2 = gen_a_D_house(ID, 10)
        push!(D, NamedTuple[h1, h2])
    end
    while ID < 6 # build house singles
        ID += 1; h = gen_a_D_house(ID, 10)
        push!(D, NamedTuple[h])
    end
    J = length(D)
    @info "a community with $J blocks and $ID houses have been built"
end;

if true # 🟪 centralized formulation: use primitive device constraints involving private data
    Χ = Vector{NamedTuple}(undef, J) # record the full standard optimal solution
    model = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # monolithic
    JuMP.set_attribute(model, "MIPGap", 0)
    JuMP.set_attribute(model, "MIPGapAbs", 0)
    for (j, block) = enumerate(D) # add intrablock constraints
        if length(block) == 2
            h1, h2 = block # decode
            # EV's part
            bLent = JuMP.@variable(model, [1:T], Bin) # 0 is lending
            pLent = JuMP.@variable(model, [1:T])
            JuMP.@constraint(model, [t = 1:T], pLent[t] ≤ (1 - bLent[t])h1[:P_EV][end])
            JuMP.@constraint(model, [t = 1:T], (1 - bLent[t])h1[:P_EV][begin] ≤ pLent[t])
            bEV = JuMP.@variable(model, [1:T, 1:2], Bin)
            pEV = JuMP.@variable(model, [1:T, 1:2])
            JuMP.@constraint(model, [t = 1:T, h = 1:2], pEV[t, h] ≤ bEV[t, h]block[h][:P_EV][end])
            JuMP.@constraint(model, [t = 1:T, h = 1:2], bEV[t, h]block[h][:P_EV][begin] ≤ pEV[t, h])
            JuMP.@constraint(model, [t = 1:T, h = 1:2], bEV[t, h] ≤ bLent[t])
            JuMP.@constraint(model, sum(        view(pEV, :, 1)) ≥ h1[:E_EV])
            JuMP.@constraint(model, sum(pLent + view(pEV, :, 2)) ≥ h2[:E_EV])
            # DW's part
            bDW = JuMP.@variable(model, [1:T, 1:2], Bin)
            pDW = JuMP.@expression(model, [t = 1:T, h = 1:2], sum(bDW[prev(t, φ-1, T), h]P for (φ, P) = enumerate(block[h][:DW])))
            JuMP.@constraint(model, [h = 1:2], sum(view(bDW, :, h)) ≥ true)
            # household balance
            pBus1 = JuMP.@variable(model, [1:T], lower_bound = -h1[:P_BUS], upper_bound = h1[:P_BUS])
            pBus2 = JuMP.@variable(model, [1:T], lower_bound = false,       upper_bound = h2[:P_BUS])
            JuMP.@constraint(model, pBus1 + h1[:G] ≥ pLent + h1[:D] + view(pEV, :, 1) + view(pDW, :, 1))
            JuMP.@constraint(model, pBus2          ≥         h2[:D] + view(pEV, :, 2) + view(pDW, :, 2))
            Χ[j] = (
                pBus1 = pBus1,
                pBus2 = pBus2,
                pLent = pLent,
                pEV = pEV,
                bLent = bLent,
                bEV = bEV,
                bDW = bDW
            )
        elseif length(block) == 1
            h = block[1] # decode
            # EV's part
            bEV = JuMP.@variable(model, [1:T], Bin)
            pEV = JuMP.@variable(model, [1:T])
            JuMP.@constraint(model, [t = 1:T], pEV[t] ≤ bEV[t]h[:P_EV][end])
            JuMP.@constraint(model, [t = 1:T], bEV[t]h[:P_EV][begin] ≤ pEV[t])
            JuMP.@constraint(model, sum(pEV) ≥ h[:E_EV])
            # DW's part
            bDW = JuMP.@variable(model, [1:T], Bin)
            pDW = JuMP.@expression(model, [t = 1:T], sum(bDW[prev(t, φ-1, T)]P for (φ, P) = enumerate(h[:DW])))
            JuMP.@constraint(model, sum(bDW) ≥ true)
            # household balance
            pBus = JuMP.@variable(model, [1:T], lower_bound = false, upper_bound = h[:P_BUS])
            JuMP.@constraint(model, pBus ≥ h[:D] + pEV + pDW)
            Χ[j] = (
                pBus = pBus,
                pEV = pEV,
                bEV = bEV,
                bDW = bDW
            )
        end
    end # we opt not to create an explicit public variable
    JuMP.@expression(model, pAgr, sum(haskey(Χ[j], :pBus) ? Χ[j][:pBus] : Χ[j][:pBus1]+Χ[j][:pBus2] for j = 1:J));
    JuMP.@constraint(model, pAgr .≤ P_AGR); # 🟡 linking constr
    # ⚠️ we do not explicitly enforce the "no reverse flow" constraint, but will test it post-optimize
    JuMP.@objective(model, Min, MarketPrice ⋅ pAgr);
    JuMP.optimize!(model); JuMP.assert_is_solved_and_feasible(model; allow_local = false)
    all(ı.(pAgr) .> 0) || error("This instance is not proper, please enlarge load")
    an_UB = JuMP.objective_value(model) + 500
end; # Note: use this standard procedure to pick seed leading to proper tests

inn = initialize_inn(D); 
my_lock = Threads.ReentrantLock();
model, θ, β = initialize_out(an_UB); # a concise outer model
B = [(p = Dict{NamedTuple, JuMP.ConstraintRef}(), d = Dict{JuMP.ConstraintRef, NamedTuple}()) for j = 1:J]; # a vec of Bijections::NamedTuple
out_upd_vec = falses(J);
parallel_CG!(B, model, θ, β); # ⚠️🔴 [deprecate] condense!(B, model)
# Note: at this line we had derived a typically tight Lagrangian bound

# Note: The following 2 methods are both primal heuristics, hence cannot guarantee optimality, even feasibility!
solve_restricted_primal(B; mip = true); # 1️⃣ This is a handy integer-feasible solution (typically good-quality)
while true # 2️⃣ the diving heuristic to recover another integer-feasible solution (dive only once here)
    restrict_a_frac_block!(B, model, inn) && break
    parallel_CG!(B, model, θ, β);
end

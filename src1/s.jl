

# include(joinpath(pwd(),"data","d_3_4_5.jl"))
include(joinpath(pwd(),"data","d_30_70_50.jl"))

# 2023/10/23
# a stable version of Lagrangian cut generation, to achieve z_LC

import LinearAlgebra
import MathOptInterface as MOI
import Gurobi
import FileIO

function test_sslp_parameters()
    @assert size(h) == (clients,scenes)
    @assert length(c) == sites
    @assert size(d) == (clients, sites)
    @assert size(d) == size(q)
    @assert length(q0) == length(c)
    @assert length(p) == scenes
end

function empty_ge_cut_vector()
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}[]
end

function is_2vec_close(v1,v2,tol=tol["Adaptive"])
    return LinearAlgebra.norm(v1-v2,Inf) <= tol
end

function perf_info_per_scene(s)
    return Q_star(s,c,1.)
end

function cut_plane_used_type()
    return Vector{<:Union{Vector{Float64},Float64}}
end

function hatE_initialization()
    tmp,hatE = similar(p),similar(p,Dict{String, cut_plane_used_type()})
    for s in 1:scenes
        ret = perf_info_per_scene(s)
        hatE[s], tmp[s]  = Dict("x" => [ret.x],"theta" => [ret.qsty]), ret.o
    end
    perf_info_bound = p' * tmp
    @info "The perfect information LB z_PI = $perf_info_bound"
    return hatE
end

function hatQ_initialization()
    hatQ = similar(p,Dict{String, cut_plane_used_type()})
    for s in 1:scenes
        hatQ[s] = Dict("pai" => [c],"pai0" => [1.], "cnst_L" => [Q_star(s,c,1.).o])
    end
    return hatQ
end

function silent_new_optimizer()
    o = Gurobi.Optimizer(GRB_ENV); # master
    MOI.set(o,MOI.RawOptimizerAttribute("OutputFlag"),0)
    return o
end

function terms_init(l)
    return [MOI.ScalarAffineTerm(0.,MOI.VariableIndex(0)) for _ in 1:l]
end

function Q(s,xt)::Float64 # s only affects rand variable at RHS of constr 2 in this function
    if any(xt .< 0.)
        @error "In Q(s,xt), there is negative entry in xt"
    elseif any(xt .> 1.)
        @error "In Q(s,xt), there is entry in xt that > 1.0"
    end
    o = silent_new_optimizer() # the s subproblem with trial point xt
    objterms = terms_init(sites + clients * sites) # if no penalty
    x = similar(xt,MOI.VariableIndex) # copy vector of xt
    for j in 1:sites
        x[j] = MOI.add_variable(o)
    end
    cpc = MOI.add_constraint.(o,x,MOI.EqualTo.(xt)) # copy constraint index vector

    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:sites
        y0[j] = MOI.add_variable(o)
    end
    objterms[1:sites] .= MOI.ScalarAffineTerm.(q0,y0)
    MOI.add_constraint.(o,y0,MOI.GreaterThan(0.)) # constr 5 (the last)

    y = similar(q,MOI.VariableIndex)
    for i in 1:clients
        terms = terms_init(sites) # constr 2
        for j in 1:sites
            y[i,j] = MOI.add_variable(o)
            objterms[sites + (i-1)sites + j] = MOI.ScalarAffineTerm(-q[i,j],y[i,j]) # NEGATIVE revenue
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.EqualTo(h[i,s])) # constr 2
    end
    MOI.add_constraint.(o,y,MOI.GreaterThan(0.)) # constr 4
    MOI.add_constraint.(o,y,MOI.LessThan(1.)) # constr 4
    # y ∈ Y
    MOI.add_constraint.(o,y,MOI.Integer())

    for j in 1:sites
        terms = terms_init(clients+2)
        for i in 1:clients
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[clients+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[clients+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.LessThan(0.)) # constr 1
    end

    # obj function and SENSE of the s subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    maxvio::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio"))
    if maxvio > 1e-5
        @warn "Gurobi's MaxVio is $maxvio currently!"
    end
    if maxvio > tol["Gurobi"]
        tol["Gurobi"] = maxvio
        tol["Adaptive"] = 1.1 * maxvio
    end
    return MOI.get(o,MOI.ObjectiveValue())
end

function Q_star(s,pai,pai0)
    if pai0 < 0.
        @error "At input of Q_star(scene=$s), π0 should be ≥ 0.0"
    elseif pai0 == 0.
        # @warn "At input of Q_star(scene=$s), π0 == 0.0"
        # If Q_star() is called by algorithm 1, this is normal, pai0 == 0. means the respective scene is already well enough
    end
    o = silent_new_optimizer()
    objterms = terms_init(sites + sites+clients*sites)
    x = similar(pai,MOI.VariableIndex)
    for j in 1:sites
        x[j] = MOI.add_variable(o)
    end
    # 1st stage cost
    objterms[1:sites] .= MOI.ScalarAffineTerm.(pai,x)
    # Ax ≥ b
    MOI.add_constraint.(o,x,MOI.GreaterThan(0.))
    MOI.add_constraint.(o,x,MOI.LessThan(1.))
    # x ∈ X
    MOI.add_constraint.(o,x,MOI.Integer())

    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:sites
        y0[j] = MOI.add_variable(o)
    end
    objterms[sites+1:2sites] .= MOI.ScalarAffineTerm.(pai0 * q0,y0)
    MOI.add_constraint.(o,y0,MOI.GreaterThan(0.)) # constr 5 (the last)

    y = similar(q,MOI.VariableIndex)
    for i in 1:clients
        terms = terms_init(sites) # constr 2
        for j in 1:sites
            y[i,j] = MOI.add_variable(o)
            objterms[2sites + (i-1)sites + j] = MOI.ScalarAffineTerm(-pai0 * q[i,j],y[i,j]) # NEGATIVE revenue
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.EqualTo(h[i,s])) # constr 2
    end
    MOI.add_constraint.(o,y,MOI.GreaterThan(0.)) # constr 4
    MOI.add_constraint.(o,y,MOI.LessThan(1.)) # constr 4
    # y ∈ Y
    MOI.add_constraint.(o,y,MOI.Integer())

    for j in 1:sites
        terms = terms_init(clients+2)
        for i in 1:clients
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[clients+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[clients+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        MOI.add_constraint(o,MOI.ScalarAffineFunction(terms, 0.),MOI.LessThan(0.)) # constr 1
    end

    # obj function and SENSE of the s subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    status = MOI.get(o,MOI.TerminationStatus())
    if status != MOI.OPTIMAL
        @warn "pai"
        show(pai)
        @error "In Q_star(s = $s,pai,pai0 = $pai0), TerminationStatus = $status"
    else
        maxvio::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio"))
        if maxvio > 1e-5
            @warn "Gurobi's MaxVio is $maxvio currently!"
        end
        if maxvio > tol["Gurobi"]
            tol["Gurobi"] = maxvio
            tol["Adaptive"] = 1.1 * maxvio
        end
        obj = MOI.get(o,MOI.ObjectiveValue())
        xt = MOI.get.(o,MOI.VariablePrimal(),x)
        y0t = MOI.get.(o,MOI.VariablePrimal(),y0)
        yt = MOI.get.(o,MOI.VariablePrimal(),y)
        qsty = q0' * y0t - sum(q .* yt) # qsty, given xt, is typically used to enlarge the set Ê^s; qsty is valid, and fast, although may not be as precise as Q(s,xt)
    end
    return (o=obj, x=xt, y0=y0t, y=yt, qsty = qsty)
end

function test_eq_13() # should always pass as long as eq(13) is a valid cut
    i::Int = typemin(Int)
    while true
        s = rand(1:sites)
        x = rand(sites)
        pai = [rand(1:1e6) - 5e5 + rand() - .5 for _ in 1:sites]
        pai0 = rand() * rand(1:100)
        validity_13_check(s,x,pai,pai0)
        if i == typemax(Int)
            i = typemin(Int)
        else
            i += 1
        end
        print("\r")
        print(i)
    end
end

function validity_13_check(s,x,pai,pai0)
    if any(x .< 0.)
        @error "In validity_13_check, there is negative entry in xt"
    elseif any(x .> 1.)
        @error "In validity_13_check, there is entry in xt that > 1.0"
    end
    if pai0 == 0.0
        @warn "At validity_13_check, π0 == 0.0"
    elseif pai0 < 0.0 
        @error "At validity_13_check, π0 should be ≥ 0.0"
    end
    value = Q(s,x)
    under_val = (Q_star(s,pai,pai0).o - pai' * x)/pai0
    if value < under_val
        @error "validity_13_check Fails, (13) is not a valid cut!"
    elseif under_val + 0.5 > value
        @info "a tight (13) cut!"
    end
end

function algorithm1(s,hat_x,hat_θ_s) # there is also a global input hatE
    if any(hat_x .< 0.)
        @error "In algorithm1, there is negative entry in xt"
    elseif any(hat_x .> 1.)
        @error "In algorithm1, there is entry in xt that > 1.0"
    end
    delta = .99 # strict[0, 1]early_terminating
    # --------------------------------- Basic Framework ---------------------------------
    o = silent_new_optimizer()
    objterms = terms_init(sites+1+1) # (17) : pi, pi0, phi (φ <= ..., VS. theta >= ...)
    # π
    pai = similar(hat_x,MOI.VariableIndex)
    apai = similar(pai)
    for j in 1:sites
        pai[j] = MOI.add_variable(o)
        MOI.set(o,MOI.VariableName(),pai[j],"π[$j]")
        # abs(π[j]), with the following 2 constrs
        apai[j] = MOI.add_variable(o)
        # abs(π) >= π, >= -π  
        MOI.add_constraint(o,MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.,apai[j]),MOI.ScalarAffineTerm(-1.,pai[j])], 0.),MOI.GreaterThan(0.))
        MOI.add_constraint(o,MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.,apai[j]),MOI.ScalarAffineTerm(1.,pai[j])], 0.),MOI.GreaterThan(0.))
    end
    objterms[1:sites] .= MOI.ScalarAffineTerm.(-hat_x,pai) # neg
    # π0
    pai0 = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),pai0,"π0")
    objterms[sites+1] = MOI.ScalarAffineTerm(-hat_θ_s,pai0) # neg
    MOI.add_constraint(o,pai0,MOI.GreaterThan(0.)) # π0 ≥ 0
    # ||π||₁ + 1.0 * π0 ≤ 1. (α = 1.0 according to paper, for SSLP case)
    MOI.add_constraint(o, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm.(1.,apai);MOI.ScalarAffineTerm(1.0,pai0)], 0.), MOI.LessThan(1.))
    # φ 
    phi = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),phi,"φ")
    # register obj
    objterms[sites+2] = MOI.ScalarAffineTerm(1.,phi) # pos
    f = MOI.ScalarAffineFunction(objterms, 0.)
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f) # obj function
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE) # Maximize problem
    # --------------------------------- Basic Framework ---------------------------------
    # cuts describing φ, currently stored in hatE[s]
    phi_cuts = empty_ge_cut_vector()
    for i in eachindex(hatE[s]["theta"]) # initializing φ cuts
        terms = [MOI.ScalarAffineTerm.(hatE[s]["x"][i],pai); MOI.ScalarAffineTerm(hatE[s]["theta"][i],pai0); MOI.ScalarAffineTerm(-1.,phi)]
        push!(phi_cuts, MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(0.))) # 0 ≤ -φ + z * pai + thetaz * pai0, cf. (18)
    end
    # ------------------------ Backup Region ------------------------ 
    lb = -Inf # check this value is updated at least once before leaving algorithm 1 
    solution = [Inf for _ in 1:sites+1] # (π*,π0*)
    old_pai = [Inf for _ in 1:sites]
    incumbent_RHS_13 = Inf
    # ------------------------ Backup Region ------------------------ 
    while true
        # line 3: solve current cutting plane model, get an upper bound (MAX problem), get a new trial solution (π,π0)
        MOI.optimize!(o)
        if MOI.get(o,MOI.TerminationStatus()) != MOI.OPTIMAL
            @warn "input arg of algorithm1"
            show((s,hat_x,hat_θ_s))
            @error "In Algorithm 1, line 3, status != MOI.OPTIMAL"
        else
            maxvio::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio"))
            if maxvio > 1e-5
                @warn "Gurobi's MaxVio is $maxvio currently!"
            end
            if maxvio > tol["Gurobi"]
                tol["Gurobi"] = maxvio
                tol["Adaptive"] = 1.1 * maxvio
            end
            ub,pait,pai0t = MOI.get(o,MOI.ObjectiveValue()),MOI.get.(o,MOI.VariablePrimal(),pai),MOI.get(o,MOI.VariablePrimal(),pai0)
        end
        # line 4, part 1: check the quality of the trial solution (π,π0) derived in line 3, and evaluate the true Q_star(π,π0), returns the true value (later for lb), slope
        if pai0t >= 1e-4 # the normal case
            ret = Q_star(s,pait,pai0t) # This line takes much time due to MIP
            precise_value, x, theta = ret.o, ret.x, ret.qsty
        else
            if pai0t < 0.
                @warn "Gurobi gives a negative pai0t solution in line3, algorithm1, we reset it manually"
                pai0t = 0.
            end
            # This block ( if 0 <= pai0t < 1e-4 ) is common, thus we do not add @warn information temporarily
            ret = Q_star(s,pait,pai0t) # This line takes much time due to MIP
            precise_value, x = ret.o, ret.x
            theta = Q(s,x) # have to do one more MIP
        end
        # line 4, part 2: check the slope derived is new in hatE[s], and update both the current cutting plane model and global hatE accordingly
        slope_is_new = true # where slope means (x,theta), cf.(18)
        curr_len = length(hatE[s]["theta"]) # to ensures a static iteration
        for i in 1:curr_len
            if is_2vec_close(x,hatE[s]["x"][i])
                if theta <= hatE[s]["theta"][i] - tol["Adaptive"] # this current point is strictly better
                    # replace the old cut with the new, and record constrIndex in phi_cuts at the old place
                    MOI.delete(o,phi_cuts[i])
                    terms = [MOI.ScalarAffineTerm.(x, pai); MOI.ScalarAffineTerm(theta, pai0); MOI.ScalarAffineTerm(-1., phi)]
                    phi_cuts[i] = MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(0.)) # this new cut in place of the old one
                    # in-(old)place update hatE
                    hatE[s]["theta"][i] = theta
                    hatE[s]["x"][i] .= x
                elseif theta >= hatE[s]["theta"][i] + tol["Adaptive"]
                    @warn "The new generated cut reproduce an old same slope, and is an inferior one!"
                end
                slope_is_new = false
                break # early break, because x can at most be close to one slope stored in vector hatE[s]["x"]
            end
        end
        if slope_is_new
            # add this new extended distinct trial point to hatE
            push!(hatE[s]["x"], x)
            push!(hatE[s]["theta"], theta)
            terms = [MOI.ScalarAffineTerm.(x, pai); MOI.ScalarAffineTerm(theta, pai0); MOI.ScalarAffineTerm(-1., phi)]
            # 1, add this constraint to optimizer, 2, register it in vector phi_cuts
            push!(phi_cuts,MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(0.)))
        end
        # line 5
        newlb = precise_value - hat_x' * pait - hat_θ_s' * pai0t
        if newlb > lb # Must run into this if for at least once before leaving Algorithm 1
            lb, incumbent_RHS_13 = newlb, precise_value
            solution .= [pait; pai0t]
        end
        # println("lb = $lb < $ub = ub")
        if is_2vec_close(old_pai,pait)
            @warn "leave algorithm1 due to π stalling."
            break
        else
            old_pai .= pait
        end
        if ub <= 1e-6        # 1e-6 * (abs(hat_θ_s) + 1.)
            # @warn "leave algorithm1 (scene=$s) due to hopeless to find a strong violated cut. > $hat_θ_s <"
            break
        end
        if lb > (1.0 - delta) * ub # this exit is healthy, so we omit this output info
            # @info "leave algorithm1 for a faster speed."
            break
        end
    end
    if lb == -Inf
        @error "lb should be updated at least once in Algorithm 1!"
    end
    return solution[1:end-1], solution[end], incumbent_RHS_13 # used to construct "(1) x + (2) θs ≥ (3)" the Lag cut (13)
end

function master()
    o = silent_new_optimizer()
    objterms = terms_init(sites + scenes) # c[1]x[1]+c[2]x[2]+c[3]x[3] + p[1]θ[1] + ... + p[5]θ[5]
    # x
    x = similar(c,MOI.VariableIndex)
    for j in 1:sites
        x[j] = MOI.add_variable(o)
        MOI.set(o,MOI.VariableName(),x[j],"x[$j]")
    end
    # 1st stage cost
    objterms[1:sites] .= MOI.ScalarAffineTerm.(c,x)
    # only Ax ≥ b, relax x ∈ X, in master problem
    MOI.add_constraint.(o,x,MOI.GreaterThan(0.))
    MOI.add_constraint.(o,x,MOI.LessThan(1.))
    # θ[s]'s for s in 1:scenes
    theta = similar(p,MOI.VariableIndex)
    for s in 1:scenes
        theta[s] = MOI.add_variable(o)
        MOI.set(o,MOI.VariableName(),theta[s],"θ[$s]")
    end
    # recourse cost
    objterms[sites+1:end] .= MOI.ScalarAffineTerm.(p,theta)
    # register objective function and sense
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f) # obj function
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE) # obj sense
    # θ[s] - x cuts in master problem
    theta_cuts = [empty_ge_cut_vector() for _ in 1:scenes] # currently only supports L-cut (13), thus ge by convention
    # add initial cuts to avoid unboundness
    L_cut_num = zeros(Int,scenes)
    for s in 1:scenes
        for i in eachindex(hatQ[s]["cnst_L"]) # initializing (13) cuts
            terms = [MOI.ScalarAffineTerm.(hatQ[s]["pai"][i],x); MOI.ScalarAffineTerm(hatQ[s]["pai0"][i],theta[s])]
            push!(theta_cuts[s], MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(hatQ[s]["cnst_L"][i])))
            L_cut_num[s] += 1
        end
    end
    times_entering_while = 1
    while true # improving Q̂(s,x) for s in 1:scenes, using L-cut (13) generated by algorithm1
        if maximum(L_cut_num) % 50 == 0
            @info "L_cut_num: $L_cut_num"
        end
        MOI.optimize!(o)
        status = MOI.get(o,MOI.TerminationStatus())
        if status != MOI.OPTIMAL
            @error "In master, TerminationStatus = $status"
        else
            maxvio::Float64 = MOI.get(o, Gurobi.ModelAttribute("MaxVio"))
            if maxvio > 1e-5
                @warn "Gurobi's MaxVio is $maxvio currently!"
            end
            if maxvio > tol["Gurobi"]
                tol["Gurobi"] = maxvio
                tol["Adaptive"] = 1.1 * maxvio
            end
            lb = MOI.get(o,MOI.ObjectiveValue())
            push!(lb_vector,lb)
            xt = MOI.get.(o,MOI.VariablePrimal(),x)
            # to avoid gurobi error, do the following Ax ≥ b rectification
            any(xt .< 0.) && (xt[xt .< 0.] .= 0.)
            any(xt .> 1.) && (xt[xt .> 1.] .= 1.)
            thetat = MOI.get.(o,MOI.VariablePrimal(),theta) # should be a violated trial value before converging
            println("\nCurrent Iterate ($times_entering_while):")
            @show xt' thetat' lb_vector'
        end
        new_cut_gened = false
        for s in 1:scenes
            pai,pai0,cnst_L = algorithm1(s,xt,thetat[s]) # Get the θ[s] - x cut coefficients
            if pai0 < 0.
                @error "In master, main loop, I think this is impossible!"
            elseif pai0 == 0.
                # @warn "At master_cut_generating: scene = $s, Going to generate a feasibility cut pai0 == 0.0"
            end
            if pai' * xt + pai0 * thetat[s] <= cnst_L - tol["Adaptive"] # extended trial (xt, thetat[s]) violates the L-cut generated
                slope_is_new = true
                curr_len = length(hatQ[s]["cnst_L"]) # to ensures a static iteration
                for i in 1:curr_len
                    if is_2vec_close( [pai; pai0], [hatQ[s]["pai"][i]; hatQ[s]["pai0"][i]] )
                        if cnst_L >= hatQ[s]["cnst_L"][i] + tol["Adaptive"] # this current L-cut is strictly better
                            # replace the old cut with the new, and record constrIndex in theta_cuts[s] at the old place
                            MOI.delete(o,theta_cuts[s][i])
                            terms = [MOI.ScalarAffineTerm.(pai, x); MOI.ScalarAffineTerm(pai0, theta[s])]
                            theta_cuts[s][i] = MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(cnst_L))
                            hatQ[s]["pai"][i] .= pai # in-(old)place update hatQ
                            hatQ[s]["pai0"][i], hatQ[s]["cnst_L"][i] = pai0, cnst_L
                        elseif cnst_L <= hatQ[s]["cnst_L"][i] - tol["Adaptive"]
                            @warn "In master: The new generated L-cut reproduce an old same slope, and is an inferior one!"
                        end
                        slope_is_new = false
                        break # early break, because is_2vec_close() can only be true at most once
                    end
                end
                if slope_is_new
                    # add this new extended distinct trial point to hatE
                    push!(hatQ[s]["pai"], pai)
                    push!(hatQ[s]["pai0"], pai0)
                    push!(hatQ[s]["cnst_L"], cnst_L)
                    terms = [MOI.ScalarAffineTerm.(pai, x); MOI.ScalarAffineTerm(pai0, theta[s])]
                    push!(theta_cuts[s],MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(cnst_L)))
                    L_cut_num[s] += 1
                    # @info "generating No. $(L_cut_num[s]) L-cut in scene($s): $pai * x + $pai0 * θ$s ≥ $cnst_L"
                    new_cut_gened = true
                end
            else
                @info "scene = $s, No violation" # often revealing that Q̂($s,x) is already well enough w.r.t. the master obj
            end
        end # end of for s in 1:scenes
        if !new_cut_gened
            str1 = "No more cut is generated in this iteration, ∀s ∈ S. Check if this current lb is well enough.\n"
            str1 *= "lb = $lb\n"
            str1 *= "L_cut_num: $L_cut_num\n"
            str1 *= "Leaving the loop on improving Q̂(s,x), ∀s ∈ S, in master()"
            @info str1
            break
        end
        times_entering_while += 1
    end # end of while
end

test_sslp_parameters()
const GRB_ENV = Gurobi.Env()
train_data_dir = joinpath(pwd(),"data","sslp_train_data.jld2")
# first train goes here
lb_vector = Float64[]
tol = Dict("Gurobi" => eps(), "Adaptive" => 1e-10) # a starting point
@info "- begin hatE initialization"
hatE = hatE_initialization()
@info "- begin hatQ initialization"
hatQ = hatQ_initialization() # use Lagrangian cuts, thus must be after hatE's initialization
@info "- begin master()"
master()

# # # if interrupted, save parameters
# FileIO.save(train_data_dir,"tol",tol,"hatE",hatE,"hatQ",hatQ,"lb_vector",lb_vector)

# # # if Exception happens during training, you have to fix errors. Continue to train goes here
# tol = FileIO.load(train_data_dir,"tol")
# hatE = FileIO.load(train_data_dir,"hatE")
# hatQ = FileIO.load(train_data_dir,"hatQ")
# lb_vector = FileIO.load(train_data_dir,"lb_vector")
# master()








# struct HatEs # please instantiate one hatEs per scene
#     x::Vector{Vector{Float64}}
#     theta::Vector{Float64} # should >= Q_s(x)
#     function HatEs(x,t)
#         return new(deepcopy(x),deepcopy(t))
#     end
# end

# struct HatQs # please instantiate one hatQs per scene
#     # Benders cuts      λ * x + 1 * θ_s ≥ cnst_B
#     lambda::Vector{Vector{Float64}}
#     cnst_B::Vector{Float64}
#     # Lag (13) cuts     π * x + π0 * θ_s ≥ cnst_L
#     pai::Vector{Vector{Float64}}
#     pai0::Vector{Float64}
#     cnst_L::Vector{Float64}
#     # inner constructor
#     function HatQs(t)
#         if length(t) == 2
#             l,c = t
#             return new(deepcopy(l),deepcopy(c),Vector{Float64}[],Float64[],Float64[])
#         elseif length(t) == 3
#             p,p0,c = t
#             return new(Vector{Float64}[],Float64[],deepcopy(p),deepcopy(p0),deepcopy(c))
#         end
#     end
# end


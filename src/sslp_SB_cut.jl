                # for the easy of copy
include(joinpath(pwd(),"data","d.jl"))
include(joinpath(pwd(),"data","h.jl"))
struct HatEs # please instantiate one hatEs per scene
    x::Vector{Vector{Float64}}
    theta::Vector{Float64} # should >= Q_s(x)
    function HatEs(x,t)
        return new(deepcopy(x),deepcopy(t))
    end
end
struct HatQs # please instantiate one hatQs per scene
    # Benders cuts      λ * x + 1 * θ_s ≥ cnst_B
    lambda::Vector{Vector{Float64}}
    cnst_B::Vector{Float64}
    # Lag (13) cuts     π * x + π0 * θ_s ≥ cnst_L
    pai::Vector{Vector{Float64}}
    pai0::Vector{Float64}
    cnst_L::Vector{Float64}
    # inner constructor
    function HatQs(l,c)
        a = new(deepcopy(l),deepcopy(c),Vector{Float64}[],Float64[],Float64[])
    end
end

function is_2vec_close(v1,v2,tol=1e-6)
    return LinearAlgebra.norm(v1-v2,Inf) <= tol
end

function is_int(v::Union{Vector{Float64},Float64})
    return LinearAlgebra.norm(v-round.(v),Inf) <= 1e-6
end

function silent_new_optimizer()
    o = Gurobi.Optimizer(GRB_ENV); # master
    MOI.set(o,MOI.RawOptimizerAttribute("OutputFlag"),0)
    return o
end

function terms_init(l)::Vector{MOI.ScalarAffineTerm{Float64}}
    return [MOI.ScalarAffineTerm(0.,MOI.VariableIndex(0)) for _ in 1:l]
end

function Q(xt,s_ind)::Float64 # precisely evaluate Q_s(x) at x = xt (not necessary int points)
    @assert -1e-6 <= minimum(xt) && maximum(xt) <= 1. + 1e-6 # validity of trial point
    o = silent_new_optimizer() # the s_ind subproblem with trial point xt
    objterms = terms_init(m+n*m) # if no penalty
    x = similar(xt,MOI.VariableIndex) # copy vector of xt
    cpc = similar(xt,MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}})
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        cpc[j] = MOI.add_constraint(o,x[j],MOI.EqualTo(xt[j])) # add copy constr immediately
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[0+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne())
            objterms[m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end

function Q_B(xt,s_ind)::NamedTuple # From Q: 1, relaxing y ∈ Y; 2, output
    @assert -1e-6 <= minimum(xt) && maximum(xt) <= 1. + 1e-6
    o = silent_new_optimizer() # the s_ind subproblem with trial point xt
    objterms = terms_init(m+n*m) # if no penalty
    x = similar(xt,MOI.VariableIndex) # copy vector of xt
    cpc = similar(xt,MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}})
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        cpc[j] = MOI.add_constraint(o,x[j],MOI.EqualTo(xt[j])) # add copy constr immediately
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[0+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.GreaterThan(0.))
            MOI.add_constraint(o,y[i,j],MOI.LessThan(1.))
            objterms[m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    obj = MOI.get(o,MOI.ObjectiveValue())
    slope = MOI.get.(o, MOI.ConstraintDual(), cpc) # slope is given by the solver!
    lambda = -slope # lambda is in section 2.3.BDD p2335
    return (l = lambda,c=obj + lambda' * xt)
end

function Q_star(s_ind,pai,pai0)::NamedTuple # MIP problem (12)
    @assert pai0 >= -1.0e-6
    o = silent_new_optimizer()
    objterms = terms_init(m + m+n*m) # 2 stages
    x = similar(pai,MOI.VariableIndex)
    for j in 1:m
        x[j] = MOI.add_variable(o)
        MOI.add_constraint(o,x[j],MOI.ZeroOne())
        objterms[0+j] = MOI.ScalarAffineTerm(pai[j],x[j])
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[m+j] = MOI.ScalarAffineTerm(pai0 * q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne())
            objterms[2m + m*(i-1)+j] = MOI.ScalarAffineTerm(pai0 * -d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j])
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    println("inside Q_star")
    obj = MOI.get(o,MOI.ObjectiveValue()) # precise value of Q_star(s_ind,pai,pai0)
    # num_result::Int = MOI.get(o,MOI.ResultCount()) # not necessary currently
    yt = MOI.get.(o,MOI.VariablePrimal(),y)
    y0t = MOI.get.(o,MOI.VariablePrimal(),y0)
    qsty = q0' * y0t -sum(d .* yt) # so called qˢ_⊤y, or theta_s^z
    xt = MOI.get.(o,MOI.VariablePrimal(),x)
    return (obj = obj,x = xt,theta = qsty)
end

function Q_star_hat(pai,pai0,s_ind)::Tuple{Float64, Int64} # take finite minimum (18)
    z_vec, value2_vec = eshat.z, pai0 * eshat.theta_z
    value1_vec = similar(value2_vec)
    for i in eachindex(value2_vec)
        value1_vec[i] = pai' * z_vec[i]
    end
    value_vec = value1_vec .+ value2_vec
    return findmin(value_vec)
end

function perf_info_per_scene(s_ind)::NamedTuple # the problem following p_s in (14.9)
    o = silent_new_optimizer()
    objterms = terms_init(m + m+n*m) # stage 1 and stage 2
    x = similar(c,MOI.VariableIndex)
    for j in 1:m
        x[j] = MOI.add_variable(o)
        MOI.add_constraint(o, x[j], MOI.ZeroOne()) # x ∈ X
        objterms[0+j] = MOI.ScalarAffineTerm(c[j],x[j])
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[m+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne()) # y ∈ Y
            objterms[2m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    objval = MOI.get(o,MOI.ObjectiveValue())
    xval = MOI.get.(o,MOI.VariablePrimal(),x)
    objval_2nd_stage = objval - c' * xval
    return (o = objval, x = xval, theta = objval_2nd_stage) # (z, theta_z) in (18)
end

function hatE_initialization()::Vector{HatEs}
    tmp,hatE = similar(p),similar(p,HatEs)
    for s_ind in eachindex(p)
        tmp[s_ind], x, theta = perf_info_per_scene(s_ind)
        hatE[s_ind] = HatEs([x],[theta])
    end
    perf_info_bound = p' * tmp
    @info "The perfect information LB z_PI = $perf_info_bound"
    return hatE
end


function algorithm1(s_ind,hat_x,hat_θ_s)
    delta = .5 # strict[0, 1]early_terminating
    # --------------------------------- Basic Framework ---------------------------------
    o = silent_new_optimizer()
    objterms = terms_init(m+1+1); # (17) : pi, pi0, phi (φ <= ..., VS. theta >= ...)
    pai = similar(hat_x,MOI.VariableIndex);
    apai = similar(pai)
    for j in eachindex(pai)
        pai[j] = MOI.add_variable(o)
        MOI.set(o,MOI.VariableName(),pai[j],"π[$j]")
        objterms[0+j] = MOI.ScalarAffineTerm(-hat_x[j],pai[j]) # neg
        apai[j] = MOI.add_variable(o) # abs(π[j]), with the following 2 constrs 
        MOI.add_constraint(o,MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.,apai[j]),MOI.ScalarAffineTerm(-1.,pai[j])], 0.),MOI.GreaterThan(0.))
        MOI.add_constraint(o,MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.,apai[j]),MOI.ScalarAffineTerm(1.,pai[j])], 0.),MOI.GreaterThan(0.))
    end
    pai0 = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),pai0,"π0")
    objterms[m+1] = MOI.ScalarAffineTerm(-hat_θ_s,pai0) # neg
    MOI.add_constraint(o,pai0,MOI.GreaterThan(0.)) # π0 ≥ 0
    # ||π||₁ + 1.0 * π0 ≤ 1. (α = 1.0 according to paper, for SSLP case)
    MOI.add_constraint(o, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm.(1.,apai);MOI.ScalarAffineTerm(1.0,pai0)], 0.), MOI.LessThan(1.))
    phi = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),phi,"φ")
    objterms[m+2] = MOI.ScalarAffineTerm(1.,phi)
    f = MOI.ScalarAffineFunction(objterms, 0.)
    MOI.set(o,MOI.ObjectiveFunction{typeof(f)}(),f) # obj function
    MOI.set(o, MOI.ObjectiveSense(), MOI.MAX_SENSE) # Maximize problem
    # --------------------------------- Basic Framework ---------------------------------
    # cuts describing φ
    for i in eachindex(hatE[s_ind].theta)
        terms = [MOI.ScalarAffineTerm.(hatE[s_ind].x[i],pai); MOI.ScalarAffineTerm(hatE[s_ind].theta[i],pai0); MOI.ScalarAffineTerm(-1.,phi)]
        MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(0.)) # 0 ≤ -φ + z * pai + thetaz * pai0, cf. (18)
    end
    lb = -Inf
    solution = [Inf for _ in 1:m+1] # (π*,π0*)
    old_pai = [Inf for _ in 1:m]
    while true
        # line 3
        MOI.optimize!(o)
        @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
        ub,pait,pai0t = MOI.get(o,MOI.ObjectiveValue()),MOI.get.(o,MOI.VariablePrimal(),pai),MOI.get(o,MOI.VariablePrimal(),pai0)
        # line 4 (discard suboptimal solutions)
        precise_value,x,theta = Q_star(s_ind,pait,pai0t) # abuse of notation here, whatever
        push!(hatE[s_ind].x, x)
        push!(hatE[s_ind].theta, theta)
        terms = [MOI.ScalarAffineTerm.(x,pai); MOI.ScalarAffineTerm(theta,pai0); MOI.ScalarAffineTerm(-1.,phi)]
        MOI.add_constraint(o, MOI.ScalarAffineFunction(terms, 0.), MOI.GreaterThan(0.)) # 0 ≤ -φ + z * pai + thetaz * pai0, cf. (18)
        # line 5
        newlb = precise_value - hat_x' * pait - hat_θ_s' * pai0t
        if newlb > lb
            lb = newlb
            solution .= [pait; pai0t]
        end
        println("lb = $lb < $ub = ub")
        if is_2vec_close(old_pai,pait,1e-10)
            @info "leave algorithm1 due to π stalling."
            break
        else
            old_pai .= pait
        end
        if ub <= 1e-6 * (abs(hat_θ_s) + 1.)
            @info "leave algorithm1 due to hopeless to find a strong violated cut."
            break
        end
        if lb > (1.0 - delta) * ub
            @info "leave algorithm1 for a faster speed."
            break
        end
    end
    return solution[1:end-1],solution[end]
end

const GRB_ENV = Gurobi.Env()

o = silent_new_optimizer()
objterms = terms_init(m+scenes); # the master problem
x = similar(c,MOI.VariableIndex);
for j in eachindex(x)
    x[j] = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),x[j],"x[$j]")
    objterms[0+j] = MOI.ScalarAffineTerm(c[j],x[j])
    MOI.add_constraint(o, x[j], MOI.GreaterThan(0.))
    MOI.add_constraint(o, x[j], MOI.LessThan(1.))
end
xt::Vector{Float64} = let # initial Trial Point
    MOI.optimize!(o) # solve a FEASIBILITY problem,
    MOI.get.(o,MOI.VariablePrimal(),x)
end
theta = similar(p,MOI.VariableIndex);
for s_ind in eachindex(theta)
    theta[s_ind] = MOI.add_variable(o)
    MOI.set(o,MOI.VariableName(),theta[s_ind],"θ[$s_ind]")
    objterms[m+s_ind] = MOI.ScalarAffineTerm(p[s_ind],theta[s_ind])
end
hatQ = similar(p,HatQs) # create Q̂, the model in Algorithm 2
for s_ind in eachindex(p)
    ret = Q_B(xt,s_ind)
    hatQ[s_ind] = HatQs([ret.l], [ret.c]) # initialize hatQ
    terms = [MOI.ScalarAffineTerm.(ret.l,x); MOI.ScalarAffineTerm(1.,theta[s_ind])]
    f = MOI.ScalarAffineFunction(terms, 0.)
    MOI.add_constraint(o,f,MOI.GreaterThan(ret.c)) # B-cut at initial trial point
end
let f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f) # obj function
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE) # obj sense
end
thetat::Vector{Float64} = similar(p)
BendersNumber = 30
for cutFinding in 1:BendersNumber       # --------------- Benders phase ---------------
    MOI.optimize!(o) # solve master problem
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    lb = MOI.get(o,MOI.ObjectiveValue()) # c' * xt + p' * thetat
    @info " at $cutFinding, lb = $lb"
    xt .= MOI.get.(o,MOI.VariablePrimal(),x); # This is the 1st Trial Point
    # show(xt)
    # println()
    thetat .= MOI.get.(o,MOI.VariablePrimal(),theta);
    # show(thetat) # a component of thetat's update means the large update of lb
    # println()
    for s_ind in eachindex(p)
        ret = Q_B(xt,s_ind)
        if ret.l' * xt + thetat[s_ind] < ret.c - 1e-6 # violation at scene s_ind
            push!(hatQ[s_ind].lambda, ret.l)
            push!(hatQ[s_ind].cnst_B, ret.c)
            terms = [MOI.ScalarAffineTerm.(ret.l,x); MOI.ScalarAffineTerm(1.,theta[s_ind])]
            f = MOI.ScalarAffineFunction(terms, 0.)
            MOI.add_constraint(o,f,MOI.GreaterThan(ret.c))
        else
            error("-------------  Not Violated ! -------------")
        end
    end
end
@assert length(hatQ[rand(1:scenes)].lambda) == length(hatQ[rand(1:scenes)].cnst_B) == 1 + BendersNumber
hatE = hatE_initialization() # this takes much time, do not run every time!
s_ind = 1
pai, pai0 = algorithm1(s_ind,xt,thetat[s_ind])
pai' * xt + pai0 * thetat[s_ind] , Q_star(s_ind,pai,pai0)
# for trialPointNumber in 1:30000
#     if !isnewtrial(tpool,xt)
#         @warn "Main loop exit due to no new Trial Point"
#         return
#     end
#     tpool = [tpool xt]
#     trial_is_int = is_int(xt)
#     if trial_is_int # update ub
#         @info "Int Trial x found in trialPointNumber $trialPointNumber"
#         for s_ind in 1:scenes
#             obj_2nd[s_ind] = Q(xt,s_ind)
#         end
#         newub = c' * xt + p' * obj_2nd
#         ub = newub < ub ? newub : ub
#     end
#     for s_ind in 1:scenes
#         ret = Q_SB(Q_B(xt,s_ind).l,s_ind) # trial_is_int ? Q_B(xt,s_ind) : Q_SB(Q_B(xt,s_ind).l,s_ind)
#         cnst = ret.o # o is the RHS at (10)
#         if ret.l' * xt + thetat[s_ind] < cnst - 1e-6 # violation
#             terms = [MOI.ScalarAffineTerm.(ret.l,x); MOI.ScalarAffineTerm(1.,theta[s_ind])] # λ = -slope
#             f = MOI.ScalarAffineFunction(terms, 0.)
#             newcol = [ret.l; cnst]
#             cpool[s_ind] = [cpool[s_ind] newcol] # record the cut coefficient, one pool per scene
#             MOI.add_constraint(o,f,MOI.GreaterThan(cnst)) # The initial B cut to avoid Unboundness
#         else
#             @warn ("No Violation Occurs!")
#             println("trialPointNumber = $trialPointNumber")
#             println("s_ind = $s_ind")
#             println("This is xt")
#             println(xt)
#             println("Theta[ind] is $(thetat[s_ind])")
#             println("lhs $(ret.l' * xt + thetat[s_ind]) >= rhs $(cnst)")
#             @warn ("No Violation Occurs!")
#             return
#         end
#     end
#     MOI.optimize!(o) # solve master problem
#     @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
#     lb = MOI.get(o,MOI.ObjectiveValue()) # c' * xt + p' * thetat
#     @assert lb < ub
#     @info "At Trial x $trialPointNumber: $lb < $ub"
#     xt = MOI.get.(o,MOI.VariablePrimal(),x)
#     thetat = MOI.get.(o,MOI.VariablePrimal(),theta)
# end






function zpi_c1()::Float64
    o = silent_new_optimizer()
    objterms = terms_init(m+scenes)
    x = similar(c,MOI.VariableIndex)
    for j in 1:m
        x[j] = MOI.add_variable(o)
        # MOI.add_constraint(o, x[j], MOI.GreaterThan(0.))
        # MOI.add_constraint(o, x[j], MOI.LessThan(1.))
        objterms[0+j] = MOI.ScalarAffineTerm(c[j],x[j])
    end
    theta = similar(p,MOI.VariableIndex);
    for s_ind in 1:scenes
        theta[s_ind] = MOI.add_variable(o)
        objterms[m+s_ind] = MOI.ScalarAffineTerm(p[s_ind],theta[s_ind])
    end
    # add c1 cut
    for s_ind in 1:scenes
        terms = terms_init(m+1)
        for j in 1:m
            terms[j] = MOI.ScalarAffineTerm(c[j],x[j])
        end
        terms[m+1] = MOI.ScalarAffineTerm(1.,theta[s_ind])
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.GreaterThan(Q_star(c,1.,s_ind))) # constr 1
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms, 0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    return MOI.get(o,MOI.ObjectiveValue())
end

function distill(z::Vector{Vector{Float64}},t::Vector{Float64})
    z_vec,t_vec = deepcopy(z),deepcopy(t)
    n_vec = Vector{Float64}[]
    nt_vec = Float64[]
    rep_vec = Vector{Float64}[]
    rept_vec = Float64[]
    while !isempty(t_vec)
        rep = 0
        for i in 2:length(t_vec)
            if is_2vec_close(z_vec[1],z_vec[i])
                rep = i
                break
            end
        end
        if rep == 0
            push!(n_vec,popfirst!(z_vec))
            push!(nt_vec,popfirst!(t_vec))
        else # currently z_vec[1] == z_vec[rep]
            delInd = t_vec[1] < t_vec[rep] ? rep : 1
            push!(rep_vec,popat!(z_vec,delInd))
            push!(rept_vec,popat!(t_vec,delInd))
        end
    end
    return n_vec,nt_vec #,rep_vec,rept_vec
end


function Q_SB(lambda,s_ind)::NamedTuple # precisely evaluate Q_s(x) at x = xt (not necessary int points)
    o = silent_new_optimizer() # the s_ind subproblem with trial point xt
    objterms = terms_init(m+n*m+m) # penalty has m terms
    x = similar(c,MOI.VariableIndex) # copy vector of xt
    for j in 1:m
        x[j] = MOI.add_variable(o) # copy variable named x
        MOI.add_constraint(o,x[j],MOI.ZeroOne()) # 1st-stage constraint add to copy vector, after relaxing copy constr
    end
    y0 = similar(q0,MOI.VariableIndex)
    for j in 1:m
        y0[j] = MOI.add_variable(o)
        MOI.add_constraint(o,y0[j],MOI.GreaterThan(0.))
        objterms[0+j] = MOI.ScalarAffineTerm(q0[j],y0[j])
    end
    y = similar(d,MOI.VariableIndex)
    for i in 1:n # row 1:70
        terms = terms_init(m) # constr 2
        for j in 1:m # col 1:30
            y[i,j] = MOI.add_variable(o)
            MOI.add_constraint(o,y[i,j],MOI.ZeroOne()) # 2nd-stage do NOT relax Int-constr!
            objterms[m+m*(i-1)+j] = MOI.ScalarAffineTerm(-d[i,j],y[i,j]) # negative sign! 
            terms[j] = MOI.ScalarAffineTerm(1.,y[i,j]) # constr 2
        end 
        f = MOI.ScalarAffineFunction(terms, 0.) # constr 2
        MOI.add_constraint(o,f,MOI.EqualTo(h[i,s_ind])) # constr 2
    end
    for j in 1:m
        terms = terms_init(n+2)
        for i in 1:n
            terms[i] = MOI.ScalarAffineTerm(d[i,j],y[i,j])
        end
        terms[n+1] = MOI.ScalarAffineTerm(-1.,y0[j])
        terms[n+2] = MOI.ScalarAffineTerm(-u,x[j]) # use of copy variable
        f = MOI.ScalarAffineFunction(terms, 0.)
        MOI.add_constraint(o,f,MOI.LessThan(0.)) # constr 1
    end
    # penalty in Obj
    for j in 1:m
        objterms[m+n*m+j] = MOI.ScalarAffineTerm(lambda[j],x[j])
    end
    # obj function and SENSE of the s_ind subproblem
    f = MOI.ScalarAffineFunction(objterms,0.)
    type_matters = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(o,type_matters,f)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # optimize!
    MOI.optimize!(o)
    @assert MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL
    xval = MOI.get.(o, MOI.VariablePrimal(), x)
    obj = MOI.get(o,MOI.ObjectiveValue()) # the equation above (10)
    return (l=lambda,o=obj)
end




function vv2m(vv) # vector of vector to matrix
    m = zeros(length(vv[1]),length(vv))
    for i in eachindex(vv)
        m[:,i] .= vv[i]
    end
    return m
end

function bin_vec_to_1_places(v::Vector{<:Real})::Vector{Int}
    return findall(x -> x > .5,v)
end

function bin_vec_to_0_places(v::Vector{<:Real})::Vector{Int}
    return findall(x -> x < .5,v)
end


function isnewtrial(p,x)::Bool
    for c in eachcol(p)
        tmp = maximum(abs.(c - x))
        if tmp < 1e-5
            @warn ("This Trial Already Exists!")
            println()
            @warn x
            println()
            @warn ("This Trial Already Exists!")
            return false
        end
    end
    return true
end

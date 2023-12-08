import JuMP
import Ipopt
import Gurobi
import PowerModels
import IntervalOptimisation

# 08/12/23
# we shall always apply PWL technique to CRITICAL nonlinear constraints, i.e., those that really needed
# because the computation burden is somewhat nontrivial
# we should use cutting planes and other alternatives for non-critical nonlinear constraints.

if true # functions
    function Base.sum(e::Vector{Any})
        return isempty(e) ? 0. : error("cannot sum a Any-element Vector")
    end
    function ct(uk,ul,tkl) # this function is not related to m building
        return uk * ul * cos(tkl), -uk * ul * sin(tkl) # ckl, tkl
    end # Additional: ckk = uk ^ 2, for any bus k
    function cnt_range(k, sense)
        if sense == 0 # injecting
            return ND["injecting_break_p"][k] : ND["injecting_break"][k]
        else # sense == 1: leaving
            return ND["leaving_break_p"][k] : ND["leaving_break"][k]
        end
    end
    function net_read(addr)
        D = PowerModels.parse_file(addr)
        @assert D["per_unit"] == true
        @assert D["source_version"] == "2"
        tmp = D["bus"],D["gen"],D["branch"],D["shunt"],D["load"]
        @assert all(length.(tmp) .> 0)
        return tmp
    end
    function l(n::Int)::String
        return "$n"
    end
    function data()
        tmp = net_read(joinpath(pwd(),"data","case30.m")) # We do not support double-line between 2 adjacent nodes currently!
        N, NG, L, Nshunt, Nload = length.(tmp)
        Dbus, Dgen, Dbranch, Dshunt, Dload = tmp 
        LD = Dict( # lines
            "from"  => Int[Dbranch[l(b)]["f_bus"] for b in 1:L],
            "to"    => Int[Dbranch[l(b)]["t_bus"] for b in 1:L],
            "r"     => Float64[Dbranch[l(b)]["br_r"] for b in 1:L],
            "x"     => Float64[Dbranch[l(b)]["br_x"] for b in 1:L],
            "l_b" => Float64[Dbranch[l(b)]["b_fr"] for b in 1:L],
            "r_b" => Float64[Dbranch[l(b)]["b_to"] for b in 1:L],
            "l_g" => Float64[Dbranch[l(b)]["g_fr"] for b in 1:L],
            "r_g" => Float64[Dbranch[l(b)]["g_to"] for b in 1:L]
        )
        GD = Dict(  # generators
            "node" => Int[Dgen[l(g)]["gen_bus"] for g in 1:NG],
            "Pmin" => Float64[Dgen[l(g)]["pmin"] for g in 1:NG],
            "Pmax" => Float64[Dgen[l(g)]["pmax"] for g in 1:NG],
            "Qmin" => Float64[Dgen[l(g)]["qmin"] for g in 1:NG],
            "Qmax" => Float64[Dgen[l(g)]["qmax"] for g in 1:NG],
            "c0" =>   Float64[Dgen[l(g)]["cost"][3] for g in 1:NG],
            "c1" =>   Float64[Dgen[l(g)]["cost"][2] for g in 1:NG],
            "c2" =>   Float64[Dgen[l(g)]["cost"][1] for g in 1:NG],
            "Pg" =>   Float64[Dgen[l(g)]["pg"] for g in 1:NG],
            "Qg" =>   Float64[Dgen[l(g)]["qg"] for g in 1:NG]
        )
        if false # This is a patch for [ case14.m ] due to the inherent contradiction in the MATPOWER source data
            GD["Qmax"] .= .5
            GD["Qmin"] .= -.5
        end
        loadD = Dict(
            "node"  => Int[Dload[l(ld)]["load_bus"] for ld in 1:Nload],
            "Pload" => Float64[Dload[l(ld)]["pd"] for ld in 1:Nload],
            "Qload" => Float64[Dload[l(ld)]["qd"] for ld in 1:Nload]
        )
        shuntD = Dict(
            "node"  => Int[Dshunt[l(s)]["shunt_bus"] for s in 1:Nshunt],
            "g"     => Float64[Dshunt[l(s)]["gs"] for s in 1:Nshunt],
            "b"     => Float64[Dshunt[l(s)]["bs"] for s in 1:Nshunt]
        )
        if true # ND = the most involved node_dictionary
            ND = merge( Dict(
                "type"  => Int[Dbus[l(n)]["bus_type"] for n in 1:N],
                "Vmin"  => Float64[Dbus[l(n)]["vmin"] for n in 1:N],
                "Vmax"  => Float64[Dbus[l(n)]["vmax"] for n in 1:N],
                "V"     => Float64[Dbus[l(n)]["vm"] for n in 1:N],
                "tha"   => Float64[Dbus[l(n)]["va"] for n in 1:N],
                "P"     => Float64[sum(loadD["Pload"][findall(x -> x == n, loadD["node"])]) for n in 1:N],
                "Q"     => Float64[sum(loadD["Pload"][findall(x -> x == n, loadD["node"])]) for n in 1:N] 
            ),          Dict(  # this part is auto-generated
            "injecting_branch" => [findall(x -> x == n,LD["to"]) for n in 1:N],
            "leaving_branch" => [findall(x -> x == n,LD["from"]) for n in 1:N],
            "G_ind" => [findall(x -> x == n, GD["node"]) for n in 1:N],
            "load_ind" => [findall(x -> x == n, loadD["node"]) for n in 1:N]
            ) )
            tmp1, tmp2 = [length(i) for i in ND["leaving_branch"]], [length(i) for i in ND["injecting_branch"]]
            ND = merge(ND, Dict( # this part is auto-generated
            "leaving_break"     => [sum(tmp1[1:e]) for e in eachindex(tmp1)],
            "injecting_break"   => [sum(tmp2[1:e]) for e in eachindex(tmp2)],
            "injecting_node" => [Int[LD["from"][b] for b in ND["injecting_branch"][n]] for n in 1:N],
            "leaving_node" => [Int[LD["to"][b] for b in ND["leaving_branch"][n]] for n in 1:N])
            )
            ND = merge(ND, Dict(
                "leaving_break_p" => ND["leaving_break"] .- tmp1 .+ 1,
                "injecting_break_p" => ND["injecting_break"] .- tmp2 .+ 1
            ))
        end
        ykk = [ sum([LD["l_g"][b] + im * LD["l_b"][b] for b in ND["leaving_branch"][n]]) + 
                sum([LD["r_g"][b] + im * LD["r_b"][b] for b in ND["injecting_branch"][n]]) +
                sum([shuntD["g"][s] + im * shuntD["b"][s] for s in findall(x -> x == n, shuntD["node"])]) for n in 1:N ]
        gkk, bkk = real.(ykk), imag.(ykk)
        # these parameters presents in paper 
        Y = [(i == j ? ykk[i] : im * 0.) for i in 1:N, j in 1:N] # node admittance matrix
        for b in 1:L, k in [LD["from"][b]], l in [LD["to"][b]] # loop of k,l is dumb
            y = 1.0 / (LD["r"][b] + LD["x"][b] * im)
            Y[l,k] = Y[k,l] = -y # off-diagonal
            Y[k,k] += y
            Y[l,l] += y
        end # this version is Y in TextBook
        G,B = real(Y),imag(Y)
        N, NG, L, Nshunt, Nload, Dbus, Dgen, Dbranch, Dshunt, Dload, LD, GD, loadD, shuntD, ND, gkk, bkk, G, B
    end
    # PWL related
    function aminimizer(f, l, h) # a global solution, IntervalOptimisation boasted 
        interval = IntervalOptimisation.minimise(f, IntervalOptimisation.interval(l, h), tol=1e-4)[2][1]
        (interval.lo + interval.hi)/2
    end
    function dim1_x_max(f, l, h) # find a quasi global minimum x∗ of f over [l,h]
        x1 = aminimizer(x -> -f(x), l, h)
        m = JuMP.Model(Ipopt.Optimizer) # The valid range for this real option is 0 < tol and its default value is 1e-8
        JuMP.@variable(m, l <= x <= h, start = x1)
        JuMP.@objective(m, Max, f(x))
        JuMP.set_silent(m)
        JuMP.optimize!(m) 
        @assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
        x2 = JuMP.value(x)
        if abs(x2 - x1) > .5
            @error("x2 leaves x1 by $(abs(x2-x1)), where x1 = $x1 vs. $x2 = x2")
        end
        return (f(x2) > f(x1)) ? x2 : x1 # the quasi global optimum ("quasi" owing to no dual bounding provided)
    end
    function slo(f, xp, s) # (NL function, PWL ends, seg_num)
        (f(xp[s+1]) - f(xp[s])) / (xp[s+1] - xp[s])
    end
    function err_bnd(f, xp) # (NL function, PWL ends)
        n = length(xp) - 1
        epso, epsu = zeros(n), zeros(n)
        for s in 1:n
            anf = x -> ((f(xp[s]) + slo(f, xp, s) * (x - xp[s])) - f(x))
            epso[s] = anf(dim1_x_max(anf, xp[s], xp[s+1]))
            anf = x -> (f(x) - (f(xp[s]) + slo(f, xp, s) * (x - xp[s])))
            epsu[s] = anf(dim1_x_max(anf, xp[s], xp[s+1]))
        end
        return epso, epsu
    end
    function loc_seg(_x, xp) # locate the segment for which _x resides
        if _x >= xp[end]
            return n
        elseif _x <= xp[1]
            return 1
        end
        tmp = findall(x -> x == 0., xp .- _x)
        if !isempty(tmp)
            return (rand() > .5) ? tmp[1] : (tmp[1]-1)
        end
        return findall(x -> x > 0., xp .- _x)[1] - 1
    end
end

N, NG, L, Nshunt, Nload, Dbus, Dgen, Dbranch, Dshunt, Dload, LD, GD, loadD, shuntD, ND, gkk, bkk, G, B = data()

const line_heat_cap = .9
const dtmax = pi/4
GRB_ENV = Gurobi.Env()

input1 = [[-dtmax, 0.,dtmax] for _ in 1:L]

function socp_opf(vec_of_ps)
    m = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    # JuMP.set_silent(m)
    JuMP.@variable(m, ND["Vmin"][LD["from"][b]] * ND["Vmin"][LD["to"][b]] * cos(dtmax) <= ckl[b=1:L] <= ND["Vmax"][LD["from"][b]] * ND["Vmax"][LD["to"][b]] * cos(0.))
    JuMP.@variable(m, ND["Vmax"][LD["from"][b]] * ND["Vmax"][LD["to"][b]] * sin(-dtmax) <= tkl[b=1:L] <= ND["Vmax"][LD["from"][b]] * ND["Vmax"][LD["to"][b]] * sin(dtmax))
    JuMP.@variable(m, ND["Vmin"][n]^2 <= ckk[n=1:N] <= ND["Vmax"][n]^2)
    JuMP.@variable(m, tha[n=1:N])
    JuMP.fix(tha[begin], 0.)
    JuMP.@variable(m, -dtmax <= thakl[b=1:L] <= dtmax) # this is where the definition of dtmax comes from
    JuMP.@variable(m, sin(-dtmax) <= sterm[b=1:L] <= sin(dtmax)) # sin(thakl[b])
    JuMP.@variable(m, cos(dtmax) <= cterm[b=1:L] <= cos(0.)) # cos(thakl[b])
    JuMP.@variable(m, GD["Pmin"][g] <= pg[g = 1:NG] <= GD["Pmax"][g])
    JuMP.@variable(m, GD["Qmin"][g] <= qg[g = 1:NG] <= GD["Qmax"][g])
    JuMP.@variable(m, pbl[b=1:L])
    JuMP.@variable(m, qbl[b=1:L])
    JuMP.@variable(m, pbi[b=1:L])
    JuMP.@variable(m, qbi[b=1:L])
    if true # modeling of the complicated last 4 variables
        for k in 1:N, (b,l,cnt) in zip(ND["leaving_branch"][k], ND["leaving_node"][k], cnt_range(k, 1)) # (k) ---b--> (l)
            JuMP.@constraint(m, G[k, l] * (ckl[b]-ckk[k]) - B[k, l] * tkl[b] == pbl[cnt])
            JuMP.@constraint(m, -G[k, l] * tkl[b] - B[k, l] * (ckl[b]-ckk[k]) == qbl[cnt])
        end
        for k in 1:N, (b,l,cnt) in zip(ND["injecting_branch"][k], ND["injecting_node"][k], cnt_range(k, 0)) # (l) ---b--> (k)
            JuMP.@constraint(m, G[l, k] * (ckl[b]-ckk[k]) + B[l, k] * tkl[b] == pbi[cnt])
            JuMP.@constraint(m, G[l, k] * tkl[b] - B[l, k] * (ckl[b]-ckk[k]) == qbi[cnt])
        end
        JuMP.@constraint(m, [b = 1:L], pbl[b] ^ 2 + qbl[b] ^ 2 <= line_heat_cap ^ 2) # 3i_part_1
        JuMP.@constraint(m, [b = 1:L], pbi[b] ^ 2 + qbi[b] ^ 2 <= line_heat_cap ^ 2) # 3i_part_2
    end
    JuMP.@constraint(m, [k = 1:N], gkk[k] * ckk[k] + sum(pbl[ cnt_range(k, 1) ]) + sum(pbi[ cnt_range(k, 0) ]) 
    - sum([pg[g] for g in ND["G_ind"][k]]) == -ND["P"][k]) # reformulated nodal P balance 
    JuMP.@constraint(m, [k = 1:N], -bkk[k] * ckk[k] + sum(qbl[ cnt_range(k, 1) ]) + sum(qbi[ cnt_range(k, 0) ])
    - sum([qg[g] for g in ND["G_ind"][k]]) == -ND["Q"][k]) # reformulated nodal Q balance
    JuMP.@constraint(m, [b = 1:L], thakl[b] == tha[LD["from"][b]] - tha[LD["to"][b]])
    JuMP.@constraint(m, leqd[b = 1:L], ckl[b]^2 + tkl[b]^2 <= ckk[LD["from"][b]] * ckk[LD["to"][b]]) # The relaxed Quad constraint for Gurobi
    if true # JuMP.@NLconstraint(m, [b=1:L], ckl[b] * sin(thakl[b]) + tkl[b] * cos(thakl[b]) == 0.)
        JuMP.@variable(m, ckl_sterm_prod[b=1:L]) # bilinear term 
        JuMP.@variable(m, tkl_cterm_prod[b=1:L]) # bilinear term 
        JuMP.@constraint(m, [b=1:L], ckl_sterm_prod[b] + tkl_cterm_prod[b] == 0.) ### build the NL constraint -- the last step
        # McCormick 1
        JuMP.@constraint(m, [b=1:L], JuMP.lower_bound(ckl[b]) * sterm[b] + JuMP.lower_bound(sterm[b]) * ckl[b] - ckl_sterm_prod[b] <= JuMP.lower_bound(ckl[b]) * JuMP.lower_bound(sterm[b]))
        JuMP.@constraint(m, [b=1:L], JuMP.upper_bound(ckl[b]) * sterm[b] + JuMP.upper_bound(sterm[b]) * ckl[b] - ckl_sterm_prod[b] <= JuMP.upper_bound(ckl[b]) * JuMP.upper_bound(sterm[b]))
        JuMP.@constraint(m, [b=1:L], JuMP.upper_bound(ckl[b]) * sterm[b] + JuMP.lower_bound(sterm[b]) * ckl[b] - ckl_sterm_prod[b] >= JuMP.upper_bound(ckl[b]) * JuMP.lower_bound(sterm[b]))
        JuMP.@constraint(m, [b=1:L], JuMP.lower_bound(ckl[b]) * sterm[b] + JuMP.upper_bound(sterm[b]) * ckl[b] - ckl_sterm_prod[b] >= JuMP.lower_bound(ckl[b]) * JuMP.upper_bound(sterm[b]))
        # McCormick 2
        JuMP.@constraint(m, [b=1:L], JuMP.lower_bound(tkl[b]) * cterm[b] + JuMP.lower_bound(cterm[b]) * tkl[b] - tkl_cterm_prod[b] <= JuMP.lower_bound(tkl[b]) * JuMP.lower_bound(cterm[b]))
        JuMP.@constraint(m, [b=1:L], JuMP.upper_bound(tkl[b]) * cterm[b] + JuMP.upper_bound(cterm[b]) * tkl[b] - tkl_cterm_prod[b] <= JuMP.upper_bound(tkl[b]) * JuMP.upper_bound(cterm[b]))
        JuMP.@constraint(m, [b=1:L], JuMP.upper_bound(tkl[b]) * cterm[b] + JuMP.lower_bound(cterm[b]) * tkl[b] - tkl_cterm_prod[b] >= JuMP.upper_bound(tkl[b]) * JuMP.lower_bound(cterm[b]))
        JuMP.@constraint(m, [b=1:L], JuMP.lower_bound(tkl[b]) * cterm[b] + JuMP.upper_bound(cterm[b]) * tkl[b] - tkl_cterm_prod[b] >= JuMP.lower_bound(tkl[b]) * JuMP.upper_bound(cterm[b]))
        # PWL - sin portion
        n = length(vec_of_ps[1]) - 1 # assume length(vec_of_ps[b]) is the same for all b, thus take b=1 
        JuMP.@variable(m, 0. <= d1[1:L,1:n] <= 1.)
        JuMP.@variable(m, z1[1:L,1:n-1], Bin)
        for b in 1:L # PWL of JuMP.@constraint(m, [b=1:L], sterm[b] == sin(thakl[b]))
            xp = vec_of_ps[b]
            epso, epsu = err_bnd(sin, xp)
            JuMP.@constraint(m, [i=1:n-1], z1[b,i] >= d1[b,i+1])
            JuMP.@constraint(m, [i=1:n-1], z1[b,i] <= d1[b,i])
            JuMP.@constraint(m, thakl[b] == xp[1] + sum(d1[s] * (xp[s+1] - xp[s]) for s in 1:n))
            e = sterm[b] - (sin(xp[1]) + sum(d1[s] * (sin(xp[s+1]) - sin(xp[s])) for s in 1:n))
            JuMP.@constraint(m,  e <= epsu[1] + sum(z1[s] * (epsu[s+1] - epsu[s]) for s in 1:n-1))
            JuMP.@constraint(m, -e <= epso[1] + sum(z1[s] * (epso[s+1] - epso[s]) for s in 1:n-1))
        end
        # PWL - cos portion
        n = length(vec_of_ps[1]) - 1 # assume length(vec_of_ps[b]) is the same for all b, thus take b=1 
        JuMP.@variable(m, 0. <= d2[1:L,1:n] <= 1.)
        JuMP.@variable(m, z2[1:L,1:n-1], Bin)
        for b in 1:L # PWL of JuMP.@constraint(m, [b=1:L], cterm[b] == cos(thakl[b]))
            xp = vec_of_ps[b]
            epso, epsu = err_bnd(cos, xp)
            JuMP.@constraint(m, [i=1:n-1], z2[b,i] >= d2[b,i+1])
            JuMP.@constraint(m, [i=1:n-1], z2[b,i] <= d2[b,i])
            JuMP.@constraint(m, thakl[b] == xp[1] + sum(d2[s] * (xp[s+1] - xp[s]) for s in 1:n))
            e = cterm[b] - (cos(xp[1]) + sum(d2[s] * (cos(xp[s+1]) - cos(xp[s])) for s in 1:n))
            JuMP.@constraint(m,  e <= epsu[1] + sum(z2[s] * (epsu[s+1] - epsu[s]) for s in 1:n-1))
            JuMP.@constraint(m, -e <= epso[1] + sum(z2[s] * (epso[s+1] - epso[s]) for s in 1:n-1))
        end
    end
    JuMP.@objective(m, Min, sum(GD["c2"][g] * pg[g] * pg[g] + GD["c1"][g] * pg[g] + GD["c0"][g] for g in 1:NG)) # only pg related
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == JuMP.OPTIMAL
    @info "Relaxed SOCP OPF Optimal_ObjVal = $(JuMP.objective_value(m))" # 577.2636227512833
    # vio_dict = JuMP.primal_feasibility_report(m; atol = 1e-9);
    # @warn "After solving SOCP OPF, MaxVio = $(maximum(values(vio_dict)))" # 1.2115783500354915e-6
    JuMP.objective_value(m), [JuMP.value( thakl[b] ) for b in 1:L]
end

for i in 1:50000
    lb, vec = socp_opf(input1)
    println("at Main $i Ite: lb = $lb")
    for b in 1:L
        xp = input1[b]
        s = loc_seg(vec[b], xp) # locate the segment s where the current MILP solution _x resides
        xp = [xp[1:s]; (xp[s] + xp[s+1])/2; xp[s+1:end]] # longest edge bisection
        input1[b] = xp
    end
end





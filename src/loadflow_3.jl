import JuMP
import Ipopt
import PowerModels

# On solving a NLP OPF
# Use PowerModels.jl to read Matpower .m data files
# firstly, we initialize PF problem with heuristically generated operating states
# Then, we initialize OPF problem with PF solutions to make it more likely that Ipopt do not give a local_infeasible solution
# Thus we finally get a locally optimal solution of OPF problem such that the (TextBook) AC-power-flow equations is satisfied
# 02/12/23

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
    function PF_check_of_OPF_results(_Pinj, _Qinj, _V, _tha)
        m = JuMP.Model(Ipopt.Optimizer)
        JuMP.@variable(m, V[n=1:N], start = _V[n])
        JuMP.@variable(m, tha[n=1:N], start = _tha[n])
        JuMP.@variable(m, Pinj[n=1:N], start = _Pinj[n])
        JuMP.@variable(m, Qinj[n=1:N], start = _Qinj[n])
        JuMP.@NLconstraint(m, [i = 1:N], Pinj[i] == V[i] * sum(V[j]*(G[i,j] * cos(tha[i] - tha[j]) + B[i,j] * sin(tha[i] - tha[j])) for j in 1:N))
        JuMP.@NLconstraint(m, [i = 1:N], Qinj[i] == V[i] * sum(V[j]*(G[i,j] * sin(tha[i] - tha[j]) - B[i,j] * cos(tha[i] - tha[j])) for j in 1:N))
        pdict = Dict(x => JuMP.start_value(x) for x in JuMP.all_variables(m))
        PF_constraint_violations = JuMP.primal_feasibility_report(m, pdict)
        @info "" PF_constraint_violations
        @warn "MaxVio = $(maximum(values(PF_constraint_violations)))"
        @info "PF_check_of_OPF_results Finished."
    end
end

if true # data
    tmp = net_read(joinpath(pwd(),"data","case14.m")) # modify data source file here
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
    if true # This is a patch due to the inherent contradiction in the MATPOWER source data
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
    line_heat_cap = .9
end

if true # PF modeling
    m = JuMP.Model(Ipopt.Optimizer)
    JuMP.@variable(m, V[1:N])
    JuMP.@variable(m, tha[1:N])
    JuMP.@variable(m, Pinj[1:N])
    JuMP.@variable(m, Qinj[1:N])
    JuMP.@NLconstraint(m, [i = 1:N], Pinj[i] == V[i] * sum(V[j]*(G[i,j] * cos(tha[i] - tha[j]) + B[i,j] * sin(tha[i] - tha[j])) for j in 1:N))
    JuMP.@NLconstraint(m, [i = 1:N], Qinj[i] == V[i] * sum(V[j]*(G[i,j] * sin(tha[i] - tha[j]) - B[i,j] * cos(tha[i] - tha[j])) for j in 1:N))
    PQ_node_vec = findall(x -> x == 1, ND["type"])
    PV_node_vec = findall(x -> x == 2, ND["type"])
    Vt_node_vec = findall(x -> x == 3, ND["type"])
    @assert sum([length(PQ_node_vec),length(PV_node_vec),length(Vt_node_vec)]) == N
    @assert isdisjoint(GD["node"], PQ_node_vec) # PQ_node have load only
    for n in PQ_node_vec
        JuMP.fix(Pinj[n], -ND["P"][n])
        JuMP.fix(Qinj[n], -ND["Q"][n])
        JuMP.set_start_value(V[n], 1.)
        JuMP.set_start_value(tha[n], 0.)
        JuMP.set_start_value(Pinj[n], JuMP.fix_value(Pinj[n]))
        JuMP.set_start_value(Qinj[n], JuMP.fix_value(Qinj[n]))
    end
    @assert all( length.(ND["G_ind"][PV_node_vec]) .== 1 ) # G must exists at PV_node, and |G| == 1
    PV_RATIO = 4
    for n in PV_node_vec
        gind = ND["G_ind"][n][1]
        JuMP.fix(Pinj[n], GD["Pmin"][gind] + (GD["Pmax"][gind] - GD["Pmin"][gind])/PV_RATIO) # chosen by myself
        JuMP.fix(V[n], ND["V"][n]) # recommend by matpower
        JuMP.set_start_value(Qinj[n], GD["Qmin"][gind] + (GD["Qmax"][gind] - GD["Qmin"][gind])/2)
        JuMP.set_start_value(tha[n], 0.)
        JuMP.set_start_value(Pinj[n], JuMP.fix_value(Pinj[n]))
        JuMP.set_start_value(V[n], JuMP.fix_value(V[n]))
    end
    @assert length(Vt_node_vec) == 1 # only one Vt_node
    @assert length(ND["G_ind"][Vt_node_vec]) == 1 # only one Slack generator
    for n in Vt_node_vec # dumb loop
        gind = ND["G_ind"][n][1]
        JuMP.fix(V[n], ND["V"][n]) # recommend by matpower
        JuMP.fix(tha[n], 0.)
        JuMP.set_start_value(Pinj[n],sum(ND["P"][n] for n in PQ_node_vec) - sum(GD["Pmin"][gind] + (GD["Pmax"][gind] - GD["Pmin"][gind])/PV_RATIO for n in PV_node_vec))
        JuMP.set_start_value(Qinj[n],sum(ND["Q"][n] for n in PQ_node_vec) - sum(GD["Qmin"][gind] + (GD["Qmax"][gind] - GD["Qmin"][gind])/2 for n in PV_node_vec))
        JuMP.set_start_value(V[n], JuMP.fix_value(V[n]))
        JuMP.set_start_value(tha[n], JuMP.fix_value(tha[n]))
    end
end
pdict = Dict(x => JuMP.start_value(x) for x in JuMP.all_variables(m))
vio_dict = JuMP.primal_feasibility_report(m, pdict)
@warn "By heuristically initialization PF variables, MaxVio = $(maximum(values(vio_dict)))"

JuMP.optimize!(m) # pure PF solving, with 2N variables + 2N constraints
@assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
_V, _tha, _Pinj, _Qinj = JuMP.value.(V), JuMP.value.(tha), JuMP.value.(Pinj), JuMP.value.(Qinj)
# pdict = Dict(x => JuMP.value(x) for x in JuMP.all_variables(m)) # solutions after solving to Local_Optimal
vio_dict = JuMP.primal_feasibility_report(m) # with only `m` as arg, automatically use the last solution
@warn "After solving PF, MaxVio = $(maximum(values(vio_dict)))"

if true # The NLP -- OPF modeling
    m = JuMP.Model(Ipopt.Optimizer)
    JuMP.@variable(m, ckl[b=1:L], start = _V[LD["from"][b]] * _V[LD["to"][b]] * cos(_tha[LD["from"][b]] - _tha[LD["to"][b]]))
    JuMP.@variable(m, tkl[b=1:L], start = _V[LD["from"][b]] * _V[LD["to"][b]] * sin(_tha[LD["to"][b]] - _tha[LD["from"][b]]))
    JuMP.@variable(m, ND["Vmin"][n]^2 <= ckk[n = 1:N] <= ND["Vmax"][n]^2, start = _V[n] ^ 2)
    JuMP.@variable(m, tha[n=1:N], start = _tha[n])
    JuMP.fix(tha[begin], 0.)
    JuMP.@variable(m, thakl[b=1:L], start = _tha[LD["from"][b]] - _tha[LD["to"][b]])
    JuMP.@variable(m, GD["Pmin"][g] <= pg[g = 1:NG] <= GD["Pmax"][g], start = _Pinj[GD["node"][g]] + ND["P"][GD["node"][g]])
    JuMP.@variable(m, GD["Qmin"][g] <= qg[g = 1:NG] <= GD["Qmax"][g], start = _Qinj[GD["node"][g]] + ND["Q"][GD["node"][g]])
    JuMP.@variable(m, pbl[b=1:L])   # define 4 quantities related to heat capacity of a line, where pbl = real(sul), pbi = real(sur), [ul = upper left in a `π`-m]
    JuMP.@variable(m, qbl[b=1:L])
    JuMP.@variable(m, pbi[b=1:L])
    JuMP.@variable(m, qbi[b=1:L])
    if true # modeling of the complicated last 4 variables
        for k in 1:N, (b,l,cnt) in zip(ND["leaving_branch"][k], ND["leaving_node"][k], cnt_range(k, 1)) # (k) ---b--> (l)
            JuMP.@constraint(m, G[k, l] * (ckl[b]-ckk[k]) - B[k, l] * tkl[b] == pbl[cnt])
            JuMP.set_start_value(pbl[cnt], G[k, l] * ( JuMP.start_value(ckl[b]) - JuMP.start_value(ckk[k])) - B[k, l] * JuMP.start_value(tkl[b]))
            JuMP.@constraint(m, -G[k, l] * tkl[b] - B[k, l] * (ckl[b]-ckk[k]) == qbl[cnt])
            JuMP.set_start_value(qbl[cnt], -G[k, l] * JuMP.start_value(tkl[b]) - B[k,l] * (JuMP.start_value(ckl[b]) - JuMP.start_value(ckk[k])))
        end
        for k in 1:N, (b,l,cnt) in zip(ND["injecting_branch"][k], ND["injecting_node"][k], cnt_range(k, 0)) # (l) ---b--> (k)
            JuMP.@constraint(m, G[l, k] * (ckl[b]-ckk[k]) + B[l, k] * tkl[b] == pbi[cnt])
            JuMP.set_start_value(pbi[cnt] , G[l,k] * (JuMP.start_value(ckl[b]) - JuMP.start_value(ckk[k])) + B[l,k] * JuMP.start_value(tkl[b]))
            JuMP.@constraint(m, G[l, k] * tkl[b] - B[l, k] * (ckl[b]-ckk[k]) == qbi[cnt])
            JuMP.set_start_value(qbi[cnt] , -B[l,k] * (JuMP.start_value(ckl[b]) - JuMP.start_value(ckk[k])) + G[l,k] * JuMP.start_value(tkl[b]))
        end
        JuMP.@constraint(m, [b = 1:L], pbl[b] ^ 2 + qbl[b] ^ 2 <= line_heat_cap ^ 2) # 3i_part_1
        JuMP.@constraint(m, [b = 1:L], pbi[b] ^ 2 + qbi[b] ^ 2 <= line_heat_cap ^ 2) # 3i_part_2
    end
    JuMP.@constraint(m, [k = 1:N], gkk[k] * ckk[k] + sum(pbl[ cnt_range(k, 1) ]) + sum(pbi[ cnt_range(k, 0) ]) 
    - sum([pg[g] for g in ND["G_ind"][k]]) == -ND["P"][k] ) # reformulated nodal P balance 
    JuMP.@constraint(m, [k = 1:N], -bkk[k] * ckk[k] + sum(qbl[ cnt_range(k, 1) ]) + sum(qbi[ cnt_range(k, 0) ])
    - sum([qg[g] for g in ND["G_ind"][k]]) == -ND["Q"][k]) # reformulated nodal Q balance
    for b in 1:L, k in [LD["from"][b]], l in [LD["to"][b]] # loop for k,l is dumb
        JuMP.@constraint(m, thakl[b] == tha[k] - tha[l]) # defining relation of thakl
        JuMP.@constraint(m, ckl[b]^2 + tkl[b]^2 - ckk[k] * ckk[l] == 0.) # 3f, nonconvex quadratic
        JuMP.@NLconstraint(m, ckl[b] * sin(thakl[b]) + tkl[b] * cos(thakl[b]) == 0.) # (5) revised, << NONLINEAR Constraint !!
    end
    JuMP.@objective(m, Min, sum(GD["c2"][g] * pg[g] * pg[g] + GD["c1"][g] * pg[g] + GD["c0"][g] for g in 1:NG)) # only pg related
end
pdict = Dict(x => JuMP.start_value(x) for x in JuMP.all_variables(m))
vio_dict = JuMP.primal_feasibility_report(m, pdict; atol = 1e-6)
@warn "Initializing OPF variables with PF solutions, MaxVio = $(maximum(values(vio_dict)))"

JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED
@info "NLP OPF Local_Optimal_ObjVal = $(JuMP.objective_value(m))"
vio_dict = JuMP.primal_feasibility_report(m; atol = 1e-9)
@warn "After solving NLP OPF, MaxVio = $(maximum(values(vio_dict)))"

if true # get solutions of NLP OPF model
    _pg = JuMP.value.(  pg  )
    _qg = JuMP.value.(  qg  )
    _ckl = JuMP.value.( ckl )
    _tkl = JuMP.value.( tkl )
    _ckk = JuMP.value.( ckk )
    _tha = JuMP.value.( tha )
    _thakl = JuMP.value.( thakl )
    _pbl = JuMP.value.(pbl)
    _qbl = JuMP.value.(qbl)
    _pbi = JuMP.value.(pbi)
    _qbi = JuMP.value.(qbi)
end

# finally, check the PF-validity of the OPF results, such that the OPF formulation is proved to be equivalent to TextBook-style PF equations
Pinj, Qinj = -ND["P"], -ND["Q"]
Pinj[GD["node"]] .= Pinj[GD["node"]] .+ _pg
Qinj[GD["node"]] .= Qinj[GD["node"]] .+ _qg
V = sqrt.(_ckk)
PF_check_of_OPF_results(Pinj, Qinj, V, _tha)


import JuMP
import Ipopt
import PowerModels

if true # functions
    function Base.sum(e::Vector{Any})
        return isempty(e) ? 0. : error("cannot sum a Any-element Vector")
    end
    function vdict(x, x_) # used only once, at primal_feasibility_report
        return Dict(x[i] => x_[i] for i in eachindex(x))
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
    "G_ind" => [findall(x -> x == n,GD["node"]) for n in 1:N]
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

if true # feasibility PF modeling
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
    primal_value_dict = Dict{JuMP.VariableRef, Float64}()
    @assert sum([length(PQ_node_vec),length(PV_node_vec),length(Vt_node_vec)]) == N
    @assert isdisjoint(GD["node"], PQ_node_vec) # PQ_node have load only
    for n in PQ_node_vec
        JuMP.fix(Pinj[n], -ND["P"][n])
        JuMP.fix(Qinj[n], -ND["Q"][n])
        JuMP.set_start_value(V[n], 1.)
        JuMP.set_start_value(tha[n], 0.)
        primal_value_dict = merge(primal_value_dict, Dict(
            Pinj[n] => -ND["P"][n],
            Qinj[n] => -ND["Q"][n],
            V[n] => 1.,
            tha[n] => 0.
        ))
    end
    @assert all( length.(ND["G_ind"][PV_node_vec]) .== 1 ) # G must exists at PV_node, and |G| == 1
    PV_RATIO = 4
    for n in PV_node_vec
        gind = ND["G_ind"][n][1]
        JuMP.fix(Pinj[n], GD["Pmin"][gind] + (GD["Pmax"][gind] - GD["Pmin"][gind])/PV_RATIO) # chosen by myself
        JuMP.fix(V[n], ND["V"][n]) # recommend by matpower
        JuMP.set_start_value(Qinj[n], GD["Qmin"][gind] + (GD["Qmax"][gind] - GD["Qmin"][gind])/2)
        JuMP.set_start_value(tha[n], 0.)
        primal_value_dict = merge(primal_value_dict, Dict(
            Pinj[n] => GD["Pmin"][gind] + (GD["Pmax"][gind] - GD["Pmin"][gind])/PV_RATIO,
            Qinj[n] => GD["Qmin"][gind] + (GD["Qmax"][gind] - GD["Qmin"][gind])/2,
            V[n] => ND["V"][n],
            tha[n] => 0.
        ))
    end
    @assert length(Vt_node_vec) == 1 # only one Vt_node
    @assert length(ND["G_ind"][Vt_node_vec]) == 1 # only one Slack generator
    for n in Vt_node_vec # dumb loop
        gind = ND["G_ind"][n][1]
        JuMP.fix(V[n], ND["V"][n]) # recommend by matpower
        JuMP.fix(tha[n], 0.)
        JuMP.set_start_value(Pinj[n],sum(ND["P"][n] for n in PQ_node_vec) - sum(GD["Pmin"][gind] + (GD["Pmax"][gind] - GD["Pmin"][gind])/PV_RATIO for n in PV_node_vec))
        JuMP.set_start_value(Qinj[n],sum(ND["Q"][n] for n in PQ_node_vec) - sum(GD["Qmin"][gind] + (GD["Qmax"][gind] - GD["Qmin"][gind])/2 for n in PV_node_vec))
        primal_value_dict = merge(primal_value_dict, Dict(
            Pinj[n] => sum(ND["P"][n] for n in PQ_node_vec) - sum(GD["Pmin"][gind] + (GD["Pmax"][gind] - GD["Pmin"][gind])/PV_RATIO for n in PV_node_vec),
            Qinj[n] => sum(ND["Q"][n] for n in PQ_node_vec) - sum(GD["Qmin"][gind] + (GD["Qmax"][gind] - GD["Qmin"][gind])/2 for n in PV_node_vec),
            V[n] => ND["V"][n],
            tha[n] => 0.
        ))
    end
end

JuMP.optimize!(m)
@assert JuMP.termination_status(m) == JuMP.LOCALLY_SOLVED

_V, _tha, _Pinj, _Qinj = JuMP.value.(V), JuMP.value.(tha), JuMP.value.(Pinj), JuMP.value.(Qinj)
primal_value_dict = merge(
    vdict(V, _V),
    vdict(tha, _tha),
    vdict(Pinj, _Pinj),
    vdict(Qinj, _Qinj)
)
vio_dict = JuMP.primal_feasibility_report(m, primal_value_dict)
println("MaxVio = $(maximum(values(vio_dict)))")

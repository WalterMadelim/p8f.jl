# 📘 used letters
# R: RateA, this is an "add-on-demand" vector

# A: incidence
# B: num of branches
# N: num of nodes
# Nı: (@expr) num of nodes that power can be injected into
# L: (Data) number of loads, i.e. num of nodes that power can be consumed by load
# LD: load data
# f: branch flows

# TODO first use LP to check feasibility
# how to decide the demand level
# TODO then goes to practical MIP
# g = SimpleGraph(maximum(bft)); for (n, m) in eachrow(bft) # graph construction
#     add_edge!(g, n, m)
# end;
# Random.seed!(2527829590547944688);
# xv, yv = spring_layout(g); # supported by the layout algorithm, crude
# angle = 39; # choose an apt angle
# temp = clockwise_rotate([xv'; yv'], π/180 * angle);
# xv, yv = temp[1, :], temp[2, :];

using GLMakie
function m2(a, b) return (a + b)/2 end
function relocate_as_mean!(xv, yv, n, n1, n2) return (xv[n] = m2(xv[n1], xv[n2]); yv[n] = m2(yv[n1], yv[n2])) end
function relocate_similar!(xv, yv, n, n1, n2) return (xv[n] = xv[n1]; yv[n] = yv[n2]) end
function clockwise_rotate(A, t) return ((c, s) = (cos(t), sin(t)); [c s; -s c] * A) end
function load_xv_yv() return [-1.2332694234183728, -1.1423019779333998, -1.2332694234183728, -1.2332694234183728, -1.0753291281778001, -0.8712846524471404, -1.2332694234183728, -1.2332694234183728, -0.7131676455610054, -0.9732185344896891, -1.0242227096950343, -0.9616800619396255, -1.1057435104546034, -1.0337117861971143, -0.8165561156913704, -0.9909631789710944, -0.7486569345238162, -0.6355593963178844, -0.6632351535774644, -0.5244422772838138, -0.5064356474291548, -0.5070015905588766, -0.44166274957021806, -0.2524615058410598, -0.45311675663232176, -0.6124769366298216, -1.2332694234183728, -1.2332694234183728, -0.9173888329372275, -0.6555469573765893, -0.822913056058446, -0.7179522827007666, -0.7103046254215685, -0.978217597490834, -0.49046678844696323, -0.8157661429030421, -0.7343421929688987, -0.38871238850872064, -0.3827871060177682, -0.3901870319316366, -0.22459844347591224, -0.2324902337020509, -0.6030143547240652, -0.3415084771039843, -0.1840493884434555, -0.41702363107151186, -0.2699723062823807, -0.2361466088427399, -0.05526958661396796, -0.09324068603271475, -0.3901870319316366, -0.49046678844696323, -0.25819619018710044, -0.03661693037932902, 0.17983963704030623, -0.039178276573397106, -0.1029504045075008, -0.34624648244432366, 0.2860199469836183, 0.39822534028408313, 0.08645518004902494, 0.4383852539014527, 0.799339863314405, 0.08336863815804392, -0.46172963925033333, -0.07782449133480573, 0.35078666475954334, 0.17983963704030623, 0.08689646830922788, -0.032943022005884844, -0.16537321643966948, -0.1930658677036381, 0.11396203922533976, 0.11396203922533976, 0.15595321070539517, 0.35227506726030305, 0.3327226643187838, 0.4209899461543176, 0.4209899461543176, 0.44100951594343224, -0.7666446910745421, 0.5842335563231874, 0.5842335563231874, 1.0068859020946586, 0.9129593731242475, 0.8190101966189364, 0.2144625808699965, 0.9992395880909232, 0.9142908076844382, 1.0068859020946586, 0.8753287284835267, 0.7437715548723948, 1.0068859020946586, 0.6864742948716502, 0.8530136899457468, 0.6134820310739824, 0.7917365722070553, 0.5842335563231874, 0.5217336924787804, 0.5984557581029353, 0.2806601060833791, 0.19155783364374238, 1.0068859020946586, 0.6165897987807262, 0.6086568243465657, 0.5074075695324046, 0.39822534028408313, 0.807771363220612, 0.6050837585962591, 1.0068859020946586, 0.2661882930020731], [-0.6359322324887655, -0.2801354077171337, -0.13916047389182876, 0.357611284705108, 0.357611284705108, -0.09823662524979936, 0.057439388244673935, 0.24881459863298336, 0.5904994519511381, 0.5904994519511381, -0.07353986067309838, -0.18810869498709595, -0.6359322324887655, -0.4120204637379307, -0.18120827337862294, 0.057439388244673935, 0.057439388244673935, -0.025266401713440443, -0.15247798390566472, -0.020771795466661233, 0.16532886169602198, 0.3287166409894211, 0.44588858043852053, 0.48175941324764276, 0.5904994519511381, 0.36286628461652476, 0.4590827894279587, 0.5904994519511381, 0.357611284705108, 0.18839052610724827, 0.24881459863298336, 0.3970159282436284, -0.2792550322827453, -0.6359322324887655, -0.539211099566924, -0.539211099566924, -0.6359322324887655, -0.07158167687496608, -0.21383871847106745, -0.3890085343334909, -0.5009558087985091, -0.2631489069973201, -0.2669685726936926, -0.17527229949782977, -0.11589338972638322, 0.3736671277956627, 0.3701031661308718, 0.11262236156813654, -0.1484224046593896, -0.31930292792695336, -0.4424899666450825, -0.6359322324887655, -0.6359322324887655, -0.3849372530476313, -0.6359322324887655, -0.6359322324887655, -0.46270051803009415, -0.539211099566924, -0.4068442906786539, -0.30395100301703154, -0.27731196190273244, -0.014744650957470767, -0.601913204753674, -0.012644672129689408, 0.26949799474584235, 0.1279664535815132, 0.0906040454670044, 0.15128458190046368, 0.20475587231780748, 0.41148969127232243, 0.49476123263341265, 0.5904994519511381, 0.5904994519511381, 0.5464735117109394, 0.4079860017441864, 0.5204232203824178, 0.35352543714946266, 0.5904994519511381, 0.3686274436838723, 0.20276554121589438, 0.22923751353899313, 0.38397687856484647, 0.5904994519511381, 0.5904994519511381, 0.43164339368166266, 0.5218198861302757, 0.30517979810710005, 0.3816069992771466, 0.25223634296977604, 0.19977999798317844, -0.20502008172991626, -0.07158167687496608, 0.01947128045109523, 0.0695788792707168, 0.31751414094538627, 0.22462991997011136, 0.3827919389209965, 0.18966591169260563, 0.029923509033545126, -0.13202129486066388, -0.2206150963858769, -0.08158352780843019, -0.33845848658486644, -0.2928077881123642, -0.42875132830967644, -0.29837939556469784, -0.6359322324887655, -0.532341780399221, -0.6359322324887655, -0.6359322324887655, 0.5464735117109394] end
function custom_modification!(xv, yv)
    relocate_similar!(xv, yv, 87, 80, 68)
end;
(xv, yv) = load_xv_yv();
custom_modification!(xv, yv);
line_colors = [
    :chocolate, # Yellow
    :gold1,     # Yellow
    :blue,      # blue 
    :cyan,      # blue
    :lime,      # green lemon
    :darkgreen, # green
    :olive,     # green
    :deeppink,  # Red
    :red4,      # Red
    :black      
];
f = Figure(); ax = Axis(f[1, 1]); # 🖼️
Random.seed!(6)
function get_line_width(R_b) # arg: Rate_A of branch b
    m, M = minimum(R)-0.001, maximum(R)+0.001
    Δ = (M - m) / 18
    return 1 + (1 + Int(floor((R_b - m)/Δ)))
end
fontsize = 16
for b in 1:size(bft, 1) # Draw lines
    n, m = bft[b, :]; R_b = R[b]
    linesegments!(ax, [xv[n], xv[m]], [yv[n], yv[m]]; linewidth = get_line_width(R_b), color = rand(line_colors))
    text!(ax, m2(xv[n], xv[m]), m2(yv[n], yv[m]); text = "a$R_b", fontsize = fontsize, color = :black, align = (:center, :top))
end
fontsize = 18
for n in 1:maximum(bft) # Draw Points
    if n in gnv
        if n in lnv
            text!(ax, [xv[n]], [yv[n]]; color = :purple, text = "$n", fontsize = fontsize, align = (:left, :baseline))
            # scatter!(ax, [xv[n]], [yv[n]]; color = :purple, markersize=markersize)
            # text!(ax, [xv[n]], [yv[n]]; color = :purple, text = "$(Int(round(R_n[n])))", fontsize = fontsize, align = (:left, :top))
        else
            text!(ax, [xv[n]], [yv[n]]; color = :red, text = "$n", fontsize = fontsize, align = (:left, :baseline))
            # scatter!(ax, [xv[n]], [yv[n]]; color = :red, markersize=markersize)
            # text!(ax, [xv[n]], [yv[n]]; color = :red, text = "$(Int(round(R_n[n])))", fontsize = fontsize, align = (:left, :top))
        end
    elseif n in lnv
        text!(ax, [xv[n]], [yv[n]]; color = :skyblue, text = "$n", fontsize = fontsize, align = (:left, :baseline))
        # scatter!(ax, [xv[n]], [yv[n]]; color = :skyblue, markersize=markersize)
        # text!(ax, [xv[n]], [yv[n]]; color = :skyblue, text = "$(Int(round(R_n[n])))", fontsize = fontsize, align = (:left, :top))
    else
        text!(ax, [xv[n]], [yv[n]]; color = :black, text = "$n", fontsize = fontsize, align = (:left, :baseline))
        # scatter!(ax, [xv[n]], [yv[n]]; color = :black, markersize=markersize)
        # text!(ax, [xv[n]], [yv[n]]; color = :black, text = "$(Int(round(R_n[n])))", fontsize = fontsize, align = (:left, :top))
    end
end
hidedecorations!(ax); # hidespines!(ax);
f # After typing this in REPL, it will generate the picture

import JuMP, Gurobi
import LinearAlgebra.dot as dot
import SparseArrays.sparse as sparse
import Random
function adjnode(n)
    nv = Int[]
    for ci in findall(x -> x == n, bft)
        b, c = ci.I # row(= branch), col
        on = bft[b, 3 - c]
        on ∈ nv && error("in Function adjnode(), shouldn't happen")
        push!(nv, on)
    end
    sort(nv)
end
function search_extreme_cases_code() # 📘 invoke this code on demand
    sd_vec, obj_vec = Int[], Float64[]
    while true
        sd = rand(Int) # seed
        Random.seed!(sd)
        JuMP.delete(ø, c)
        LD = rand(0.03:7e-15:1) * lrand(L) .* R_lnv; # randomly generate a load vector at an instance
        c = JuMP.@constraint(ø, [n in 1:N], ı_at(n) + dot( 𝐟 , view(A, :, n)) == w_at(n)); # 📘 KCL
        JuMP.optimize!(ø)
        if JuMP.termination_status(ø) == JuMP.OPTIMAL
            push!(sd_vec, sd)
            push!(obj_vec, JuMP.objective_value(ø))
            l = length(sd_vec)
            print("\rl = $l")
            l == 1 && break
        end
    end
    while true
        sd = rand(Int) # seed
        Random.seed!(sd)
        JuMP.delete(ø, c)
        LD = rand(0.03:7e-15:1) * lrand(L) .* R_lnv; # randomly generate a load vector at an instance
        c = JuMP.@constraint(ø, [n in 1:N], ı_at(n) + dot( 𝐟 , view(A, :, n)) == w_at(n)); # 📘 KCL
        JuMP.optimize!(ø)
        if JuMP.termination_status(ø) == JuMP.OPTIMAL
            o = JuMP.objective_value(ø)
            if o < obj_vec[1]
                pushfirst!(sd_vec, sd)
                pushfirst!(obj_vec, o)
            elseif o > obj_vec[end]
                push!(sd_vec, sd)
                push!(obj_vec, o)
            else
                continue 
            end
            l = length(sd_vec)
            print("\rl = $l")
            l == 20 && break
        end
    end
end
function lrand() return sin(rand(0.1:7e-15:1.57)) end;
function lrand(N) return sin.(rand(0.1:7e-15:1.57, N)) end;
function bft_2_A(bft)
    B, N = size(bft, 1), maximum(bft)
    return sparse([Vector(1:B); Vector(1:B)], vec(bft), [-ones(Int, B); ones(Int, B)], B, N)
end;
function load_111_data()
    R = [10.76, 25.35, 137.5, 10.15, 20.32, 52.75, 42.08, 15.63, 15.79, 54.85, 17.45, 6.72, 32.03, 14.7, 15.21, 4.4, 5.51, 13.06, 24.61, 6.05, 21.62, 22.23, 9.39, 27.28, 12.94, 11.32, 6.91, 22.02, 13.79, 29.41, 6.77, 12.82, 11.56, 28.96, 22.21, 13.01, 6.88, 32.27, 9.4, 10.92, 14.24, 8.64, 4.35, 107.59, 22.07, 7.59, 39.87, 115.33, 29.96, 10.15, 6.31, 20.73, 17.77, 22.11, 5.88, 7.96, 4.44, 6.49, 12.1, 7.95, 8.48, 5.67, 17.19, 6.79, 5.67, 20.97, 14.08, 7.73, 18.06, 6.67, 9.0, 7.47, 15.46, 113.06, 70.8, 10.96, 7.91, 10.96, 14.73, 4.79, 8.71, 5.09, 7.57, 7.32, 81.68, 19.56, 29.19, 41.93, 11.35, 37.06, 24.0, 5.03, 9.38, 30.37, 10.81, 69.97, 3.87, 3.32, 30.37, 8.61, 2.73, 30.72, 5.56, 6.06, 24.31, 8.13, 7.63, 8.74, 26.49, 7.27, 10.64, 5.38, 86.71, 44.94, 32.17, 15.58, 12.44, 29.32, 7.69, 7.29, 15.86, 8.79, 10.81, 6.43, 15.49, 16.72, 12.86, 28.73, 8.45, 12.68, 6.8, 14.68, 24.77, 6.06, 20.27, 12.35, 11.81, 10.16, 5.33, 3.72, 18.52, 19.61, 12.46, 6.13, 13.49, 8.7, 19.63, 9.8, 20.47, 5.38, 6.81, 6.57, 4.74, 28.75, 19.9, 5.9, 14.98, 5.9, 36.65, 6.06, 13.85, 16.38, 35.72, 5.3, 17.93, 14.81, 105.49, 276.46, 22.37, 19.78, 29.11, 30.37]
    gnv = [1, 4, 6, 8, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40, 42, 46, 49, 54, 55, 56, 59, 61, 62, 63, 65, 66, 69, 70, 72, 73, 74, 76, 77, 80, 81, 85, 86, 87, 89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110]
    lnv = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 66, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    bft = [1 2; 1 3; 4 5; 3 5; 5 6; 6 7; 5 8; 4 11; 5 11; 11 12; 2 12; 3 12; 7 12; 11 13; 12 14; 13 15; 14 15; 12 16; 15 17; 16 17; 17 18; 18 19; 19 20; 15 19; 20 21; 21 22; 22 23; 23 24; 23 25; 25 26; 25 27; 27 28; 28 29; 17 30; 8 30; 26 30; 17 31; 29 31; 23 32; 31 32; 27 32; 15 33; 19 34; 35 36; 35 37; 33 37; 34 36; 34 37; 37 38; 37 39; 37 40; 30 38; 39 40; 40 41; 40 42; 41 42; 43 44; 34 43; 44 45; 45 46; 46 47; 46 48; 47 49; 42 49; 45 49; 48 49; 49 50; 49 51; 51 52; 52 53; 53 54; 49 54; 54 55; 54 56; 55 56; 56 57; 50 57; 56 58; 51 58; 54 59; 56 59; 55 59; 59 60; 59 61; 60 61; 60 62; 61 62; 61 64; 38 65; 64 65; 49 66; 62 66; 62 67; 65 66; 66 67; 65 68; 47 69; 49 69; 68 69; 69 70; 24 70; 70 71; 24 72; 71 72; 71 73; 70 74; 70 75; 69 75; 74 75; 76 77; 69 77; 75 77; 77 78; 78 79; 77 80; 79 80; 77 82; 82 83; 83 84; 83 85; 84 85; 85 86; 85 88; 85 89; 88 89; 89 90; 90 91; 89 92; 91 92; 92 93; 92 94; 93 94; 94 95; 80 96; 82 96; 94 96; 80 97; 80 98; 80 99; 92 100; 94 100; 95 96; 96 97; 98 100; 99 100; 100 101; 92 102; 101 102; 100 103; 100 104; 103 104; 103 105; 100 106; 104 105; 105 106; 105 107; 105 108; 106 107; 108 109; 103 110; 109 110; 63 110; 17 81; 32 81; 9 32; 10 27; 9 10; 68 87; 75 111; 76 111; 59 64; 68 80]
    return R, gnv, lnv, bft # RateA, generator_node_vector, load_node_vector, branch_from_to
end;
function ı_at(n) # return the power injection Given a node index
    i = findfirst(x -> x == n, gnv) # injection index, NOT a generator index
    b = isnothing(i)
    if b
        return !b # no injection
    else
        return ı[i] # a VariableRef
    end
end;
function w_at(n) # return the power withdrawal Given a node index
    l = findfirst(x -> x == n, lnv) # load index
    b = isnothing(l)
    if b
        return !b # no injection
    else
        return LD[l] # load data
    end
end;
R, gnv, lnv, bft = load_111_data(); # ✅ `gnv` and `lnv` have the agreeable property---strictly increasing
A = bft_2_A(bft); R_n = transpose(A .!= 0) * R; # capacity of a node
R_gnv, R_lnv = R_n[gnv], R_n[lnv]; # capacity associated with gen_nodes and load_nodes
Nı, L = length(gnv), length(lnv);
B, N = size(A);

# The 111 node test network can be regarded as "established"
# Load is not full (few) initially
# But might increase in the next 15 years

sd_vec = [-4414725144922642859, 5047609688043979672, -1070745752371855362, 6110427412149971190] # seed vector
obj_vec = [95.90161379738188, 100.33793582360737, 1528.7522869179868, 1637.5530773395192]

GRB_ENV = Gurobi.Env();

ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)); JuMP.set_silent(ø)
JuMP.@variable(ø, -R[b] ≤ 𝐟[b = 1:B] ≤ R[b]);
JuMP.@variable(ø, ı[i = 1:Nı] >= 0); # power injection at Nı-nodes
JuMP.@objective(ø, Min, sum(ı))

# TODO resolve the power injection into power from generators

Random.seed!(sd_vec[1])
LD = rand(0.03:7e-15:1) * lrand(L) .* R_lnv; # randomly generate a load vector as an instance
c = JuMP.@constraint(ø, [n in 1:N], ı_at(n) + dot( 𝐟 , view(A, :, n)) == w_at(n)); # 📘 KCL
JuMP.optimize!(ø)
if JuMP.termination_status(ø) == JuMP.OPTIMAL
    @info JuMP.objective_value(ø)
end



import PowerModels
PowerModels.silence(); ⅁ = PowerModels.make_basic_network(PowerModels.parse_file("data/case118.m"));
nvg = [⅁["gen"]["$g"]["gen_bus"] for g in 1:length(⅁["gen"])]; # 🟠 a bus_index vector that has generators
nvl = [⅁["load"]["$l"]["load_bus"] for l in 1:length(⅁["load"])]; # 🟠 a bus index vector that has loads
A = -PowerModels.calc_basic_incidence_matrix(⅁);
R = [⅁["branch"]["$b"]["rate_a"] for b in 1:size(A, 1)]; # 🟠


# JuMP.value.(ı)
# JuMP.value.(𝐟)


# function max_support_power(n, 𝐑 , A) return sum(𝐑[SparseArrays.findnz(view(A, :, n))[1]]) end
# function load_demand_data(N, 𝐑 , A, LiR)
#     f = n -> max_support_power(n, 𝐑 , A)
#     return 𝐋 = SparseArrays.SparseVector(N, Vector(LiR), f.(LiR))
# end
# function p_at(n)
#     n in DiR && return p[n]  # ⚠️ Note that we've used n=g
#     return false
# end

# LiR = 4:6 # load nodes' index range
# DiR = 1:3 # Decision nodes' (i.e. Generator power output) index range
# G = 3
# 𝐆 = [15, 12, 14] # max output power of g-th generator
# B, N, 𝐑 , A = load_network_data()
# 𝐋 = .7 * load_demand_data(N, 𝐑 , A, LiR)

# println(JuMP.termination_status(ø))
# JuMP.value.(f)
# pt = JuMP.value.(p)
# sum(pt)

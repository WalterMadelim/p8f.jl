# import PowerModels
import LinearAlgebra
import Distributions
import Random
import Gurobi
import JuMP
using Logging
## By the wind curtail - generation curtail - load shedding system we have RCR and can enforce the power balance equation.
@enum State begin
    t1
    t2f
    t3
    t2b
end

# Unit Commitment skeleton
# OPTIMAL VALUE = 5.87190287
# 13/11/24

global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function norm1(x) return LinearAlgebra.norm(x, 1) end
function ip(x, y) return LinearAlgebra.dot(x, y) end
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        # JuMP.set_attribute(ø, "QCPDual", 1)
        # vio = JuMP.get_attribute(ø, Gurobi.ModelAttribute("MaxVio")) 🍀 we suggest trial-and-error to decide if a constr is tight, rather than using this cmd
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
function load_data()
    # network_data = PowerModels.parse_file("data/case6ww.m")
    # basic_net_data = PowerModels.make_basic_network(network_data)
    # F = PowerModels.calc_basic_ptdf_matrix(basic_net_data)
    # S_BASE = 100 # MVA
    T, G, W, L = 8, 3, 2, 3
    F = let
        [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    end
    B, N = size(F)
    Bℷ = Dict(
        "f" =>  Int[1,1,1,2,2,2,2,3,3,4,5],
        "t" =>  Int[2,4,5,3,4,5,6,5,6,5,6],
        "BC" => [6.803707434852969, 5.306275635065887, 4.628677110801281, 3.2842185358904556, 5.997601724860079, 1.5388456812040012, 2.2162912431269106, 3.3613002359967883, 4.481394762233808, 1.1612717404468085, 2.1327646654597934]
    )
    Wℷ = Dict(
        "id" => Int[1, 2],
        "n" => Int[2, 3],
        "CW" => 200. * ones(T, W)
    )
    Lℷ = Dict(
        "id" => Int[1,2,3],
        "n" => Int[4,5,6],
        "CL" => [8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 16.0 13.776 14.443],
        "M" => [4, 3.5, 3]
    )
    Gℷ = Dict(
        "id" => Int[1, 2, 3],
        "n" => Int[1, 2, 3],
        "ZS" => [1., 0, 0],
        "ZP" => [0.5, 0, 0], # Pzero, consistent with ZS
        "C2" => [23.3, 20.1, 18.5], # this quadratic cost is not prominent
        "C1" => [166.9, 180.2, 150.3],
        "C0" => [113.1, 40.2, 20.1],
        "CR" => [83., 74, 77],
        "CG" => [4.0, 3.4, 3.6],
        "CST" => [0.63, 0.60, 0.72], # ✅
        "CSH" => [0.15, 0.15, 0.15], # ✅
        "PI" => [0.5, 0.375, 0.45],
        "PS" => [4.5, 4, 5.5],
        "RU" => [.6, .6, .6] * 10,
        "SU" => [.6, .6, .6] * 10,
        "RD" => [.6, .6, .6] * 10,
        "SD" => [.6, .6, .6] * 10,
        "UT" => Int[2, 2, 2],
        "DT" => Int[2, 2, 2]
    )
    Gℷ = merge(Gℷ, Dict("M" => [Gℷ["C2"][g] * Gℷ["PS"][g]^2 + Gℷ["C1"][g] * Gℷ["PS"][g] + Gℷ["C0"][g] for g in 1:G]))
    SRD = 1.5 # system reserve demand
    PE = 0.
    yM, MY = let
        [3.05859962924333 0.22129314909074682; 0.31753646450048895 0.2888728744013412; 0.27256656877799856 0.29547258378275026; 0.23910049918737447 0.29184920087643973; 0.2618263982426271 0.21039631467919426; 0.26299411377195747 0.31786708911852707; 0.28175032000925565 0.27988140319606497; 0.3623420214378466 0.22988918082592985;;; 0.3270142614850655 0.2437922786722481; 3.2216666535358964 0.25230084646797907; 0.21077310404968044 0.2228459244541164; 0.29720887486472636 0.2701304595146522; 0.35027456634255405 0.3074463999534698; 0.28675747094003773 0.2582064181512781; 0.2968194724538678 0.28495037759305053; 0.21139582361359377 0.15883249966262034;;; 0.30893982261630343 0.23635489823809003; 0.23766856090340874 0.29275689095104507; 3.2738947421131877 0.30470591369330413; 0.30130579694679144 0.3106466979614247; 0.11492286406562156 0.2110580861494358; 0.31480802584939943 0.2645081523107029; 0.30071445370031447 0.2208679426775983; 0.37525257752802843 0.24475842137879997;;; 0.2674523493864944 0.3119056042295227; 0.31608292807926985 0.22314510875734475; 0.29328439330760653 0.28889060619463297; 3.5764098154049586 0.34346332783825245; 0.2768493752224806 0.28427697285355014; 0.3752931902482649 0.22936086459450508; 0.3061686555722745 0.21354403990817256; 0.27170781142993383 0.053920992052867864;;; 0.2717574415308641 0.31269065480917796; 0.3507278126462144 0.21290953591222375; 0.08848065351555368 0.2640410081018837; 0.2584285683115975 0.2282111920693616; 3.6193300903600685 0.2694961113987438; 0.24444250159946096 0.1996148649554183; 0.26417635173347664 0.29121826303263615; 0.2460955090764697 0.2159730638603265;;; 0.27489650210676875 0.1460037040478942; 0.2891820622902724 0.27424927056837345; 0.29033716034590573 0.35744342175575483; 0.35884372838395623 0.3521243145218948; 0.24641384664603508 0.2875109035930402; 3.2936532197996535 0.27296928504877144; 0.37289850173526257 0.21230948138150044; 0.20244933510854768 0.2036038927478527;;; 0.2932060031681682 0.20805322909527263; 0.29879735862820367 0.26369044329179064; 0.27579688302092215 0.2551060521631845; 0.2892724885320672 0.2879276128215445; 0.2657009916041522 0.22744572825072373; 0.37245179655936383 0.3005439916935925; 3.00544670967344 0.24995680351117416; 0.24298777849540484 0.24301231613239627;;; 0.3930645195856088 0.14779185673282344; 0.2326405247767792 0.24898947586903486; 0.36960182183748574 0.27094715569976885; 0.274078459378576 0.25245375887789573; 0.2668869639359949 0.20636736421419996; 0.2212694449214986 0.2090490078831887; 0.26225459348425445 0.26191891309439796; 3.591212106778445 0.21276299008226401;;; 0.2617508049573694 3.1148472129588427; 0.274772137554294 0.230861039485004; 0.24043930026640756 0.24442340753418412; 0.32401140989702526 0.12893271294086822; 0.3432172673875635 0.33076897305755537; 0.1745589715797054 0.24359433389327995; 0.2370552018029826 0.23643289560639155; 0.15752701445168366 0.22655744179516168;;; 0.327501876602911 0.22903238581995147; 0.2814520516849723 3.239501573505252; 0.29501263931431 0.2963919443857751; 0.23342226075979455 0.3026989763582024; 0.24160749482555655 0.2529212801262277; 0.300975884435132 0.2692718067754279; 0.29086376233444794 0.20413093610958188; 0.2568959799228425 0.2221607401113527;;; 0.3182985424933479 0.22679171037815934; 0.23619408618013743 0.28058890089480293; 0.2911586185655969 2.97297833854534; 0.28336471470611063 0.34932768842856543; 0.2769359235242444 0.14217521335870598; 0.36836699213154117 0.21017326860603525; 0.2664763277148695 0.26496815968706017; 0.26305061626260434 0.20014492265130654;;; 0.3095240358767111 0.10614989207451718; 0.2783274975303469 0.28174480915690403; 0.2919482791233911 0.3441765647182391; 0.3327863126394038 3.0623795357447046; 0.23595498378139604 0.1547047856926807; 0.35789676118735486 0.2705385910014243; 0.2941467646629033 0.29883050722098853; 0.2394060957304048 0.20946338535789719;;; 0.2542653631156345 0.3341803656273732; 0.34183765140533334 0.25816132636109806; 0.21855388074757107 0.16321830308454843; 0.29979417109087036 0.18089899912884955; 0.303434116546947 3.219888185753861; 0.3194775636946691 0.2751678473324358; 0.2598590935282514 0.23093734481226902; 0.2195139145028779 0.3022244344090633;;; 0.36258471452591495 0.24785430343404524; 0.2934462465740891 0.27536042998124566; 0.2728525238797856 0.2320649353028252; 0.24572663980277282 0.2975813814085406; 0.23440144707456897 0.2760164243033833; 0.3057845221213479 3.331975696311318; 0.3338059339420677 0.22986053709744453; 0.22304413514281418 0.262985324648876;;; 0.30473053259902405 0.2208243691427282; 0.300321710011433 0.19035106331097112; 0.20934381824225248 0.2669913303794216; 0.21004131911201157 0.30600480162367627; 0.3061363491473582 0.2119174257787879; 0.22525622244964824 0.2099920410930159; 0.2633502497552206 2.9623619636450154; 0.2560455443495948 0.2427313815310452;;; 0.2812688515815907 0.2374794566842001; 0.20073437343370445 0.23491140866544363; 0.2597648382961558 0.22869863469636953; 0.07694881260940858 0.24316822111328662; 0.25742169132775017 0.30973505672828394; 0.24308117516870226 0.26964736999714906; 0.2829363037291445 0.2692619228837469; 0.23342016269016258 3.1506100778778623;;; 0.0 0.2205983125019094; 0.27199141480366285 0.1906090233476236; 0.13021288989441138 0.20331836742947548; 0.2842829742794221 0.2697917487245768; 0.3397981495029214 0.31193655773381984; 0.2823443504823441 0.176234826087362; 0.2630561015217323 0.22723031908379843; 0.19054805192654278 0.18642541548339175;;; 0.20848714805609664 0.2444063576415555; 0.0 0.2928550834068516; 0.2901051783198936 0.2921797780560889; 0.253472446169264 0.26244226782355234; 0.22642530045139583 0.28229010841320307; 0.23892219968352346 0.17162614725420436; 0.29693471674331573 0.24880800990415988; 0.3327042114543252 0.27476556835581545;;; 0.22992528603728385 0.26405398163803984; 0.291912617203835 0.25751402122238315; 0.0 0.2498254247374867; 0.30519924547217586 0.2421182586676962; 0.42292517610598757 0.3214464416653418; 0.23021411441930376 0.189281906616853; 0.24079393562318688 0.33826848811720733; 0.03195340986618991 0.26178542081349204;;; 0.4098418076830045 0.15624663803685584; 0.20683214015664458 0.3755903448958537; 0.2248970784246649 0.26828553676774564; 0.0 0.1990621344641028; 0.2937579254378023 0.2083458781918536; 0.20409714951575408 0.27640041241730207; 0.21069884499254826 0.3388413400329392; 0.24013411627972875 0.3986402187239855;;; 0.3054986508390967 0.15682136310013758; 0.19657955018218815 0.35014302969979666; 0.4259682760379778 0.26407769827213423; 0.2740162537029863 0.4099158520733623; 0.0 0.26421345555130177; 0.3704322441422444 0.3026931941236391; 0.3203652848542024 0.17013128213377726; 0.25538708941959143 0.27963527253042636;;; 0.2795751351292254 0.38166076182615627; 0.3047446683034931 0.22203356565301804; 0.22496425339992857 0.20345251345065435; 0.17592725628195782 0.19481296048040284; 0.4085627555787813 0.2770057102348909; 0.0 0.20270074830675838; 0.20190532227659966 0.2684310027482865; 0.4470927220314157 0.26373614363422115;;; 0.2996405133659934 0.25472396295341576; 0.23435216072799603 0.22048926994861257; 0.2905293416926053 0.24133591193957094; 0.30447520498024 0.2996592507703861; 0.2787615343971066 0.2621660522297694; 0.22034728601756665 0.19131812205139612; 0.0 0.3217723661475461; 0.23124465789484494 0.29423566421241615;;; 0.17642221094770533 0.36778697319770937; 0.3559273341854472 0.20630640935770006; 0.1901321790191478 0.2611522239150567; 0.3342842871414502 0.3231649263722015; 0.3664131773402556 0.26488346843033844; 0.32003332244441246 0.35434297152078814; 0.3635335962357172 0.25686856918196754; 0.0 0.2513901877315536;;; 0.3152253315870693 0.0; 0.29459020995830515 0.22260125025633778; 0.38539271444372075 0.312037223364905; 0.1877984806701446 0.44473364043047015; 0.28032352897336005 0.23381039872251003; 0.39208414615383047 0.19587197862101377; 0.29767475194908877 0.24358146991355167; 0.383647542334475 0.2408350731059668;;; 0.20529760222814916 0.2601884375745854; 0.2442883254546317 0.0; 0.17269424006133216 0.19815765990720893; 0.32017417164073486 0.257276524619968; 0.32879605257898126 0.23335052613373267; 0.2638385438314035 0.20889464047719322; 0.18810546351924876 0.3192780875837021; 0.26959769819660584 0.2716184254998685;;; 0.2676208503307123 0.3187295199544548; 0.3536893292281777 0.19133832652300806; 0.2579733583908312 0.0; 0.3091152375175884 0.20598822321913973; 0.2710194456106294 0.3873698143896652; 0.184423645596558 0.2642055898783238; 0.2424367727133554 0.2613726266226804; 0.3206002169378032 0.28621734168506746;;; 0.24937940799233618 0.3304345774538696; 0.30896143385607805 0.1790549437359309; 0.21089994230089792 0.27575606855376605; 0.21280860943198932 0.0; 0.3300316931721543 0.36247791245733985; 0.17950324571257817 0.15784142786450156; 0.25195116228308423 0.20362853941848427; 0.2548658659701156 0.27431087009502514;;; 0.2524545174840982 0.21981791337159676; 0.21449139711100362 0.25601819862015834; 0.26841318964643535 0.42711144485034064; 0.253984786175963 0.34255929993636913; 0.25668392849475646 0.0; 0.27390798618603124 0.28111162002209533; 0.3293624894761731 0.327008288413846; 0.36759525821131556 0.15955355357442538;;; 0.2253628270288189 0.2575367749731632; 0.29488426965210357 0.26134686863348194; 0.2574636791435125 0.3535252952896456; 0.2660638105698949 0.10918940695134041; 0.35431963290010443 0.18086109366301267; 0.19610453650530657 0.0; 0.2924830491750049 0.34613381434067925; 0.3150997523189212 0.2394800866951381;;; 0.2669927354648018 0.2648321348951373; 0.2978988298885863 0.26998129148037; 0.31214794721860584 0.2488989005280773; 0.3034388003671083 0.2952083426379613; 0.1613815226632105 0.2881744779667845; 0.33911303032522044 0.2728253995620948; 0.31397893378828684 0.0; 0.3399477072428517 0.20234479090275603;;; 0.3481136599518056 0.20628459985644343; 0.4009250069028098 0.23594682544599205; 0.2666843277965364 0.21526567513382255; 0.45339440966123584 0.32230839722878957; 0.3139790935591275 0.20426511302680414; 0.30825451976317836 0.22109685422563555; 0.279734640421502 0.311793103851355; 0.24234810775272092 0.0], [0.2767427835484444 0.25774469422226304; 0.28105102750237376 0.24083979895680727; 0.2595149062305072 0.2618425548212915; 0.2625761272079569 0.26947622779521374; 0.26999888616058937 0.24738582314805127; 0.28055551906984305 0.24694412275060976; 0.27689914681373523 0.2708880025205796; 0.2599342017075787 0.2512244910308245]
    end
    NY = size(yM, 3)
    return T, B, G, W, L, F, SRD, Bℷ, Gℷ, Wℷ, Lℷ, PE, MY, yM, NY
end

Δβ, βnm1V = let # 🍀 hyperparameter on exploit & explore
    Δβ = 1.0 # if you have numerical issues reported from Gurobi, you had best tune Δβ down to encourage more explore, but small Δβ might be slow to find valid lb
    βnm1V = 0.0 : Δβ : 1e3
    Δβ, βnm1V
end
T, B, G, W, L, F, SRD, Bℷ, Gℷ, Wℷ, Lℷ, PE, MY, yM, NY = load_data()
Random.seed!(8544002554851888986)
MZ = let
    Dload = Distributions.Arcsine.(Lℷ["M"])
    vec = [rand.(Dload) for t in 1:T]
    [vec[t][l] for t in 1:T, l in 1:L]
end
u, v, x, Y, Z, x1, x2, β1, β2 = let
    u, v, x = 1. * rand(Bool, T, G), 1. * rand(Bool, T, G), 1. * rand(Bool, T, G)
    x1 = u, v, x
    β1, Y = rand(T, W), rand(T, W)
    x2 = x1, Y
    β2, Z = rand(T, L), rand(T, L)
    u, v, x, Y, Z, x1, x2, β1, β2
end
ℶ1, ℶ2, Δ1, Δ2 = let
    ℶ1 = Dict(
        "st" => Bool[],
        "x" =>  typeof(x1)[],
        "rv" => typeof(Y)[],
        "cn" => Float64[],
        "px" => typeof(x1)[],
        "pβ" => typeof(β1)[]
    )
    ℶ2 = Dict(
        "st" => Bool[],
        "x" =>  typeof(x2)[],
        "rv" => typeof(Z)[],
        "cn" => Float64[],
        "px" => typeof(x2)[],
        "pβ" => typeof(β2)[]
    )
    Δ1 = Dict(
        "f" => Float64[],
        "x" => typeof(x1)[],
        "β" => typeof(β1)[]
    )
    Δ2 = Dict(
        "f" => Float64[],
        "x" => typeof(x2)[],
        "β" => typeof(β2)[]
    )
    ℶ1, ℶ2, Δ1, Δ2
end
function pushCut(D, x, rv, cn, px, pβ) return (push!(D["st"], true); push!(D["x"], x); push!(D["rv"], rv); push!(D["cn"], cn); push!(D["px"], px); push!(D["pβ"], pβ)) end
function pushSimplicial(D, f, x, β) return (push!(D["f"], f); push!(D["x"], x); push!(D["β"], β)) end
        
function master() # initialization version
    ø = JumpModel(0)
    if true
        JuMP.@variable(ø, u[t = 1:T, g = 1:G], Bin) # 🌴 if integer is relaxed, unexpected outcomes might occur due to vague physical problem
        JuMP.@variable(ø, v[t = 1:T, g = 1:G], Bin)
        JuMP.@variable(ø, x[t = 1:T, g = 1:G], Bin)
        JuMP.@constraint(ø, [g = 1:G],          x[1, g] - Gℷ["ZS"][g] == u[1, g] - v[1, g])
        JuMP.@constraint(ø, [t = 2:T, g = 1:G], x[t, g] - x[t-1, g]   == u[t, g] - v[t, g])
        JuMP.@expression(ø, o1, sum(Gℷ["CST"][g] * u[t, g] + Gℷ["CSH"][g] * v[t, g] for t in 1:T, g in 1:G)) # 1st stage objective here
    end
    JuMP.@variable(ø, -1 <= β1[t = 1:T, w = 1:W] <= 1)
    JuMP.@objective(ø, Min, o1 + ip(MY, β1))
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    @assert status == JuMP.OPTIMAL " in master0(), $status "
    x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
    β1 = JuMP.value.(β1)
    return x1, β1
end
function master(masterIsMature, iCnt, R2, cnV2, px1V2, pβ1V2, stV2) # enforcing boundedness version
    ø = JumpModel(0)
    if true
        JuMP.@variable(ø, u[t = 1:T, g = 1:G], Bin) # 🌴 if integer is relaxed, unexpected outcomes might occur due to vague physical problem
        JuMP.@variable(ø, v[t = 1:T, g = 1:G], Bin)
        JuMP.@variable(ø, x[t = 1:T, g = 1:G], Bin)
        JuMP.@constraint(ø, [g = 1:G],          x[1, g] - Gℷ["ZS"][g] == u[1, g] - v[1, g])
        JuMP.@constraint(ø, [t = 2:T, g = 1:G], x[t, g] - x[t-1, g]   == u[t, g] - v[t, g])
        JuMP.@expression(ø, o1, sum(Gℷ["CST"][g] * u[t, g] + Gℷ["CSH"][g] * v[t, g] for t in 1:T, g in 1:G)) # 1st stage objective here
    end
    JuMP.@variable(ø, β1[t = 1:T, w = 1:W])
        (bind = iCnt[1]; βnormBnd = βnm1V[bind])
        bind == length(βnm1V) && error(" in master(): enlarge the scale of βnm1V please. ")
        JuMP.@variable(ø, aβ[eachindex(eachrow(β1)), eachindex(eachcol(β1))])
        JuMP.@constraint(ø, aβ .>=  β1)
        JuMP.@constraint(ø, aβ .>= -β1)
        JuMP.@constraint(ø, sum(aβ) <= βnormBnd)
    JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, o2 >= ip(MY, β1))
    JuMP.@variable(ø, o3)
        for r in 1:R2
            stV2[r] && JuMP.@constraint(ø, o3 >= cnV2[r] + ip(px1V2[r], (u, v, x)) + ip(pβ1V2[r], β1))
        end
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    @assert status == JuMP.OPTIMAL " in master1(), $status "
    lb, x1, β1, cost1plus2, o3 = let
        lb = JuMP.objective_value(ø)
        x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
        β1 = JuMP.value.(β1)
        cost1plus2 = JuMP.value(o1) + JuMP.value(o2)
        o3 = JuMP.value(o3)
        lb, x1, β1, cost1plus2, o3
    end
    (norm1(β1) > βnormBnd - Δβ/3) ? (iCnt[1] += 1) : (masterIsMature[1] = true)
    return lb, x1, β1, cost1plus2, o3
end
function master(R2, cnV2, px1V2, pβ1V2, stV2) # final version
    ø = JumpModel(0)
    if true
        JuMP.@variable(ø, u[t = 1:T, g = 1:G], Bin)
        JuMP.@variable(ø, v[t = 1:T, g = 1:G], Bin)
        JuMP.@variable(ø, x[t = 1:T, g = 1:G], Bin)
        JuMP.@constraint(ø, [g = 1:G],          x[1, g] - Gℷ["ZS"][g] == u[1, g] - v[1, g])
        JuMP.@constraint(ø, [t = 2:T, g = 1:G], x[t, g] - x[t-1, g]   == u[t, g] - v[t, g])
        JuMP.@expression(ø, o1, sum(Gℷ["CST"][g] * u[t, g] + Gℷ["CSH"][g] * v[t, g] for t in 1:T, g in 1:G)) # 1st stage objective here
    end
    JuMP.@variable(ø, β1[t = 1:T, w = 1:W])
    JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, o2 >= ip(MY, β1))
    JuMP.@variable(ø, o3)
        for r in 1:R2
            stV2[r] && JuMP.@constraint(ø, o3 >= cnV2[r] + ip(px1V2[r], (u, v, x)) + ip(pβ1V2[r], β1))
        end
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    @assert status == JuMP.OPTIMAL " in masterFinal(), $status "
    lb, x1, β1, cost1plus2, o3 = let
        lb = JuMP.objective_value(ø)
        x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
        β1 = JuMP.value.(β1)
        cost1plus2 = JuMP.value(o1) + JuMP.value(o2)
        o3 = JuMP.value(o3)
        lb, x1, β1, cost1plus2, o3
    end
    return lb, x1, β1, cost1plus2, o3
end
function master(masterIsMature, iCnt, ℶ1) # portal
    R2, cnV2, px1V2, pβ1V2, stV2 = let
        cnV2, px1V2, pβ1V2, stV2 = ℶ1["cn"], ℶ1["px"], ℶ1["pβ"], ℶ1["st"]
        length(cnV2), cnV2, px1V2, pβ1V2, stV2
    end
    if masterIsMature[1]
        lb, x1, β1, cost1plus2, o3 = master(R2, cnV2, px1V2, pβ1V2, stV2) # 3️⃣
        return true, lb, x1, β1, cost1plus2, o3
    else
        if R2 >= 1
            lb, x1, β1, cost1plus2, o3 = master(masterIsMature, iCnt, R2, cnV2, px1V2, pβ1V2, stV2) # 2️⃣
            return false, lb, x1, β1, cost1plus2, o3
        else
            x1, β1 = master() # 1️⃣
            return false, -Inf, x1, β1, NaN, -Inf
        end
    end
end
function eval_Δ_at(Δ, x, β) # t1, in termination criterion
    isempty(Δ["f"]) && return Inf
    fV, xV, R2, βV = Δ["f"], Δ["x"], length(Δ["f"]), Δ["β"]
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G], sum(xV[r][i][t, g] * λ[r] for r in 1:R2) == x[i][t, g])
    JuMP.@constraint(ø, [t = 1:T, w = 1:W],          sum(βV[r][t, w]    * λ[r] for r in 1:R2) ==    β[t, w])
    JuMP.@objective(ø, Min, ip(fV, λ))
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.INFEASIBLE
            return Inf
        else
            error(" in eval_Δ_at(): $status ")
        end
    else
        return JuMP.objective_value(ø)
    end
end
function eval_φ1ub(x1, β1) # t2: a max-min problem to choose worst Y
    function eval_xby(x1, β1, Y)
        R2, fV, x2V, β2V = let
            fV, x2V, β2V = Δ2["f"], Δ2["x"], Δ2["β"]
            length(fV), fV, x2V, β2V
        end
        @assert R2 >= 1
        const_obj = -ip(β1, Y)
        ø = JumpModel(0)
        JuMP.@variable(ø, λ[1:R2] >= 0.)
        JuMP.@constraint(ø, sum(λ) == 1.)
        JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
        JuMP.@constraint(ø, [t = 1:T, l = 1:L],             sum(β2V[r][t, l]       * λ[r] for r in 1:R2) ==    β2[t, l])
        # following 2 lines are minimum infeas. system ############################################################
        JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G],    sum(x2V[r][1][i][t, g] * λ[r] for r in 1:R2) == x1[i][t, g])
        JuMP.@constraint(ø, [t = 1:T, w = 1:W],             sum(x2V[r][2][t, w]    * λ[r] for r in 1:R2) ==     Y[t, w])
        ###########################################################################################################
        JuMP.@objective(ø, Min, ip(MZ, β2) + ip(fV, λ)) # problem \bar{ψ}(x1, Y)
        (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        if status != JuMP.OPTIMAL
            if status == JuMP.INFEASIBLE_OR_UNBOUNDED
                JuMP.set_attribute(ø, "DualReductions", 0)
                (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
            end
            if status == JuMP.INFEASIBLE
                return Inf
            else
                error(" in eval_xby #END , status = $status ")
            end
        else
            return const_obj + JuMP.objective_value(ø)
        end
    end
    if isempty(Δ2["f"]) # only once
        return (φ1ub, index) = (Inf, 1)
    end
    normalVec = zeros(NY)
    for i in 1:NY
        φ1x1b1y = eval_xby(x1, β1, yM[:, :, i])
        if φ1x1b1y == Inf
            return (φ1ub, index) = (Inf, i)
        else
            normalVec[i] = φ1x1b1y
        end
    end
    return (φ1ub, index) = findmax(normalVec)
end
function psi(iCnt, x1, Y, ℶ2) # t2f, 
    x2 = x1, Y # 🍀 this is fixed
    R2, cnV2, px2V2, pβ2V2 = let
        cnV2, px2V2, pβ2V2 = ℶ2["cn"], ℶ2["px"], ℶ2["pβ"]
        length(cnV2), cnV2, px2V2, pβ2V2
    end
    ø = JumpModel(0)
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
        (bind = iCnt[1]; βnormBnd = βnm1V[bind])
        bind == length(βnm1V) && error(" in psi(): enlarge the scale of βnm1V please. ")
        JuMP.@variable(ø, aβ[eachindex(eachrow(β2)), eachindex(eachcol(β2))])
        JuMP.@constraint(ø, aβ .>=  β2)
        JuMP.@constraint(ø, aβ .>= -β2)
        JuMP.@constraint(ø, sum(aβ) <= βnormBnd)
    if R2 == 0
        JuMP.@objective(ø, Min, ip(MZ, β2))
    else
        JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, [r = 1:R2], o2 >= cnV2[r] + ip(px2V2[r], x2) + ip(pβ2V2[r], β2))
        JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    end
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    @assert status == JuMP.OPTIMAL " in psi(): $status "
    β2 = JuMP.value.(β2)
    let
        bnm = norm1(β2)
        if bnm > βnormBnd - Δβ/3
            iCnt[1] += 1
        elseif bnm < (βnormBnd - Δβ) - Δβ/3
            iCnt[1] -= 1
        end
    end
    return x2, β2
end
function maximize_φ2_over_Z(x2, β2)
    function f_primal( x2, Z, ::Nothing ) # convex conic program
        JuMP.@variable(ø, p[t = 1:T, g = 1:G])       # generator power
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # generator cutback
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.) # wind curtail
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.) # load shedding
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w]) 
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G], p[t, g] >= Gℷ["PI"][g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G], Gℷ["PS"][g] * x[t, g] >= p[t, g])
        JuMP.@constraint(ø, Dbl[t = 1:T], sum(Z[t, :]) + sum(ϖ[t, :]) + sum(ϱ[t, :]) == sum(Y[t, :]) + sum(p[t, :]) + sum(ζ[t, :]))
        JuMP.@expression(ø, lscost, sum(Lℷ["CL"][t, l] * ζ[t, l] for t in 1:T, l in 1:L))
        JuMP.@expression(ø, gccost, sum(   Gℷ["CG"][g] * ϱ[t, g] for t in 1:T, g in 1:G))
        JuMP.@expression(ø, primobj, lscost + gccost)
    end
    function f_dual( x2, Z, ::Nothing ) # dual convex conic program
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dbl[t = 1:T])
        JuMP.@constraint(ø, p[t = 1:T, g = 1:G], Dps[t, g] - Dvr[t, g] - Dpi[t, g] + Dbl[t] == 0.)
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], Dvr[t, g] - Dbl[t] + Gℷ["CG"][g] >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] - Dbl[t] >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] + Dbl[t] + Lℷ["CL"][t, l] >= 0.)
        JuMP.@expression(ø, xobj, sum(x[t, g] * (Gℷ["PI"][g] * Dpi[t, g] - Gℷ["PS"][g] * Dps[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, Yobj, sum(Y[t, w] * ( Dbl[t] - Dvp[t, w]) for t in 1:T, w in 1:W))
        JuMP.@expression(ø, Zobj, sum(Z[t, l] * (-Dbl[t] - Dzt[t, l]) for t in 1:T, l in 1:L))
        JuMP.@expression(ø, dualobj, xobj + Yobj + Zobj)
    end
    (u, v, x), Y = x2
    ø = JumpModel(0)
    JuMP.@variable(ø, 0 <= Z[t = 1:T, l = 1:L] <= Lℷ["M"][l]) # support of Z
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dbl[t = 1:T])
        JuMP.@constraint(ø, p[t = 1:T, g = 1:G], Dps[t, g] - Dvr[t, g] - Dpi[t, g] + Dbl[t] == 0.)
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], Dvr[t, g] - Dbl[t] + Gℷ["CG"][g] >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] - Dbl[t] >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], Dzt[t, l] + Dbl[t] + Lℷ["CL"][t, l] >= 0.)
        JuMP.@expression(ø, xobj, sum(x[t, g] * (Gℷ["PI"][g] * Dpi[t, g] - Gℷ["PS"][g] * Dps[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, Yobj, sum(Y[t, w] * ( Dbl[t] - Dvp[t, w]) for t in 1:T, w in 1:W))
        JuMP.@expression(ø, Zobj, sum(Z[t, l] * (-Dbl[t] - Dzt[t, l]) for t in 1:T, l in 1:L))
        JuMP.@expression(ø, dualobj, xobj + Yobj + Zobj)
    JuMP.@objective(ø, Max, -ip(β2, Z) + dualobj)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" in maximize_φ2_over_Z(): $status ")
    else
        let
            vio = JuMP.get_attribute(ø, Gurobi.ModelAttribute("MaxVio"))
            vio > 5e-6 && @warn "vio = $vio"
        end
        return φ2, Z = JuMP.objective_value(ø), JuMP.value.(Z)
    end
end
function gen_cut_f_wrt_x2(x2, Z)
    ø = JumpModel(0)
    JuMP.@variable(ø, u[t = 1:T, g = 1:G])
    JuMP.@variable(ø, v[t = 1:T, g = 1:G])
    JuMP.@variable(ø, x[t = 1:T, g = 1:G])
    JuMP.@variable(ø, Y[t = 1:T, w = 1:W])
    JuMP.@constraint(ø, cpu[t = 1:T, g = 1:G], u[t, g] == x2[1][1][t, g])
    JuMP.@constraint(ø, cpv[t = 1:T, g = 1:G], v[t, g] == x2[1][2][t, g])
    JuMP.@constraint(ø, cpx[t = 1:T, g = 1:G], x[t, g] == x2[1][3][t, g])
    JuMP.@constraint(ø, cpY[t = 1:T, w = 1:W], Y[t, w] == x2[2][t, w])
        JuMP.@variable(ø, p[t = 1:T, g = 1:G])       # generator power
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # generator cutback
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.) # wind curtail
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.) # load shedding
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w]) 
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G], p[t, g] >= Gℷ["PI"][g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G], Gℷ["PS"][g] * x[t, g] >= p[t, g])
        JuMP.@constraint(ø, Dbl[t = 1:T], sum(Z[t, :]) + sum(ϖ[t, :]) + sum(ϱ[t, :]) == sum(Y[t, :]) + sum(p[t, :]) + sum(ζ[t, :]))
        JuMP.@expression(ø, lscost, sum(Lℷ["CL"][t, l] * ζ[t, l] for t in 1:T, l in 1:L))
        JuMP.@expression(ø, gccost, sum(   Gℷ["CG"][g] * ϱ[t, g] for t in 1:T, g in 1:G))
        JuMP.@expression(ø, primobj, lscost + gccost)
    JuMP.@objective(ø, Min, primobj) # JuMP.set_attribute(ø, "QCPDual", 1)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" in gen_cut_f_wrt_x2(): $status ")
    else
        px1 = JuMP.dual.(cpu), JuMP.dual.(cpv), JuMP.dual.(cpx)
        pY  = JuMP.dual.(cpY)
        px2 = px1, pY
        cn = JuMP.objective_value(ø) - ip(px2, x2)
        return cn, px2
    end
end
function gen_cut_for_ℶ1(x1, Y) # t2b
    ret = gen_cut_ψ_wrt_x1(x1, Y)
    if length(ret) == 1
        return ret
    else
        cn, px1 = ret
        pβ1 = -Y # 💡 this is fixed, and irrespective of 'β1'
        return cn, px1, pβ1
    end
end
function gen_cut_for_ℶ2(x2, Z)
    cn, px2 = gen_cut_f_wrt_x2(x2, Z)
    pβ2 = -Z # 💡 this is fixed, and irrespective of 'β2'
    return cn, px2, pβ2
end
function gen_cut_ψ_wrt_x1(x1Γ, Y)
    R2, cnV2, px2V2, pβ2V2 = let
        cnV2, px2V2, pβ2V2 = ℶ2["cn"], ℶ2["px"], ℶ2["pβ"]
        length(cnV2), cnV2, px2V2, pβ2V2
    end
    @assert R2 >= 1 " in gen_cut_ψ_wrt_x1: #R2 "
    ø = JumpModel(0)
    JuMP.@variable(ø, x1[i = 1:3, t = 1:T, g = 1:G]) # 'x1' as a part of x2
        JuMP.@constraint(ø, cp[i = 1:3, t = 1:T, g = 1:G], x1[i, t, g] == x1Γ[i][t, g]) # 🌴 paradigm
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L]) # 🍀 should be unconstrained to ensure the validity of the cut generated
    JuMP.@variable(ø, o2)
        JuMP.@constraint(ø, [r = 1:R2], o2 >= cnV2[r] + ip(px2V2[r], ((x1[1, :, :], x1[2, :, :], x1[3, :, :]), Y)) + ip(pβ2V2[r], β2))
    JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            return -Inf
        else
            error("in gen_cut_ψ_wrt_x1: $status")
        end
    else
        tmp = JuMP.dual.(cp)
        px1 = tmp[1, :, :], tmp[2, :, :], tmp[3, :, :]
        cn = JuMP.objective_value(ø) - ip(px1, x1Γ)
        return cn, px1
    end
end

masterCnt, psiCnt = [2], [2]
masterIsMature = falses(1)
x1V, β1V, x2V, β2V, ZV = let
    x1V, β1V = [x1], [β1]
    x2V, β2V = [x2], [β2] # 💡 (x1, β1) -> Y -> β2 is essential; 'x2' is NOT essential
    ZV = [Z]
    x1V, β1V, x2V, β2V, ZV
end
tV, termination_flag = [t1], falses(1)
while true
    ST = tV[1]
    if ST == t1 
        vldt, lb, x1, β1, cost1plus2, _ = master(masterIsMature, masterCnt, ℶ1)
        let
            φ1ub = eval_Δ_at(Δ1, x1, β1)
            ub = cost1plus2 + φ1ub # 🍀 ub(Δ1) is a valid upper bound of υ_MSDRO
            gap = abs(ub - lb) / max( abs(lb), abs(ub) )
            @info " t1: masterCnt[$(masterCnt[1])] ($vldt)lb = $lb | $ub = ub, gap = $gap"
            if gap < 0.0001
                @info " 😊 gap < 0.01%, thus terminate "
                termination_flag[1] = true
            end
        end
        x1V[1], β1V[1] = x1, β1
        tV[1] = t2f
    elseif ST == t2f 
            x1, β1 = x1V[1], β1V[1]
            _, index = eval_φ1ub(x1, β1)
            Y = yM[:, :, index]
        x2, β2 = psi(psiCnt, x1, Y, ℶ2) # 2️⃣ only a trial (x2, β2) is needed here
        x2V[1], β2V[1] = x2, β2
        tV[1] = t3
    elseif ST == t3
        x2, β2 = x2V[1], β2V[1]
        φ2, Z = maximize_φ2_over_Z(x2, β2) # final stage eval is precise, therefore φ2ub ≡ φ2
        cn, px2, pβ2 = gen_cut_for_ℶ2(x2, Z)
        let 
            lower = cn + ip(px2, x2) + ip(pβ2, β2)
            upper = φ2
            (lower - 0.001 > upper) && @error " final stage #1: lower = $lower | $upper = upper"
            relerr = abs(upper - lower) / max(abs(upper), abs(lower))
            relerr > 1e-5 && @error " final stage #2: lower = $lower | $upper = upper"
            ZV[1] = Z
            termination_flag[1] && break # 🌿🌿🌿🌿
        end
        pushSimplicial(Δ2, φ2, x2, β2) # ✅ Δ2 is a precise evaluation of φ2(x2, β2)
        pushCut(ℶ2, x2, Z, cn, px2, pβ2) # ✅ ℶ2 is a valid underestimator of    φ2(x2, β2)
        tV[1] = t2b
    elseif ST == t2b 
            x1, β1 = x1V[1], β1V[1]
            φ1ub, index = eval_φ1ub(x1, β1)
            Y = yM[:, :, index]
        ret = gen_cut_for_ℶ1(x1, Y)
        if length(ret) == 1
            tV[1] = t2f
        else
            pushCut(ℶ1, x1, Y, ret...) # ✅ ℶ1(ℶ2) is a valid underestimator of φ1(x1, β1)
            φ1ub < Inf && pushSimplicial(Δ1, φ1ub, x1, β1) # ⚠️ conditional update ✅ Δ1(Δ2) is a valid overestimator of φ1(x1, β1)
            tV[1] = t1
        end
    end
end

# after termination
u, v, x = x1V[1]
β1, β2 = β1V[1], β2V[1]
Y, Z = x2V[1][2], ZV[1]
v1 =  ip(MY, β1)
v2 = -ip(β1, Y)
v3 =  ip(MZ, β2)
v4 = -ip(β2, Z)
sum([v1, v2, v3, v4])

length(ℶ1["cn"])
length(ℶ2["cn"])
length(Δ1["f"])
length(Δ2["f"])

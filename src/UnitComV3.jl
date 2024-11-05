# import PowerModels # to parse Matpower *.m data files
import LinearAlgebra
import Distributions
import Random
import Gurobi
import JuMP
using Logging

# Y is PD transformed norm-1 uncertainty set
# Z is norm-inf uncertainty set
# solve the c2g function by vertex enum
# RDDP algorithm can converge
# feasibility cut is not needed in this test case
# 5/11/24

@enum State begin
    t1
    t2f
    t3
    t2b
end
global_logger(ConsoleLogger(Info))
GRB_ENV = Gurobi.Env()
function ip(x, y) return LinearAlgebra.dot(x, y) end
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        # JuMP.set_attribute(ø, "QCPDual", 1)
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
    T = 8
    W = 2
    L = 3
    F = [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    Bℷ = Dict(
        "f" =>  Int[1,1,1,2,2,2,2,3,3,4,5],
        "t" =>  Int[2,4,5,3,4,5,6,5,6,5,6],
        "BC" => [4,6,4,4,4,3,9,7,8,2,4]/10
    ) # lines
    Wℷ = Dict(
        "id" => Int[1, 2],
        "n" => Int[2, 3],
        "CW" => 1000. * ones(T, W)
    ) # wind
    Lℷ = Dict(
        "id" => Int[1,2,3],
        "n" => Int[4,5,6],
        "CL" => [4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 4800.0 4132.8 4332.8; 9600.0 8265.6 8665.6],
        "M" => [1, 1.2, 1]
    ) # load
    Gℷ = Dict(
        "id" => Int[1, 2, 3], # the 1st one is slack generator
        "n" => Int[1, 2, 3], # the 1st generator resides at bus 1
        "ZS" => [1., 0, 0],
        "ZP" => [0.5, 0, 0], # Pzero, consistent with IS
        "C2" => [53.3, 88.9, 74.1], # this quadratic cost is not prominent
        "C1" => [1166.9, 1033.3, 1083.3],
        "C0" => [213.1, 200, 240],
        "CR" => [83., 74, 77],
        "CST" => [210., 200, 240],
        "CSH" => [210., 200, 240],
        "PI" => [0.5, 0.375, 0.45],
        "PS" => [2, 1.5, 1.8],
        "RU" => [.6, .6, .6],
        "SU" => [.6, .6, .6],
        "RD" => [.6, .6, .6],
        "SD" => [.6, .6, .6],
        "UT" => Int[3, 3, 3],
        "DT" => Int[3, 3, 3]
    )
    G = length(Gℷ["n"])
    Gℷ = merge(Gℷ, Dict("M" => [Gℷ["C2"][g] * Gℷ["PS"][g]^2 + Gℷ["C1"][g] * Gℷ["PS"][g] + Gℷ["C0"][g] for g in 1:G]))
    @assert W == length(Wℷ["n"])
    @assert L == length(Lℷ["n"])
    B, N = size(F)
    SRD = 1.5 # system reserve demand
    PE = 1.2e3
    MY = [2.213942268387555, 2.24840822001899, 2.0761192498440577, 2.1006090176636554, 2.159991089284715, 2.2444441525587444, 2.215193174509882, 2.0794736136606295, 2.0619575537781043, 1.9267183916544581, 2.094740438570332, 2.15580982236171, 1.9790865851844102, 1.975552982004878, 2.1671040201646368, 2.009795928246596]
    MZ = [[0.7310,0.4814,0.6908,0.4326,0.1753,0.8567,0.8665,0.6107] [0.7010,0.5814,0.3908,0.1326,0.4153,0.7567,0.8565,0.5107] [0.2010,0.6814,0.1908,0.4326,0.8153,0.7567,0.6565,0.7107]]
    yM = [24.46879703394664 2.616114091880524 2.4715185809304274 2.1396187950919554 2.174059532246913 2.19917201685415 2.3456480253453456 3.1445161566848703 2.0940064396589553 2.620015012823288 2.546388339946783 2.476192287013689 2.034122904925076 2.9006777162073196 2.4378442607921924 2.2501508126527257 0.0 1.667897184448773 1.8394022882982708 3.278734461464036 2.4439892067127738 2.236601081033803 2.397124106927947 1.4113776875816426 2.5218026526965542 1.6423808178251933 2.1409668026456985 1.9950352639386895 2.0196361398727856 1.8029026162305513 2.1359418837184143 2.784909279614445; 2.5402917160039116 25.77333322828717 1.90134848722727 2.528663424634159 2.805822501169715 2.313456498322179 2.3903788690256293 1.8611241982142337 2.198177100434352 2.2516164134797783 1.8895526894410994 2.2266199802427753 2.7347012112426667 2.3475699725927126 2.402573680091464 1.6058749874696356 2.1759313184293028 0.0 2.33530093763068 1.6546571212531567 1.5726364014575052 2.437957346427945 1.8748172858239682 2.8474186734835776 2.356721679666441 1.9543066036370536 2.8295146338254216 2.4716914708486244 1.715931176888029 2.3590741572168286 2.3831906391086903 3.2074000552224784; 2.1805325502239885 1.6861848323974435 26.1911579369055 2.3462751464608522 0.7078452281244294 2.322697282767246 2.206375064167377 2.956814574699886 1.9235144021312605 2.36010111451448 2.329268948524775 2.335586232987129 1.7484310459805685 2.1828201910382847 1.6747505459380199 2.0781187063692466 1.041703119155291 2.3208414265591486 0.0 1.799176627397319 3.4077462083038226 1.7997140271994285 2.3242347335408424 1.5210574321531825 3.083141715549766 1.3815539204906573 2.0637868671266495 1.6871995384071834 2.1473055171714828 2.0597094331481 2.4971835777488467 2.133474622372291; 1.9128039934989958 2.377670998917811 2.4104463755743315 28.61127852323967 2.06742854649278 2.87074982707165 2.3141799082565377 2.192627675028608 2.592091279176202 1.8673780860783564 2.266917717648885 2.6622905011152302 2.398353368726963 1.9658131184221825 1.6803305528960926 0.6155905008752687 2.274263794235377 2.027779569354112 2.441593963777407 0.0 2.1921300296238906 1.4074180502556626 2.43580163984192 2.6742742971316016 1.5023878453611568 2.561393373125879 2.4729219001407072 1.7024688754559145 2.031878289407704 2.1285104845591594 2.4275104029368664 3.6271552772898867; 2.094611185941017 2.8021965307404324 0.9193829125249725 2.214795001779845 28.954640722880548 1.9713107731682806 2.125607932833218 2.1350957114879594 2.745738139100508 1.9328599586044524 2.215487388193955 1.8876398702511683 2.427472932375576 1.8752115765965518 2.4490907931788657 2.0593735306220013 2.7183851960233714 1.8114024036111667 3.3834014088479005 2.3500634035024186 0.0 3.2685020446302504 2.230092275176853 2.9313054187220446 2.2425882317868804 2.63036842063185 2.1681555648850352 2.6402535453772344 2.0534714279580517 2.8345570632008354 1.291052181305684 2.51183274847302; 2.1039529101756598 2.294059767520302 2.5184642067951954 3.0023455219861193 1.9555400127956877 26.34922575839723 2.9796143724749107 1.7701555593719889 1.3964717726376432 2.407807075481056 2.9469359370523294 2.863174089498839 2.555820509557353 2.446276176970783 1.802049779597186 1.944649401349618 2.2587548038587526 1.9113775974681877 1.84171291535443 1.6327771961260327 2.9634579531379552 0.0 1.7627782881405332 2.5602665795552997 3.1366731692306438 2.110708350651228 1.475389164772464 1.4360259657006254 2.19126388948825 1.5688362920424526 2.7129042426017635 2.466036158105427; 2.254002560074045 2.3745557796309424 2.4057156296025157 2.449349244578196 2.113410813867813 2.9831880138821005 24.04357367738752 2.0980367478740356 1.8964416144238607 2.3269100986755835 2.131810621718956 2.3531741173032263 2.078872748226011 2.6704474715365416 2.1068019980417647 2.263490429833156 2.1044488121738585 2.375477733946526 1.926351484985495 1.685590759940386 2.5629222788336192 1.6152425782127973 0.0 2.9082687698857375 2.38139801559271 1.50484370815399 1.9394941817068432 2.015609298264674 2.6348999158093847 2.339864393400039 2.5118314703062947 2.237877123372016; 2.898736171502773 1.6911665889087502 3.0020206202242274 2.1736624914394707 1.9687640726117577 1.6195946808683814 1.9439022279632387 28.72969685422756 1.2602161156134692 2.05516783938274 2.1044049301008347 1.9152487658432384 1.7561113160230233 1.7843530811425135 2.0483643547967585 1.8673613015213006 1.5243844154123423 2.6616336916346017 0.25562727892951925 1.92107293023783 2.0430967153567314 3.5767417762513256 1.8499572631587595 0.0 3.0691803386758 2.1567815855728467 2.5648017355024257 2.038926927760925 2.9407620656905245 2.5207980185513694 2.719581657942814 1.9387848620217674; 1.7703451927259746 1.9503382293779847 1.8908391859047202 2.4952448338361815 2.5015252384734237 1.1680296323831536 1.664425832762181 1.1823348538625875 24.91877770367074 1.8322590865596118 1.8143336830252748 0.8491991365961374 2.6734429250189855 1.982834427472362 1.7665949531418257 1.8998356534736007 1.7647865000152752 1.955250861132444 2.1124318531043187 1.2499731042948468 1.2545709048011007 3.05328609460925 2.037791703627326 2.942295785581675 0.0 2.081507500596683 2.5498361596356385 2.643476619630957 1.758543306972774 2.0602941997853055 2.1186570791610984 1.6502767988515474; 2.3109829952107295 2.0184067717438325 2.3420551276083605 1.785160870058758 1.70327628729779 2.1939941645469876 2.109523546334325 1.9919158069522789 1.846888315880032 25.916012588042015 2.2447112071584234 2.2539584732552322 2.0652906108887845 2.2028834398499653 1.522808506487769 1.879291269323549 1.524872186780989 2.342840667254813 2.060112169779065 3.0047227591668295 2.8011442375983733 1.7762685252241444 1.7639141595889005 1.6504512748616005 1.7808100020507023 0.0 1.5307066121840645 1.4324395498874471 2.0481455889612667 2.0907749490678555 2.15985033184296 1.8875746035679364; 2.363780670262002 1.7827673956329313 2.437647309546433 2.3111248495570638 2.1123280648150695 2.8595473740460386 2.040848417305476 2.167577245598151 1.955387260273473 2.3711355550862008 23.78382670836272 2.753412517745913 1.3057464246763875 1.8565194824226017 2.1359306430353726 1.8295890775709562 1.6265469394358039 2.3374382244487113 1.9986033978998936 2.146284294141965 2.112621586177074 1.6276201076052348 1.9306872955165675 2.0892177913204537 2.49629778691924 1.5852612792576715 0.0 2.2060485484301284 3.416891558802725 2.828202362317165 1.9911912042246185 1.7221254010705804; 2.334793607011518 2.1610436761172176 2.4851735836913975 2.7477066227060196 1.8256895365548929 2.8169945161751584 2.303420902572356 2.019630071023166 1.0314617035269458 2.4215918108656194 2.7946215074285234 24.499036285957636 1.4471919930307964 2.380651051268325 2.44803841298941 1.945345768906293 2.1583339897966143 2.0995381425884188 1.9369460693415697 1.5924970757128225 3.2793268165868983 1.5585036838432227 2.3972740061630886 2.585319410977612 3.557869123443761 2.058212196959744 1.6479057857531179 0.0 2.740474399490953 0.8735152556107233 2.3616667411036905 2.5784671778303165; 1.683170517433554 2.4595711996277583 1.6884646891954864 2.274215782828401 2.1559688911899504 2.3000872287443217 1.8195658260057899 1.6509389137135997 2.646151784460443 2.0233702410098218 1.1374017068696478 1.2376382855414456 25.759105486030887 2.2081313944270664 1.695339406230303 2.4778804538262715 2.4954924618705587 2.2583208673056245 2.5715715333227345 1.6667670255348288 2.113707644410414 2.216045681879127 2.0973284178381553 2.1190677474427075 1.8704831897800802 1.8668042090698613 3.098958515117322 2.8998232996587188 0.0 1.4468887493041014 2.305395823734276 1.6341209042144331; 2.5429367129482165 2.0656513452102248 2.1160652184856232 1.8348869167560407 1.5969189196433464 2.1837542803901715 2.40435193354874 1.6723920630655096 1.9487546711462396 2.154174454203423 1.681386148848282 2.1643087280113944 2.2013427786594866 26.655805570490543 1.6799363287441271 2.1571789599771924 1.409878608698896 1.3730091780336349 1.514255252934824 2.2112032993384165 2.421545552989113 1.621605986454067 1.530544976411169 2.834743772166305 1.5669758289681102 1.6711571238175458 2.1136447190265906 1.2627314229160125 2.2488929601767627 0.0 2.1826031964967583 1.7687748338050844; 2.2390512255685198 2.2796030207444042 1.7669435414207864 1.7083523192653804 2.329746104261089 1.6984758510520035 1.9996544280893933 2.0953513047551837 1.8914631648511324 1.633047488876655 2.1197452774964813 2.3906440577679082 1.8474987584981521 1.8388842967795562 23.698895709160123 2.154095383069975 1.8178425526703874 1.990464079233279 2.7061479049376587 2.7107307202635136 1.361050257070218 2.147448021986292 2.574178929180369 2.0549485534557403 1.9486517593084134 2.5542247006696166 2.0909810129814432 1.6290283153478742 2.616066307310768 2.769070514725434 0.0 2.49434483081084; 1.8391134466074388 1.2706599973009627 1.9580673710303997 0.4313679364229429 1.727784510882612 1.6288311419828216 1.9440985290591701 1.7021039206581121 1.8124595343612935 1.7772859208908216 1.6011593812104523 1.6757070828631775 2.4177954752725066 2.103882597191008 1.9418510522483616 25.2048806230229 1.491403323867134 2.1981245468465236 2.0942833665079363 3.189121749791884 2.237082180243411 2.1098891490737692 2.353885313699329 2.0111215018524287 1.9266805848477344 2.172947403998948 2.2897387334805397 2.194486960760201 1.276428428595403 1.9158406935611048 1.6187583272220483 0.0]
    MY = [MY[1:T] MY[T+1:end]]
    NY = size(yM, 2)
    yM = [yM[T*(j-1) + i,k] for i in 1:T, j in 1:W, k in 1:NY]
    return T, B, G, W, L, F, SRD, Bℷ, Gℷ, Wℷ, Lℷ, PE, MY, MZ, yM, NY
end
function pushCut(D, cn, px) return (push!(D["cn"], cn); push!(D["px"], px)) end
function pushCut(D, cn, px, pβ) return (push!(D["cn"], cn); push!(D["px"], px); push!(D["pβ"], pβ)) end
function pushSimplicial(D, f, x, β) return (push!(D["f"], f); push!(D["x"], x); push!(D["β"], β)) end
function eval_Δ_at(Δ, x, β) # 1️⃣ used only in termination assessment
    isempty(Δ["f"]) && return Inf
    fV, xV, R2, βV = Δ["f"], Δ["x"], length(Δ["f"]), Δ["β"]
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[r = 1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G], sum(xV[r][i][t, g] * λ[r] for r in 1:R2) == x[i][t, g])
    JuMP.@constraint(ø, [t = 1:T, w = 1:W],          sum(βV[r][t, w]    * λ[r] for r in 1:R2) ==    β[t, w])
    JuMP.@objective(ø, Min, ip(fV, λ))
    JuMP.optimize!(ø)
    return (JuMP.termination_status(ø) == JuMP.OPTIMAL ? JuMP.objective_value(ø) : Inf)
end
function eval_xby(x1, β1, Y) # used in t2f, vertex enum phase, to aid in deciding worstY
    fV, x2V, R2, β2V = Δ2["f"], Δ2["x"], length(Δ2["f"]), Δ2["β"]  
    @assert R2 >= 1
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[r = 1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
    JuMP.@constraint(ø, [t = 1:T, l = 1:L],             sum( β2V[r][t, l]      * λ[r] for r in 1:R2) ==    β2[t, l])
    # following 2 lines are minimum infeas. system ############################################################
    JuMP.@constraint(ø, [i = 1:3, t = 1:T, g = 1:G],    sum(x2V[r][1][i][t, g] * λ[r] for r in 1:R2) == x1[i][t, g])
    JuMP.@constraint(ø, [t = 1:T, w = 1:W],             sum(x2V[r][2][t, w]    * λ[r] for r in 1:R2) ==     Y[t, w])
    ###########################################################################################################
    JuMP.@objective(ø, Min, -ip(β1, Y) + ip(MZ, β2) + ip(fV, λ))
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.INFEASIBLE
            return Inf
        else
            error(" eval_xby() : $status ")
        end
    else
        return JuMP.objective_value(ø)
    end 
end
function psi(x1, β1, Y) # used in t2f only to generate (x2, β2)
    x2 = x1, Y
    cnV, px2V, R2, pβ2V = ℶ2["cn"], ℶ2["px"], length(ℶ2["cn"]), ℶ2["pβ"]
    ø = JumpModel(0)
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
    JuMP.@variable(ø, psiObj)
    JuMP.@constraint(ø, [r = 1:R2], psiObj >= cnV[r] + ip(px2V[r], x2) + ip(MZ .+ pβ2V[r], β2))
    JuMP.@objective(ø, Min, psiObj)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            JuMP.set_lower_bound(psiObj, HYPER_PARAM_LB) # ⚠️✅ This lower bound should be low enough, to encourage more trial β2's
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
            @assert status == JuMP.OPTIMAL " in value function psi: #22 $status"
            lb = -Inf
        else
            error(" in value function psi: $status ")
        end
    else
        lb = JuMP.value(psiObj)
    end
    β2 = JuMP.value.(β2)
    return lb, x2, β2 # when applied, check if lb == -Inf,
end
function gen_cut_for_ℶ1(x1Γ, Y) # used in t2b only
    pβ1 = -Y # 💡 this is fixed
    cnV, px2V, R2, pβ2V = ℶ2["cn"], ℶ2["px"], length(ℶ2["cn"]), ℶ2["pβ"]
    ø = JumpModel(0)
    JuMP.@variable(ø, x1[i = 1:3, t = 1:T, g = 1:G]) # 'x1' as a part of x2
    JuMP.@variable(ø, β2[t = 1:T, l = 1:L])
    JuMP.@variable(ø, psiObj)
    JuMP.@constraint(ø, cp[i = 1:3, t = 1:T, g = 1:G], x1[i, t, g] == x1Γ[i][t, g])
    JuMP.@constraint(ø, [r = 1:R2], psiObj >= cnV[r] + ip( px2V[r], ((x1[1, :, :], x1[2, :, :], x1[3, :, :]), Y) ) + ip(MZ .+ pβ2V[r], β2))
    JuMP.@objective(ø, Min, psiObj)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE # this may happen in nascent phase of iterations
            return -Inf # fail to generate a cut
        else
            error(" in gen_cut_for_ℶ1: $status ")
        end
    else
        psiObj = JuMP.value(psiObj)
        px1 = JuMP.value.(cp[1, :, :]), JuMP.value.(cp[2, :, :]), JuMP.value.(cp[3, :, :])
        cn = psiObj - ip(px1, x1Γ)
        return psiObj, cn, px1, pβ1
    end
end
function maximize_φ2_over_Z(is01::Bool, x2, β2)
    (u, v, x), Y = x2
    ø = JumpModel(0)
    JuMP.@variable(ø, Z[t = 1:T, l = 1:L])
    if is01
        JuMP.set_binary.(Z)
    else
        [JuMP.set_upper_bound.(Z, 1.), JuMP.set_lower_bound.(Z, 0.)]
    end
    JuMP.@variable(ø, ℵQ1[t = 1:T, g = 1:G])
    JuMP.@variable(ø, ℵQ2[t = 1:T, g = 1:G])
    JuMP.@variable(ø, ℵQ3[t = 1:T, g = 1:G])
    JuMP.@variable(ø, ℵW[t = 1:T, w = 1:W] >= 0.)
    JuMP.@variable(ø, ℵL[t = 1:T, l = 1:L] >= 0.)
    JuMP.@variable(ø, 0. <= ℵe[t = 1:T, g = 1:G] <= 1.) # RHS due to e >= 0
    JuMP.@variable(ø, ℵdl1[g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdl[t = 2:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdr1[g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵdr[t = 2:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵPI[t = 1:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵPS[t = 1:T, g = 1:G] >= 0.)
    JuMP.@variable(ø, ℵbl[t = 1:T, b = 1:B] >= 0.)
    JuMP.@variable(ø, ℵbr[t = 1:T, b = 1:B] >= 0.)
    JuMP.@variable(ø, ℵR[t = 1:T] >= 0.)
    JuMP.@variable(ø, ℵ0[t = 1:T] >= 0.)
    JuMP.@constraint(ø,  ϖ[t = 1:T, w = 1:W],  ℵ0[t] + ℵW[t, w] + Wℷ["CW"][t, w] - PE + sum(F[b, Wℷ["n"][w]] * (ℵbl[t, b] - ℵbr[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(ø,  ζ[t = 1:T, l = 1:L], -ℵ0[t] + ℵL[t, l] + Lℷ["CL"][t, l] + PE + sum(F[b, Lℷ["n"][l]] * (ℵbr[t, b] - ℵbl[t, b]) for b in 1:B) >= 0.)
    JuMP.@constraint(ø,  ρ[t = 1:T, g = 1:G], ℵPS[t, g] - ℵR[t] + Gℷ["CR"][g] >= 0.)
    JuMP.@constraint(ø, p²[t = 1:T, g = 1:G], Gℷ["C2"][g] * ℵe[t, g] - ℵQ2[t, g] - ℵQ1[t, g] == 0.)
    JuMP.@expression(ø,  pCommon[t = 1:T, g = 1:G], ℵPS[t, g] - ℵPI[t, g] - ℵ0[t] - 2. * ℵQ3[t, g] + Gℷ["C1"][g] * ℵe[t, g] + PE + sum((ℵbr[t, b] - ℵbl[t, b]) * F[b, Gℷ["n"][g]] for b in 1:B))
    JuMP.@constraint(ø,  pt1[g = 1:G], pCommon[1, g] + ℵdr1[g] - ℵdl1[g] + ℵdl[2, g] - ℵdr[2, g] == 0.)
    JuMP.@constraint(ø,  prest[t = 2:T-1, g = 1:G], pCommon[t, g] + ℵdr[t, g] - ℵdl[t, g] + ℵdl[t+1, g] - ℵdr[t+1, g] == 0.)
    JuMP.@constraint(ø,  ptT[g = 1:G], pCommon[T, g] + ℵdr[T, g] - ℵdl[T, g] == 0.)
    JuMP.@constraint(ø, [t = 1:T, g = 1:G], [ℵQ1[t, g], ℵQ2[t, g], ℵQ3[t, g]] in JuMP.SecondOrderCone())
    JuMP.@objective(ø, Max, -ip(β2, Z) # ⚠️ don't forget the outer term
        + PE * sum(sum(Y[t, w] for w in 1:W) - sum(Lℷ["M"][l] * Z[t, l]  for l in 1:L) for t in 1:T)
        + sum(ℵQ2 .- ℵQ1) + sum(ℵe[t, g] * (Gℷ["C0"][g] - (1 - x[t, g]) * Gℷ["M"][g]) for t in 1:T, g in 1:G)
        - sum(ℵW[t, w] * Y[t, w] for t in 1:T, w in 1:W) - sum(ℵL[t, l] * Lℷ["M"][l] * Z[t, l] for t in 1:T, l in 1:L)
        + SRD * sum(ℵR) + sum( ℵPI[t, g] * Gℷ["PI"][g] * x[t, g] - ℵPS[t, g] * Gℷ["PS"][g] * x[t, g]  for t in 1:T, g in 1:G)
        + sum((ℵbr[t, b] - ℵbl[t, b]) * (sum(F[b, Wℷ["n"][w]] * Y[t, w] for w in 1:W) - sum(F[b, Lℷ["n"][l]] * Lℷ["M"][l] * Z[t, l] for l in 1:L)) for t in 1:T, b in 1:B)
        + sum((ℵbl[t, b] + ℵbr[t, b]) * (-Bℷ["BC"][b]) for t in 1:T, b in 1:B)
        + sum( ℵ0[t] * (sum(Lℷ["M"][l] * Z[t, l] for l in 1:L) - sum(Y[t, w] for w in 1:W)) for t in 1:T)
        + sum(ℵdl1[g] * (Gℷ["ZP"][g] - Gℷ["RD"][g] * x[1, g] - Gℷ["SD"][g] * v[1, g]) - ℵdr1[g] * (Gℷ["RU"][g] * Gℷ["ZS"][g] + Gℷ["SU"][g] * u[1, g] + Gℷ["ZP"][g]) for g in 1:G)
        + sum( ℵdl[t, g] * (-Gℷ["RD"][g] * x[t, g] - Gℷ["SD"][g] * v[t, g]) - ℵdr[t, g] * (Gℷ["RU"][g] * x[t-1, g] + Gℷ["SU"][g] * u[t, g]) for t in 2:T, g in 1:G)
    )
    # @info " maximizing φ2 over Z given (x2, β2) ... "
    # JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            return Inf
        else
            error(" in maximize_φ2_over_Z(): $status ")
        end
    else
        return worstObj, worstZ = JuMP.objective_value(ø), JuMP.value.(Z)
    end
end
function gen_cut_for_ℶ2(x2, Z) 
    pβ2 = -Z # 💡 this is fixed
    ø = JumpModel(0)
    JuMP.@variable(ø, u[t = 1:T, g = 1:G])
    JuMP.@variable(ø, v[t = 1:T, g = 1:G])
    JuMP.@variable(ø, x[t = 1:T, g = 1:G])
    JuMP.@variable(ø, Y[t = 1:T, w = 1:W])
    JuMP.@constraint(ø, cpu[t = 1:T, g = 1:G], u[t, g] == x2[1][1][t, g])
    JuMP.@constraint(ø, cpv[t = 1:T, g = 1:G], v[t, g] == x2[1][2][t, g])
    JuMP.@constraint(ø, cpx[t = 1:T, g = 1:G], x[t, g] == x2[1][3][t, g])
    JuMP.@constraint(ø, cpY[t = 1:T, w = 1:W], Y[t, w] == x2[2][t, w])
        JuMP.@variable(ø,  ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø,  ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø,  ρ[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø,  p[t = 1:T, g = 1:G])
        JuMP.@variable(ø, p²[t = 1:T, g = 1:G])
        JuMP.@variable(ø,  e[t = 1:T, g = 1:G] >= 0.)
        JuMP.@constraint(ø, ℵW[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, ℵL[t = 1:T, l = 1:L], Lℷ["M"][l] * Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, [t = 1:T, g = 1:G], [p²[t, g] + 1, p²[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone())
        JuMP.@constraint(ø, ℵe[t = 1:T, g = 1:G], e[t, g] >= Gℷ["C2"][g] * p²[t, g] + Gℷ["C1"][g] * p[t, g] + Gℷ["C0"][g] - (1 - x[t, g]) * Gℷ["M"][g])
        JuMP.@constraint(ø, ℵdl1[g = 1:G], p[1, g] - Gℷ["ZP"][g]       >= -Gℷ["RD"][g] * x[1, g] - Gℷ["SD"][g] * v[1, g])
        JuMP.@constraint(ø, ℵdl[t = 2:T, g = 1:G], p[t, g] - p[t-1, g] >= -Gℷ["RD"][g] * x[t, g] - Gℷ["SD"][g] * v[t, g])
        JuMP.@constraint(ø, ℵdr1[g = 1:G], Gℷ["RU"][g] * Gℷ["ZS"][g] + Gℷ["SU"][g] * u[1, g]       >= p[1, g] - Gℷ["ZP"][g])
        JuMP.@constraint(ø, ℵdr[t = 2:T, g = 1:G], Gℷ["RU"][g] * x[t-1, g] + Gℷ["SU"][g] * u[t, g] >= p[t, g] - p[t-1, g])
        JuMP.@constraint(ø, ℵPI[t = 1:T, g = 1:G], p[t, g] >= Gℷ["PI"][g] * x[t, g])
        JuMP.@constraint(ø, ℵPS[t = 1:T, g = 1:G], Gℷ["PS"][g] * x[t, g] >= p[t, g] + ρ[t, g])
        JuMP.@constraint(ø, ℵbl[t = 1:T, b = 1:B],
            sum(F[b, Gℷ["n"][g]] * p[t, g] for g in 1:G) + sum(F[b, Wℷ["n"][w]] * (Y[t, w] - ϖ[t, w]) for w in 1:W) - sum(F[b, Lℷ["n"][l]] * (Lℷ["M"][l] * Z[t, l] - ζ[t, l]) for l in 1:L) >= -Bℷ["BC"][b]
        )
        JuMP.@constraint(ø, ℵbr[t = 1:T, b = 1:B],
            Bℷ["BC"][b] >= sum(F[b, Gℷ["n"][g]] * p[t, g] for g in 1:G) + sum(F[b, Wℷ["n"][w]] * (Y[t, w] - ϖ[t, w]) for w in 1:W) - sum(F[b, Lℷ["n"][l]] * (Lℷ["M"][l] * Z[t, l] - ζ[t, l]) for l in 1:L)
        )
        JuMP.@constraint(ø, ℵR[t = 1:T], sum(ρ[t, :]) >= SRD)
        JuMP.@constraint(ø, ℵ0[t = 1:T], sum(Y[t, w] - ϖ[t, w] for w in 1:W) + sum(p[t, :]) - sum(Lℷ["M"][l] * Z[t, l] - ζ[t, l] for l in 1:L) >= 0.)
        JuMP.@expression(ø, CP[t = 1:T], sum(Y[t, w] - ϖ[t, w] for w in 1:W) + sum(p[t, :]) - sum(Lℷ["M"][l] * Z[t, l] - ζ[t, l] for l in 1:L))
        JuMP.@expression(ø, CW[t = 1:T, w = 1:W], Wℷ["CW"][t, w] * ϖ[t, w])
        JuMP.@expression(ø, CL[t = 1:T, l = 1:L], Lℷ["CL"][t, l] * ζ[t, l])
        JuMP.@expression(ø, CR[t = 1:T, g = 1:G], Gℷ["CR"][g]    * ρ[t, g])
        JuMP.@expression(ø, COST2, sum(CW) + sum(CL) + sum(CR) + sum(e) + PE * sum(CP))
    JuMP.@objective(ø, Min, COST2)
    JuMP.set_attribute(ø, "QCPDual", 1)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        error(" in gen_cut_for_ℶ2(): $status ")
    else
        COST2 = JuMP.value(COST2)
        px1 = JuMP.dual.(cpu), JuMP.dual.(cpv), JuMP.dual.(cpx)
        pY = JuMP.dual.(cpY)
        px2 = px1, pY
        cn = JuMP.objective_value(ø) - ip(px2, x2)
        return COST2, cn, px2, pβ2
    end
end
function master(is01::Bool)
    cnV1, px1V1, R1 = Ϝ1["cn"], Ϝ1["px"], length(Ϝ1["cn"])
    cnV2, px1V2, R2, pβV = ℶ1["cn"], ℶ1["px"], length(ℶ1["cn"]), ℶ1["pβ"]
    ø = JumpModel(0)
    JuMP.@variable(ø, o)
    JuMP.@variable(ø, β1[t = 1:T, w = 1:W])
    JuMP.@variable(ø, u[t = 1:T, g = 1:G])
    JuMP.@variable(ø, v[t = 1:T, g = 1:G])
    JuMP.@variable(ø, x[t = 1:T, g = 1:G])
    if is01
        JuMP.set_binary.([u; v; x])
    else
        (JuMP.set_lower_bound.([u; v; x], 0.); JuMP.set_upper_bound.([u; v; x], 1.))
    end
    JuMP.@constraint(ø, [r = 1:R1], 0 >= cnV1[r] + ip(px1V1[r], (u, v, x)))
    JuMP.@constraint(ø, [r = 1:R2], o >= cnV2[r] + ip(px1V2[r], (u, v, x)) + ip(MY .+ pβV[r], β1))
        JuMP.@constraint(ø, [g = 1:G],          x[1, g] - Gℷ["ZS"][g] == u[1, g] - v[1, g])
        JuMP.@constraint(ø, [t = 2:T, g = 1:G], x[t, g] - x[t-1, g] == u[t, g] - v[t, g])
        JuMP.@constraint(ø, [g = 1:G, t = 1:T-Gℷ["UT"][g]+1], sum(x[i, g] for i in t:t+Gℷ["UT"][g]-1)      >= Gℷ["UT"][g] * u[t, g])
        JuMP.@constraint(ø, [g = 1:G, t = T-Gℷ["UT"][g]+1:T], sum(x[i, g] - u[t, g] for i in t:T)          >= 0.)
        JuMP.@constraint(ø, [g = 1:G, t = 1:T-Gℷ["DT"][g]+1], sum(1. - x[i, g] for i in t:t+Gℷ["DT"][g]-1) >= Gℷ["DT"][g] * v[t, g])
        JuMP.@constraint(ø, [g = 1:G, t = T-Gℷ["DT"][g]+1:T], sum(1. - x[i, g] - v[t, g] for i in t:T)     >= 0.)
        JuMP.@expression(ø, CST[t = 1:T, g = 1:G], Gℷ["CST"][g] * u[t, g])
        JuMP.@expression(ø, CSH[t = 1:T, g = 1:G], Gℷ["CSH"][g] * v[t, g])
        JuMP.@expression(ø, COST1, sum([CST; CSH]))
        JuMP.@expression(ø, cost1, COST1 + ip(MY, β1)) # this value is always finite
    JuMP.@objective(ø, Min, COST1 + o)
    # JuMP.unset_silent(ø)
    (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
    if status != JuMP.OPTIMAL
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
        if status == JuMP.DUAL_INFEASIBLE
            JuMP.set_lower_bound(o, HYPER_PARAM_LB)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
            @assert status == JuMP.OPTIMAL " in master(): #22 $status"
            lb, cost1, φ1lb = -Inf, JuMP.value(cost1), -Inf
        else
            error(" in master(): $status ")
        end
    else
        lb, cost1 = JuMP.objective_value(ø), JuMP.value(cost1)
        φ1lb = lb - cost1
    end
    x1 = JuMP.value.(u), JuMP.value.(v), JuMP.value.(x)
    β1 = JuMP.value.(β1)
    return lb, cost1, φ1lb, x1, β1
end

T, B, G, W, L, F, SRD, Bℷ, Gℷ, Wℷ, Lℷ, PE, MY, MZ, yM, NY = load_data()
Random.seed!(23)
u, v, x, Y, Z, x1, x2, β1, β2 = let
    u, v, x = 1. * rand(Bool, T, G), 1. * rand(Bool, T, G), 1. * rand(Bool, T, G)
    x1 = u, v, x
    β1, Y = rand(T, W), rand(T, W)
    x2 = x1, Y
    β2, Z = rand(T, L), rand(T, L)
    u, v, x, Y, Z, x1, x2, β1, β2
end
Ϝ1, ℶ1, ℶ2, Δ1, Δ2 = let
    Ϝ1 = Dict(
        "cn" => Float64[],
        "px" => typeof(x1)[]
    )
    ℶ1 = Dict(
        "cn" => Float64[],
        "px" => typeof(x1)[],
        "pβ" => typeof(β1)[]
    )
    ℶ2 = Dict(
        "cn" => Float64[],
        "px" => typeof(x2)[],
        "pβ" => typeof(β2)[]
    )
    Δ1 = Dict( # 1️⃣ used only in termination assessment
        "f" => Float64[],
        "x" => typeof(x1)[],
        "β" => typeof(β1)[]
    )
    Δ2 = Dict(
        "f" => Float64[],
        "x" => typeof(x2)[],
        "β" => typeof(β2)[]
    )
    Ϝ1, ℶ1, ℶ2, Δ1, Δ2
end
HYPER_PARAM_LB = -1e6
tV = [t1]
x1V, β1V = [x1], [β1]
x2V, β2V = [x2], [β2]
lbV, cost = zeros(1), zeros(3)
while true
    𝚃 = tV[1]
    if 𝚃 == t1
        lb, cost[1], φ1lb, x1, β1 = master(true)
        φ1ub = eval_Δ_at(Δ1, x1, β1)
        if φ1ub == Inf
            @info " ▶ lb = $lb, (Δ1 $(length(Δ1["f"])),ℶ1 $(length(ℶ1["cn"])),Δ2 $(length(Δ2["f"])),ℶ2 $(length(ℶ2["cn"])))"
        else
            ub = lb - φ1lb + φ1ub
            @info " ▶⋅▶ lb = $lb | $ub = ub, (Δ1 $(length(Δ1["f"])),ℶ1 $(length(ℶ1["cn"])),Δ2 $(length(Δ2["f"])),ℶ2 $(length(ℶ2["cn"])))"
            gap = abs((ub - lb) / ub)
            if gap < 1/10000
                @info "  😊 RDDP converges gap = $gap "
                break
            end
        end
        x1V[1], β1V[1], lbV[1] = x1, β1, lb
        tV[1] = t2f
    elseif 𝚃 == t2f
        x1, β1 = x1V[1], β1V[1]
        if isempty(Δ2["f"]) # Only once
            index = 1
        else # 📚 vertex enum
            earlyInd, earlyTer = [0], falses(1)
            normalVec = zeros(NY)
            for i in 1:NY
                value = eval_xby(x1, β1, yM[:, :, i])
                if value == Inf
                    earlyInd[1], earlyTer[1] = i, true
                    break
                else
                    normalVec[i] = value
                end
            end
            index = (earlyTer[1] ? earlyInd[1] : findmax(normalVec)[2])
        end
        worstY = yM[:, :, index] # ★ decide Y ★
        _, x2, β2 = psi(x1, β1, worstY) # ★ decide trial (x2, β2) ★
        cost[2] = -ip(worstY, β1) + ip(MZ, β2)
        x2V[1], β2V[1] = x2, β2
        tV[1] = t3
    elseif 𝚃 == t3
        x2, β2 = x2V[1], β2V[1]
        ret = maximize_φ2_over_Z(true, x2, β2)
        if length(ret) == 1
            error("TODO: Generate feasibility cut for x1, and then go back to State = t1")
        else
            worstObj, worstZ = ret
            cost[3] = worstObj
            @debug " lb = $(lbV[1]) | $(sum(cost)) = cost ◀"
            pushSimplicial(Δ2, worstObj, x2, β2)
            COST2, cn, px2, pβ2 = gen_cut_for_ℶ2(x2, worstZ)
                otherObj = -ip(β2V[1], worstZ) + COST2
                isapprox(worstObj, otherObj; atol=1) || @error " ❌ ($worstObj, $otherObj) Please check the final stage's Max and Min Program ❌ "
            pushCut(ℶ2, cn, px2, pβ2)
            tV[1] = t2b
        end
    elseif 𝚃 == t2b
        x1, β1 = x1V[1], β1V[1]
        eI, eT = [0], falses(1)
        nV = zeros(NY)
        for i in 1:NY
            value = eval_xby(x1, β1, yM[:, :, i])
            if value == Inf
                eI[1], eT[1] = i, true
                break
            else
                nV[i] = value
            end
        end
        worstObj, index = (eT[1] ? (Inf, eI[1]) : findmax(nV))
        worstY = yM[:, :, index]
        ret = gen_cut_for_ℶ1(x1, worstY)
        if length(ret) == 1
            tV[1] = t2f # because we jump to t2f directly, we don't have to update Δ1 either
        else
            psiObj, cn, px1, pβ1 = ret
            pushCut(ℶ1, cn, px1, pβ1) # UPD LOWER
            if worstObj < Inf
                t2b_lb = -ip(β1, worstY) + psiObj
                @debug " in t2b, lb = $t2b_lb | $worstObj = ub "
                pushSimplicial(Δ1, worstObj, x1, β1)
            end
            tV[1] = t1
        end
    end
end

# ...
# [ Info:  ▶⋅▶ lb = 45757.10254399604 | 45839.51035069877 = ub, (Δ1 568,ℶ1 568,Δ2 625,ℶ2 625)
# [ Info:  ▶⋅▶ lb = 45757.27349311435 | 45919.82271273894 = ub, (Δ1 569,ℶ1 569,Δ2 626,ℶ2 626)
# [ Info:  ▶⋅▶ lb = 45757.58308157501 | 45864.10043182026 = ub, (Δ1 570,ℶ1 570,Δ2 627,ℶ2 627)
# [ Info:  ▶⋅▶ lb = 45757.59160850522 | 45759.58215668354 = ub, (Δ1 571,ℶ1 571,Δ2 628,ℶ2 628)
# [ Info:   😊 RDDP converges gap = 4.350013886704033e-5



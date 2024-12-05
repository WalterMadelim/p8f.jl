import LinearAlgebra
import Distributions
import Statistics
import Random
import Gurobi
import JuMP
using Logging
GRB_ENV = Gurobi.Env()

# with only lagrangian cuts, don't consider other type of cuts because the main bottleneck is the bilinear program
# the argmaxZ is executed to global optimality, because there are no other efficive methods
# relatively complete recourse -> dual variable has an upper bound -> estimate it -> enforce beta_bound
# the lb increasing process is steady and continual ✅
# DRO UC stable version with Formulation 1 (has ofc and ofv)
# 5/12/24

macro optimise() return esc(:((_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø)))) end
macro reoptimise()
    return esc(quote
        if status == JuMP.INFEASIBLE_OR_UNBOUNDED
            JuMP.set_attribute(ø, "DualReductions", 0)
            (_, status) = (JuMP.optimize!(ø), JuMP.termination_status(ø))
        end
    end)
end
rd6(f)         = round(f; digits = 6)
gap_lu(lb, ub) = abs(ub - lb) / max(abs(lb), abs(ub)) # the `rtol` in isapprox
jv(x)          = JuMP.value.(x)
get_bin_var(x) = Bool.(round.(jv(x))) # applicable only when status == JuMP.OPTIMAL
ip(x, y)       = LinearAlgebra.dot(x, y)
norm1(x)       = LinearAlgebra.norm(x, 1)
@enum State begin t1; t2f; t3; t2b end
btBnd = 5.0 # 🧪 if global optimality cannot be attained, you may want to enlarge it
cϵ    = 1e-6 # positive, small => more precise
function JumpModel(i)
    if i == 0 
        ø = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV)) # JuMP.set_attribute(ø, "QCPDual", 1)
    elseif i == 1 
        ø = JuMP.Model(MosekTools.Optimizer) # vio = JuMP.get_attribute(ø, Gurobi.ModelAttribute("MaxVio")) 🍀 we suggest trial-and-error to decide if a constr is tight, rather than using this cmd
    elseif i == 2 
        ø = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV))
    end
    JuMP.set_silent(ø) # JuMP.unset_silent(ø)
    return ø
end
T, G, W, L, B = 8, 2, 2, 3, 11 # 🌸 G+1 is the size of (u, v, x)
function load_UC_data(T)
    @assert T in 1:8
    UT = DT = 3
    CST = [0.72, 0.60, 0.63]/5;
    CSH = [0.15, 0.15, 0.15]/5;
    CL = [8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 8.0 6.888 7.221; 16.0 13.776 14.443]/5;
    CL = CL[end-T+1:end, :]
    CG = [3.6, 3.4, 4.0]/5;
    C2 = [34, 23, 41]/2000
    C1 = [0.67, 0.41, 0.93]/2/5;
    C0 = CST / T;
    PI = [0.45, 0.375, 0.5];
    PS = [5.5,  4,     4.5];
    EM = C2 .* PS .* PS .+ C1 .* PS .+ C0;
    LM = [4, 3.5, 3];
    ZS = [0, 0, 1.0]; # Binary
    ZP = [0, 0, 0.5];
    NG = [3, 2] # 🍀 since `G+1` generator is on the slack bus, it doesn't contribute to any power flow, we omit it
    NW = [2, 3]
    NL = [4, 5, 6]
    FM = let
        [0.0 -0.44078818231301325 -0.3777905547603966 -0.2799660317280192 -0.29542103345397086 -0.3797533556433226; 0.0 -0.32937180203296385 -0.3066763814017814 -0.5368505424397501 -0.27700207451336545 -0.30738349678665927; 0.0 -0.229840015654023 -0.3155330638378219 -0.18318342583223077 -0.42757689203266364 -0.31286314757001815; 0.0 0.06057464187751593 -0.354492865935812 0.018549141862024054 -0.10590781852459258 -0.20249230420747694; 0.0 0.3216443011699881 0.23423126113776493 -0.3527138586915366 0.11993854023522055 0.23695476674932447; 0.0 0.10902536164428178 -0.020830957469362865 0.03338570129374399 -0.19061834882900958 -0.016785057525005476; 0.0 0.0679675129952012 -0.23669799249298656 0.02081298380774932 -0.11883340633558914 -0.39743076066016436; 0.0 0.06529291323070335 0.270224001096538 0.019993968970548615 -0.11415717519819152 0.1491923753527435; 0.0 -0.004718271353187364 0.3752831329676507 -0.001444827108524449 0.00824935667359894 -0.35168467956022; 0.0 -0.007727500862975828 -0.07244512026401648 0.11043559886871346 -0.15706353427814482 -0.0704287300373348; 0.0 -0.06324924164201379 -0.13858514047466358 -0.019368156699224842 0.11058404966199031 -0.2508845597796152]
    end
    BC = 2.1 * [1.0043, 2.191, 1.3047, 0.6604, 1.7162, 0.6789, 1.0538, 1.1525, 1.3338, 0.4969, 0.7816]
    (RU = [2.5, 1.9, 2.3]; SU = 1.3 * RU; RD = 1.1 * RU; SD = 1.3 * RD)
    return CST, CSH, CL, CG, C2, C1, C0, EM, PI, PS, LM, ZS, ZP, NG, NW, NL, FM, BC, RU, SU, RD, SD, UT, DT
end
CST, CSH, CL, CG, C2, C1, C0, EM, PI, PS, LM, ZS, ZP, NG, NW, NL, FM, BC, RU, SU, RD, SD, UT, DT = load_UC_data(T)
ℶ1, ℶ2, Δ2, ℸ1, ℸ2 = let
    ℸ1 = Dict( # store solutions of lag_subproblem
        "oψ" => Float64[], # trial value
        "u" => Matrix{Float64}[], # trial vector
        "v" => Matrix{Float64}[], # trial vector
        "x" => Matrix{Float64}[], # trial vector
    )
    ℸ2 = Dict( # store solutions of lag_subproblem
        "ofv" => Float64[], # trial value
        "u" => Matrix{Float64}[], # trial vector
        "v" => Matrix{Float64}[], # trial vector
        "x" => Matrix{Float64}[], # trial vector
        "Y" => Matrix{Float64}[], # trial vector
    )
    ℶ1 = Dict(
        "st" => Bool[],
        "x" =>  BitMatrix[], # contain x only, where u, v can be decoded from
        "rv" => Int[], # index of Y
        "cn" => Float64[],
        "pu" => Matrix{Float64}[],
        "pv" => Matrix{Float64}[],
        "px" => Matrix{Float64}[],
        "pβ" => Matrix{Float64}[] # slope of β1
    )
    ℶ2 = Dict(
        # "rv" is negation of pβ, thus dropped
        "st" => Bool[],
        "cn" => Float64[],
        "pu" => Matrix{Float64}[],
        "pv" => Matrix{Float64}[],
        "px" => Matrix{Float64}[],
        "pY" => Matrix{Float64}[],
        "pβ" => Matrix{Float64}[] # slope of β2
    )
    Δ2 = Dict( # 🌸 used in argmaxY
        "f" => Float64[],
        "x" => BitMatrix[],
        "Y" => Int[],
        "β" => Matrix{Float64}[] # β2
    )
    ℶ1, ℶ2, Δ2, ℸ1, ℸ2
end
brcs(v) = ones(T) * transpose(v) # to broadcast those timeless cost coeffs 
begin # load Y and Z's uncertainty data
    # T = 4
    # yM = [12.932203243205766 1.3374359039022035; 1.9446659491235203 1.4162431451003246; 1.6511993236129792 2.1773399042968133; 1.782042683828961 2.557790493339905;;; 2.0011822386947595 1.6789225705000335; 13.690850014602509 1.997467388472595; 1.9198034781728475 1.9293910538786738; 2.107819269712466 1.8111662782757756;;; 1.5403048907063075 1.9116904989768762; 1.7523927556949361 1.8959624996410318; 11.758573486702598 2.129373069191475; 2.360835863553524 1.654410550094579;;; 1.5715933164025124 1.7031010579277321; 1.8408536127147777 1.9246653956279802; 2.2612809290337474 2.075380480088833; 11.753746532489489 2.246120859260827;;; 1.4384755553357391 11.943362474288024; 1.72344593236233 1.91548036162283; 2.1236245833170844 2.0635541107258346; 2.0145900767877167 1.9949694636426214;;; 1.421390950378163 1.8195885154671323; 1.9460989041791934 12.203355270334711; 2.0120047378255426 1.2887952780925083; 2.140262568332268 1.7497035506584138;;; 1.8926711067034918 1.6778456616989779; 1.5882059667141126 0.9989786752213483; 1.9555987045048253 13.308074727575503; 2.00116104992196 1.8335475496285436;;; 2.4284307784258465 1.764570097295026; 1.6252902737904777 1.615196030466516; 1.6359452680871918 1.9888566323078063; 2.327210511773217 13.145864096700304;;; 0.0 1.9303046697671453; 1.340036139647707 2.1272564678253922; 1.9643842464561987 1.8073858830728204; 2.124702105273566 1.2240590219268372;;; 1.350599019155928 1.6358596490089128; 0.0 1.4458183973273762; 1.7581241000312189 2.446722329979181; 1.7730088350804067 1.6995757281062374;;; 1.8272891775727969 1.3644451443696721; 1.614578717229169 1.7184087355368423; 0.0 1.9730551415386182; 1.2411914714500063 1.9819256857014036;;; 1.9689580539401672 1.537755504194629; 1.2568111339355166 1.4801389249965697; 1.4256251018077006 1.9976963085038109; 0.0 1.5725426375997271;;; 1.9538021541563502 0.0; 1.4684000134553443 1.3878659549049586; 1.5755518609653678 1.9138930410237733; 1.9252030431249159 1.7622562137125504;;; 1.9884105546097959 1.4568327122536733; 1.4511724621473678 0.0; 1.8340879617445378 2.7129660182996957; 1.9294116598340794 1.9669297233369172;;; 1.3725434547187787 1.689163218152612; 1.6119892285052182 2.582693976219158; 1.7352439054027282 0.0; 2.0012409991934312 1.9773612650438648;;; 1.5651952694036888 1.4769477561641766; 1.7825097976930242 1.8205879836574106; 1.9657939031493161 2.031470938642274; 1.578855990899333 0.0]
    # MY = [1.8085556385124137 1.7156454852430747; 1.8414127842661827 1.8103587289251535; 1.8688461336064663 2.09619165798559; 2.0042839049623544 1.874316195757285]
    # MZ = [1.26723829950154 0.6727096064374787 2.2152435305495493; 3.0986377097936217 3.4229623321850227 1.6827441306529918; 2.8331247279184395 0.4899771169412037 2.331676836964725; 1.8849192189104738 3.2395724551885237 1.6774485560558825]
    # vertexY(i) = yM[:, :, i]
    # rdZ() = let
        # Dload = Distributions.Arcsine.(LM)
        # vec = [rand.(Dload) for t in 1:T]
        # [vec[t][l] for t in 1:T, l in 1:L]
    # end
    # rdYZ() = vertexY(rand(1:size(yM, 3))), rdZ() # used in deterministic formulation
    T = 8
    yM = [22.088909360533865 1.5937028236829702; 1.5082506837064866 1.951867345621028; 2.291623087390903 2.3937996539237125; 1.8566002295321558 2.362423634200421; 1.492093619215327 1.423860859526914; 1.8170048838277775 2.3215904657895217; 2.209982356424521 1.9049130474319838; 2.405448864870074 1.6561875754819675;;; 1.5087491927625256 1.7767174856836452; 24.087323242155087 2.188174086774961; 1.7768909792881198 1.6351202152770634; 2.3049799677484284 2.330144447207555; 1.8783744295053812 1.9844309040737582; 2.0476870392349475 2.03101705882146; 2.349021398149296 1.283519082316638; 0.8980622403112144 1.6018001690850212;;; 2.31137607483891 1.9665243195952595; 1.7961454576800884 1.656284870969303; 20.69012262470563 2.3761141367479; 2.164294249623073 2.141191366827183; 1.0658487341492324 1.8492858144656241; 1.9301225726915048 2.0168715018914267; 2.104770652261214 1.722050552539558; 2.6284352712858317 1.4387484566309283;;; 1.6552249007065036 1.8485734132611835; 2.103106129866737 1.6991700130158025; 1.9431659333494125 2.137899899088459; 21.79334855615469 2.133724172961574; 1.6070922571042165 1.9644453179131087; 2.274255905109503 1.9205332719712371; 2.299557635149808 1.545410304296981; 1.6168908901319237 0.9849024244351048;;; 1.5703965712998578 2.0705455720362744; 1.9561788725338725 1.8649564312665845; 1.1243986987857553 2.459570166385635; 1.8867705380143998 1.5054870267605125; 22.50816139970041 1.8544919992813125; 1.8715645000332264 1.5155511698455912; 1.6510437187755773 1.7448280512308727; 1.9542041028712775 1.960143526195284;;; 1.826425208451631 1.2399273119568464; 2.0566088548027612 1.8662155087128487; 1.9197899098673514 2.8180809485926206; 2.4850515585590083 2.2339814291266737; 1.8026818725725489 1.9722464066724377; 23.230729020441604 1.8648958896457464; 2.3776840116551305 1.8792714273731155; 1.6527284319120166 1.3463416766928757;;; 1.9734090026040993 1.296494164030331; 2.1119495352728346 2.1332125512913307; 1.8484443109927853 1.80080700793543; 2.2643596101550396 1.914962016075127; 1.336167412870625 1.7841783523160701; 2.1316903332108548 2.070020786141772; 22.39101301625161 1.978948524008241; 1.561349406762892 1.519393975639119;;; 2.404504503095904 1.4031065330396517; 0.8966193694810042 1.3620238902861248; 2.6077379220636536 2.5750830728928618; 1.8173218571834053 1.6810410177979684; 1.8749567890125762 1.6783401450831954; 1.6423637455139917 2.016762119805396; 1.7969783988091441 2.1383936450072554; 24.992510831627627 1.811097271612404;;; 1.6781307244254635 23.06615344979792; 1.8606468773700984 1.5841971429705124; 2.0311992328897457 1.9338870960069763; 2.1343766428293294 1.248234723033903; 2.076670520694236 2.2996485106061213; 1.3149348880754863 2.2420910737723068; 1.6174954185932466 1.35718409943008; 1.4884787955563157 1.9458528378491136;;; 1.9068540680735213 1.4547559646805126; 2.142662300171415 22.242323091239317; 1.5915186059737885 2.2213630494012593; 1.8555320642939483 2.0989394429778283; 1.741640201634547 1.7316300047893478; 1.8117819065414886 1.8634699128674659; 2.3247726275642466 1.5162131516812225; 1.3179549745127892 1.6273687373108647;;; 2.114878990755023 1.5705385320957943; 1.355701043052334 1.9874556637800755; 2.0774404861312026 22.35325261144647; 2.0603545647454222 2.2802779692859105; 2.102346551132414 1.6991682912587116; 2.529739960800079 1.925643823462419; 1.7584596985871626 1.7850799365765686; 2.297106771498343 1.855058009210872;;; 2.2658378971639013 1.0672210852548922; 2.2330602011149985 2.047366983488817; 2.0248526423426574 2.462612895418082; 2.2385137647507074 19.87936221856337; 1.3305983376394634 1.4620761582488084; 2.127975367466302 2.177406756366526; 2.0549496328590307 1.8491222517430748; 1.585399642535621 1.1220413233790567;;; 1.40327032622716 2.194630076563873; 1.9633418617179632 1.7560527490370994; 1.8089422937178607 1.9574984211276463; 2.1452301134390064 1.5380713619855713; 1.7555985138970267 21.750408771453998; 1.9422355487488296 2.499271967544096; 2.0001611728367377 1.429188939620812; 1.6586939735576123 2.400043368618653;;; 2.1320702977024375 1.9681430049427295; 1.840998381678337 1.7189630223278891; 1.8075983463563345 2.0150443185440245; 1.9323884327098058 2.0844723253159594; 1.2477280496739762 2.3303423327567674; 1.6659553969348087 24.550405093043793; 2.1170739718751097 1.709122341661194; 1.828186313492483 2.0321658320696714;;; 1.9742028187950875 1.3420459700506908; 1.3523103446237033 1.6305162005918328; 1.771587336454654 2.1332903711083615; 1.8160754044857377 2.014997760142696; 1.7358148705094456 1.5190692442836706; 1.9391408741123661 1.9679322811113815; 2.2848116491917665 22.30117510457243; 2.2086277781445305 1.4531085536273929;;; 1.797292375723173 2.0025297373478272; 1.7424064602701879 1.8134868150995778; 1.560100269424127 2.275083472620768; 1.3273825535019634 1.3597318606567805; 2.022945374351959 2.561738702159615; 1.4780261523102283 2.3627908003979616; 1.8970721297007478 1.5249235825054959; 1.9531464336277813 23.210356187422697;;; 0.0 1.519170623944797; 1.7832999900872273 1.5023583317370062; 1.0340945785515991 1.6239559474039198; 2.134678937777304 1.8961208078993737; 2.162689569211449 2.314077237774653; 1.8203332862608521 1.5422595733419437; 1.9880780402381675 1.430059480884302; 1.4046236309051332 1.2805221746947342;;; 1.090961496375447 1.4870996131030054; 0.0 1.463432571275598; 1.8877398274139012 3.177742468072387; 1.8112750221410379 1.4102354759841813; 1.9074435072381883 1.4281768450858716; 1.2941283652238658 1.7546699796331322; 1.851020709690133 2.467934814008157; 2.8756209704008326 1.7693206668810026;;; 1.6819487123745793 1.206119995057665; 1.9389560077255545 2.3735640740874224; 0.0 2.0718418402929655; 2.0270183569380915 1.3255671220380287; 2.2415268813208122 1.8110657437186224; 2.082175188604266 2.2374955878527176; 1.7551856896315892 1.389930148921427; 1.0167500383093693 1.6913581147306929;;; 2.4372373910466347 1.7651467699003116; 1.2281959132783826 1.7356334988133801; 1.7356574839366155 2.012711943333448; 0.0 1.6773782305571812; 2.006734217271948 1.3639607634685067; 1.3448160500580604 2.3460778134400364; 1.4799951967981544 2.4291803638347726; 2.057126159368682 2.231691836419036;;; 1.957844121820944 1.4293703641426134; 2.1235667437546906 2.131975657187372; 2.7165317282068466 2.135357751676152; 2.2927923335126392 2.5682159921494985; 0.0 1.956510637511225; 2.320282593850926 2.9301997301643787; 2.571149850891605 2.2327786355184807; 1.669583246222932 1.4116871056993214;;; 1.6216797730906631 2.3916318958093616; 1.3579774388121817 2.636571430240177; 2.318233299917687 1.4138074328772348; 1.5023681516864122 1.5928485375870804; 1.5516775667158338 1.7394807301724577; 0.0 2.6823986792782075; 2.103262323132724 1.6354946761484275; 1.7611056075407845 1.5754034458698558;;; 1.8008395632091267 2.0073782461741962; 1.0431135715235489 1.3493155345154608; 1.58030728750816 2.213696849389752; 2.143671824078984 1.8946887952679992; 1.737521803705314 1.983343085529205; 1.602480916987276 2.1408257421770625; 0.0 1.0819895190769018; 2.4100122816079126 2.1742416895639525;;; 1.6684834483402118 1.6808092994760968; 2.3706525690829676 2.450211533403086; 1.145581556785971 1.3627413326996076; 2.528087703176973 2.299127923407093; 2.0041520277237432 2.3705714524020625; 1.9293801824443548 1.8667896152805317; 2.7486192985762004 1.7911719395036294; 0.0 1.6085416074022647;;; 1.9758935653660703 0.0; 1.5770941856320115 2.0488139304021544; 1.602563129135857 2.8362079732492664; 2.6529455105093587 2.713727941018299; 1.159065260480117 1.519555831845985; 2.1641620809693802 1.190310625616825; 2.3299030154822837 1.871624157236874; 1.9997158292615576 1.2754088623207953;;; 2.5978748166845507 1.5290987967302314; 1.5064891763048875 0.0; 2.020600415839693 1.9173617907795157; 2.3596101420862947 1.8824358626993543; 1.8012365518548654 2.186986233747961; 1.8802949709095724 2.139499081020208; 2.0231572269406763 1.803461477244128; 2.3595811720460316 1.841030578404912;;; 1.8995465246664667 1.749513841697231; 2.1669030299743754 1.70644343300759; 1.2721696479923155 0.0; 1.7313224358216521 1.734623709421513; 1.7703925954960864 1.846647279150139; 0.9555004319249764 2.0150337701910925; 2.1363093619465467 2.207995207523941; 1.200683285561528 1.858146236860167;;; 0.998358938781891 2.299132058799989; 2.046973975577541 2.394532467890648; 1.609361227799718 1.7042177571250612; 2.701189481175924 0.0; 2.198260855265386 2.1498045291984287; 1.007982388035535 1.9129818285049338; 1.9677155112207327 1.5343692520254977; 2.101519845673146 2.7990695934643863;;; 2.7470036001196987 1.3958715696003863; 2.0498463788037404 1.954502894497896; 1.964861072924773 2.28787746875346; 1.8209344786098005 2.1587381722981984; 1.714229249642448 0.0; 1.7675901731100134 1.6817390782875787; 0.5374248998774477 2.021965134607554; 2.0774223752714502 0.7260592113495432;;; 1.1630574889081162 1.3057375342364435; 1.6764644751622813 1.90841733487611; 2.2961974567722514 1.5803170787820626; 2.300213160939175 1.2430436648603012; 2.0851674971937837 1.2752363506039608; 1.8810341177177357 0.0; 2.4124491517083926 1.7741946388573424; 1.7931099518532694 1.7937316289873904;;; 1.0647501993381738 2.1943131736903125; 2.3852646555225614 2.4059818092931584; 2.0418572367566354 2.6751939052384164; 2.0954063019405713 1.6287612630189847; 1.6810181589354314 2.4764315602541522; 2.341901917596953 2.225156209280123; 2.0000279282384086 0.0; 1.583363962512409 1.9556639923991788;;; 2.33938815974115 1.418675401245863; 2.738921926253877 1.8060037122977102; 1.9160938852857001 1.71225698210749; 3.1454629947898445 2.2375258767098605; 1.9920447317292596 1.574813839030247; 2.0102359423981344 1.864275839305733; 2.1182567429330734 2.0572329256113027; 1.751086472826442 0.0]
    MY = [1.8085556385124137 1.7953776740315455; 1.8414127842661827 1.8403600612668496; 1.8688461336064663 2.0619443989953696; 2.0042839049623544 1.8973823868397781; 1.7156454852430747 1.8699969464625366; 1.8103587289251535 2.0532321557780535; 2.09619165798559 1.7537762397623187; 1.874316195757285 1.7591105168802978]
    MZ = [1.26723829950154 0.6727096064374787 2.2152435305495493; 3.0986377097936217 3.4229623321850227 1.6827441306529918; 2.8331247279184395 0.4899771169412037 2.331676836964725; 1.8849192189104738 3.2395724551885237 1.6774485560558825; 3.091414942577444 3.448985827101145 2.925740192946321; 3.997679285884835 3.4972010509636995 1.7894377023726833; 2.7567825589529877 0.6524777589354528 0.021757217342317045; 0.28993486898040816 2.8774505070145464 0.17824891309650814]
end
macro primobj_code() #  entail (u, v, x, Y, Z)
    return esc(quote
    JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
        JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
        JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
        JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
        JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
        JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
        JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
        JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
        JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
        JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
        JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
        ) # 🌸
        JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
        JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
        JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
        JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
        JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe))
    end)
end
macro dualobj_code() #  entail (u, v, x, Y, Z)
    return esc(quote
    JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
        JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
        JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
        JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
        JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
        JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
        JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
        JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
        JuMP.@constraint(ø, p1[g = 1:G], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0.) # 🍀
        JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0.) # 🍀
        JuMP.@constraint(ø, pT[g = 1:G], pCom[T, g] + Du[T, g] - Dd[T, g] == 0.) # 🍀
        JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
        JuMP.@constraint(ø, psl1, pslCom[1] + Du1[G+1] - Dd1[G+1] + Dd[1+1, G+1] - Du[1+1, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, psl2[t = 2:T-1], pslCom[t] + Du[t, G+1] - Dd[t, G+1] + Dd[t+1, G+1] - Du[t+1, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, pslT, pslCom[T] + Du[T, G+1] - Dd[T, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
            -ip(Y, Dvp) - ip(Z, Dzt) + ip(Dd1 .- Du1, ZP)
            + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
            - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
            - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
            - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
            - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1)
        )
    end)
end
macro stage1feas_code()
    return esc(quote
        JuMP.@variable(ø, u[t = 1:T, g = 1:G+1], Bin)
        JuMP.@variable(ø, v[t = 1:T, g = 1:G+1], Bin)
        JuMP.@variable(ø, x[t = 1:T, g = 1:G+1], Bin)
        JuMP.@expression(ø, xm1, vcat(transpose(ZS), x)[1:end-1, :])
        JuMP.@constraint(ø, x .- xm1 .== u .- v)
        JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-UT+1], sum(x[i, g] for i in t:t+UT-1) >= UT * u[t, g])
        JuMP.@constraint(ø, [g = 1:G+1, t = T-UT+1:T], sum(x[i, g] - u[t, g] for i in t:T) >= 0)
        JuMP.@constraint(ø, [g = 1:G+1, t = 1:T-DT+1], sum(1 - x[i, g] for i in t:t+DT-1) >= DT * v[t, g])
        JuMP.@constraint(ø, [g = 1:G+1, t = T-DT+1:T], sum(1 - x[i, g] - v[t, g] for i in t:T) >= 0)
        # JuMP.@constraint(ø, u + v .<= 1) # 💥 become dispensable when UTDT are present
    end)
end
macro addMatVarViaCopy(x, xΓ) return esc(:( JuMP.@variable(ø, $x[eachindex(eachrow($xΓ)), eachindex(eachcol($xΓ))]) )) end # macro addMatCopyConstr(cpx, x, xΓ) return esc(:( JuMP.@constraint(ø, $cpx[i = eachindex(eachrow($x)), j = eachindex(eachcol($x))], $x[i, j] == $xΓ[i, j]) )) end
function decode_uv_from_x(x::BitMatrix)
    xm1 = vcat(transpose(ZS), x)[1:end-1, :]
    dif = Int.(x .- xm1)
    u = dif .== 1
    v = dif .== -1
    return u, v
end
function dualobj_value(u, v, x, Y, Z) # ofv
    ø = JumpModel(0)
    @dualobj_code()
    JuMP.@objective(ø, Max, dualobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(ø)
end
function dualobj_value(x, yM, i, Z) # wrapper
    u, v = decode_uv_from_x(x)
    return value = dualobj_value(u, v, x, yM[:, :, i], Z) 
end
function primobj_value(u, v, x, Y, Z) # ofv
    ø = JumpModel(0)
    @primobj_code()
    JuMP.@objective(ø, Min, primobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    JuMP.objective_value(ø)
end
function primobj_value(x, yM, i, Z) # wrapper
    u, v = decode_uv_from_x(x)
    return value = primobj_value(u, v, x, yM[:, :, i], Z) 
end# 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄
function master() # initialization version ⚠️ will be executed more than once
    ø = JumpModel(0)
    @stage1feas_code()
    JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
    JuMP.@variable(ø, -1/length(MY) <= β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))] <= 1/length(MY))
    JuMP.@objective(ø, Min, o1 + ip(MY, β1))
    @optimise()
    @assert status == JuMP.OPTIMAL
    return x, β1 = get_bin_var(x), jv(β1)
end
function master(ℶ1) # portal
    function readCut(ℶ) # for Benders (or SB) cut
        stV2, cnV2, puV2, pvV2, pxV2, pβ1V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2 = readCut(ℶ1)
    if R2 >= 1
        return vldt, x, β1, cost1plus2, oℶ1 = master(R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2)
    else # This part is necessary, because it will be executed more than once
        x, β1 = master()
        (vldt = false; oℶ1 = -Inf; cost1plus2 = 0.) # cost1plus2 is a finite value
        return vldt, x, β1, cost1plus2, oℶ1
    end
end
function master(R2, stV2, cnV2, puV2, pvV2, pxV2, pβ1V2) # if o3 has ≥1 cut
    ø = JumpModel(0)
    @stage1feas_code()
    JuMP.@expression(ø, o1, ip(brcs(CST), u) + ip(brcs(CSH), v))
    JuMP.@variable(ø, β1[eachindex(eachrow(MY)), eachindex(eachcol(MY))]) # free, but TBD
    JuMP.@expression(ø, o2, ip(MY, β1))
    JuMP.@variable(ø, o3)
    for r in 1:R2
        if stV2[r]
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pβ1V2[r], β1)] # modify this line
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o3 >= cut_expr)
        end
    end
    JuMP.@objective(ø, Min, o1 + o2 + o3)
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        @assert status == JuMP.DUAL_INFEASIBLE
        vldt = false
        (JuMP.set_lower_bound.(β1, -btBnd); JuMP.set_upper_bound.(β1, btBnd))
        @optimise()
        @assert status == JuMP.OPTIMAL
    else
        vldt = true
    end
    x, β1 = get_bin_var(x), jv(β1)
    cost1plus2, oℶ1 = JuMP.value(o1) + JuMP.value(o2), JuMP.value(o3)
    return vldt, x, β1, cost1plus2, oℶ1
end
function get_trial_β2_oℶ2(ℶ2, x, yM, iY) #  invoke next to argmaxY
    function readCut(ℶ)
        stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pY"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCut(ℶ2)
    ø = JumpModel(0)
    JuMP.@variable(ø, -btBnd <= β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))] <= btBnd) # 🧪 generate limited trial β2
    if R2 == 0
        JuMP.@objective(ø, Min, ip(MZ, β2))
    else
        JuMP.@variable(ø, o2)
        u, v = decode_uv_from_x(x)
        for r in 1:R2
            if stV2[r]
                Y = yM[:, :, iY]
                tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], Y), (pβ2V2[r], β2)]
                cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
                JuMP.drop_zeros!(cut_expr)
                JuMP.@constraint(ø, o2 >= cut_expr)
            end
        end
        JuMP.@objective(ø, Min, ip(MZ, β2) + o2)
    end
    @optimise()
    @assert status == JuMP.OPTIMAL " in get_trial_β2(): $status "
    oℶ2 = R2 == 0 ? -Inf : JuMP.value(o2)
    β2 = jv(β2)
    return β2, oℶ2
end
ub_φ1(Δ2, x, β1, yM, i) = -ip(β1, yM[:, :, i]) + ub_psi(Δ2, x, i)
phi_2(β2, x, yM, i, Z)  = -ip(β2, Z)           + f(x, yM, i, Z)
function f(x, yM, i, Z) # 🧊 dualFormulas inside
    ofc = ip(CL, Z)
    ofv = primobj_value(x, yM, i, Z)
    return of = ofc + ofv # 🧊 if you are not confident about dualFormulas.jl, execute the following line to check up on
    ofv2 = dualobj_value(x, yM, i, Z)
    @assert isapprox(ofv, ofv2; rtol = 1e-5) "ofv = $ofv | $ofv2 = ofv2" # to assure the validity of most hazardous Z
end
function ub_psi(Δ2, x::BitMatrix, Y::Int)::Float64
    i_vec = findall(t -> t == x, Δ2["x"]) ∩ findall(t -> t == Y, Δ2["Y"])
    isempty(i_vec) && return Inf
    R2 = length(i_vec)
    fV2 = Δ2["f"][i_vec]
    β2V2 = Δ2["β"][i_vec]
    ø = JumpModel(0)
    JuMP.@variable(ø, λ[1:R2] >= 0.)
    JuMP.@constraint(ø, sum(λ) == 1.)
    JuMP.@variable(ø, β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))])
    JuMP.@constraint(ø, sum(β2V2[r] * λ[r] for r in 1:R2) .== β2)
    JuMP.@objective(ø, Min, ip(MZ, β2) + ip(fV2, λ))
    @optimise()
    if status != JuMP.OPTIMAL
        @reoptimise()
        status == JuMP.INFEASIBLE && return Inf
        error(" in ub_psi(), status = $status ")
    end
    return JuMP.objective_value(ø)
end
function evalPush_Δ2(β2, x, yM, i, Z) # 👍 valid when Z is the argmax, not merely a feasible solution
    push!(Δ2["f"], phi_2(β2, x, yM, i, Z))
    push!(Δ2["x"], x)
    push!(Δ2["Y"], i)
    push!(Δ2["β"], β2)
end
function argmaxindY(Δ2, x, β1, yM)::Int # NOTE: it's ret(Y) is different from the underlying true argmaxY due to inexact Δ2. Nonetheless, the resultant ub is valid.
    (NY = size(yM, 3); fullVec = zeros(NY))
    for i in 1:NY
        v = ub_φ1(Δ2, x, β1, yM, i)
        v == Inf && return i
        fullVec[i] = v
    end
    return findmax(fullVec)[2]
end
function argmaxZ(u, v, x, Y, β2) # 💻 Feat
    ø = JumpModel(2)
    JuMP.@variable(ø, 0 <= Z[t = 1:T, l = 1:L] <= LM[l])
    @dualobj_code()
    JuMP.@objective(ø, Max, ip(Z, CL .- β2) + dualobj)
    @optimise() # JuMP.unset_silent(ø)
    @assert status == JuMP.OPTIMAL
    return jv(Z)
end
function argmaxZ(x, yM, i, β2)::Matrix{Float64}
    u, v = decode_uv_from_x(x)
    return argmaxZ(u, v, x, yM[:, :, i], β2)
end# 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄
function ψ_argmaxppo_master(oψ, u, v, x) #  🫖 arg is a trial point
    rsp(matrix) = reshape(matrix, (:,))
    function read(ℸ) # for Benders (or SB) cut
        oψV2, uV2, vV2, xV2 = ℸ["oψ"], ℸ["u"], ℸ["v"], ℸ["x"]
        R2 = length(oψV2)
        return R2, oψV2, uV2, vV2, xV2
    end
    R2, oψV2, uV2, vV2, xV2 = read(ℸ1)
    ϵ, HYPER_PARAM = 1e-5, 1.0 
    ø = JumpModel(0)
    JuMP.@variable(ø, po >= ϵ)
    @addMatVarViaCopy(pu, u)
    @addMatVarViaCopy(pv, v)
    @addMatVarViaCopy(px, x)
        JuMP.@expression(ø, pri,  vcat(rsp(u),  rsp(v),  rsp(x)))
        JuMP.@expression(ø, pai, vcat(rsp(pu), rsp(pv), rsp(px)))
        JuMP.@variable(ø, api[eachindex(pai)])
        JuMP.@constraint(ø, api .>=  pai)
        JuMP.@constraint(ø, api .>= -pai)
        JuMP.@constraint(ø, HYPER_PARAM * po + sum(api) <= 1.)
    JuMP.@expression(ø, o2, -ip(pai, pri))
    JuMP.@expression(ø, o3, -po * oψ)
    if R2 == 0   
        JuMP.@objective(ø, Max, o2 + o3)
    else
        JuMP.@variable(ø, o1)
        for r in 1:R2
            tmp = [(oψV2[r], po), (uV2[r], pu), (vV2[r], pv), (xV2[r], px)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp)) # anonymous because in a loop
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o1 <= cut_expr)
        end
        JuMP.@objective(ø, Max, o1 + o2 + o3)
    end
    @optimise() # Linear Program
    @assert status == JuMP.OPTIMAL
    po = JuMP.value(po)
    @assert po > ϵ/2 "Gurobi's err"
    o1 = R2 == 0 ? Inf : JuMP.value(o1)
    primVal_2 = JuMP.value(o2) + JuMP.value(o3)
    pu = jv(pu)
    pv = jv(pv)
    px = jv(px)
    return o1, primVal_2, po, pu, pv, px
end
function ψ_try_gen_vio_lag_cut(yM, iY, oψΓ, uΓ, vΓ, xΓ) # 🥑
    function pushTrial(oψ, u, v, x)
        push!(ℸ1["oψ"], oψ)
        push!(ℸ1["u"], u)
        push!(ℸ1["v"], v)
        push!(ℸ1["x"], x)
    end
    [empty!(ℸ1[k]) for k in keys(ℸ1)] # ∵ param Y may vary
    if oψΓ == -Inf
        cn = ψ_lag_subproblem(yM, iY, 1., zero(uΓ), zero(vΓ), zero(xΓ))[1] 
        return cn, 1., zero(uΓ), zero(vΓ), zero(xΓ) # generate a horizontal cut, although cn might be -Inf
    end
    PATIENCE = 0.9
    while true
        o1, primVal_2, po, pu, pv, px = ψ_argmaxppo_master(oψΓ, uΓ, vΓ, xΓ)
        dualBnd = o1 + primVal_2
        dualBnd < cϵ && return -Inf, 1., zero(uΓ), zero(vΓ), zero(xΓ)
        cn, (oψ, u, v, x) = ψ_lag_subproblem(yM, iY, po, pu, pv, px)
        cn > o1 + 5e-5 && @warn "ψ_try_gen_vio_lag_cut cn=$cn | $o1=o1"
        primVal = cn + primVal_2
        primVal > PATIENCE * dualBnd && return cn, po, pu, pv, px
        pushTrial(oψ, u, v, x)
    end
end
function ψ_lag_subproblem(yM, iY, po, pu, pv, px) # 🩳 according to the incumbent ℶ2
    function readCut(ℶ)
        stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = ℶ["st"], ℶ["cn"], ℶ["pu"], ℶ["pv"], ℶ["px"], ℶ["pY"], ℶ["pβ"]
        R2 = length(cnV2)
        return R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2
    end
    R2, stV2, cnV2, puV2, pvV2, pxV2, pYV2, pβ2V2 = readCut(ℶ2)
    ø = JumpModel(0)
    @stage1feas_code()
    tmp = [(u, pu), (v, pv), (x, px)]
    JuMP.@expression(ø, o1, mapreduce(t -> ip(t[1], t[2]), +, tmp))
    JuMP.drop_zeros!(o1)
    JuMP.@variable(ø, β2[eachindex(eachrow(MZ)), eachindex(eachcol(MZ))]) # must be free
    JuMP.@variable(ø, objℶ2)
    for r in 1:R2
        if stV2[r]   
            tmp = [(cnV2[r], 1.), (puV2[r], u), (pvV2[r], v), (pxV2[r], x), (pYV2[r], yM[:, :, iY]), (pβ2V2[r], β2)] # Y is fixed as a param
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp))
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, objℶ2 >= cut_expr)
        end
    end
    JuMP.@expression(ø, oψ, ip(MZ, β2) + objℶ2) # oψ is the essential 2nd stage convex function, resembling ofv 
    JuMP.@objective(ø, Min, o1 + po * oψ)
    @optimise()
    if status == JuMP.OPTIMAL
        cn = JuMP.objective_value(ø)
        oψ = JuMP.value(oψ)
        u = jv(u)
        v = jv(v)
        x = jv(x)
        return cn, (oψ, u, v, x) # [1] RHS of lag cut [2] ℸ requisites
    elseif status == JuMP.INFEASIBLE_OR_UNBOUNDED
        JuMP.set_attribute(ø, "DualReductions", 0)
        @optimise()
    end
    @assert status == JuMP.DUAL_INFEASIBLE "$status"
    cn = oψ = -Inf
    u, v, x = zero(pu), zero(pv), zero(px)
    return cn, (oψ, u, v, x)
end
function gencut_ψ_uvx(yM, iY, oψ, u, v, x)
    cn, po, pu, pv, px = ψ_try_gen_vio_lag_cut(yM, iY, oψ, u, v, x) # lag cut
    cn =  cn / po
    pu = -pu / po
    pv = -pv / po
    px = -px / po
    return cn, pu, pv, px # Ben cut
end
function gencut_ℶ1(yM, iY, oψ, u, v, x)
    cn, pu, pv, px = gencut_ψ_uvx(yM, iY, oψ, u, v, x)
    pβ1 = -yM[:, :, iY]
    return cn, pu, pv, px, pβ1
end
function tryPush_ℶ1(yM, iY, oψ, u, v, x)
    cn, pu, pv, px, pβ1 = gencut_ℶ1(yM, iY, oψ, u, v, x)
    cn == -Inf && return false # no cut can be generated
    push!(ℶ1["st"], true)
    push!(ℶ1["x"], x)
    push!(ℶ1["rv"], iY)
    push!(ℶ1["cn"], cn)
    push!(ℶ1["pu"], pu)
    push!(ℶ1["pv"], pv)
    push!(ℶ1["px"], px)
    push!(ℶ1["pβ"], pβ1)
    return true
end
function tryPush_ℶ1(yM, iY, oℶ1, x, β1)::Bool # 👍 use this directly
    oψ = oℶ1 + ip(β1, yM[:, :, iY])
    u, v = decode_uv_from_x(x)
    return success_flag = tryPush_ℶ1(yM, iY, oψ, u, v, x)
end
function argmaxppo_master(ofv, u, v, x, Y) # 🫖 arg is a trial point
    rsp(matrix) = reshape(matrix, (:,))
    function read(ℸ) # for Benders (or SB) cut
        ofvV2, uV2, vV2, xV2, YV2 = ℸ["ofv"], ℸ["u"], ℸ["v"], ℸ["x"], ℸ["Y"]
        R2 = length(ofvV2)
        return R2, ofvV2, uV2, vV2, xV2, YV2
    end
    R2, ofvV2, uV2, vV2, xV2, YV2 = read(ℸ2)
    ϵ, HYPER_PARAM = 1e-5, 1.0 
    ø = JumpModel(0)
    JuMP.@variable(ø, po >= ϵ)
    @addMatVarViaCopy(pu, u)
    @addMatVarViaCopy(pv, v)
    @addMatVarViaCopy(px, x)
    @addMatVarViaCopy(pY, Y)
        JuMP.@expression(ø, pri,  vcat(rsp(u),  rsp(v),  rsp(x),  rsp(Y)))
        JuMP.@expression(ø, pai, vcat(rsp(pu), rsp(pv), rsp(px), rsp(pY)))
        JuMP.@variable(ø, api[eachindex(pai)])
        JuMP.@constraint(ø, api .>=  pai)
        JuMP.@constraint(ø, api .>= -pai)
        JuMP.@constraint(ø, HYPER_PARAM * po + sum(api) <= 1.)
    JuMP.@expression(ø, o2, -ip(pai, pri))
    JuMP.@expression(ø, o3, -po * ofv)
    if R2 == 0
        JuMP.@objective(ø, Max, o2 + o3)
        vldt = false
    else
        JuMP.@variable(ø, o1)
        for r in 1:R2
            tmp = [(ofvV2[r], po), (uV2[r], pu), (vV2[r], pv), (xV2[r], px), (YV2[r], pY)]
            cut_expr = JuMP.@expression(ø, mapreduce(t -> ip(t[1], t[2]), +, tmp)) # anonymous because in a loop
            JuMP.drop_zeros!(cut_expr)
            JuMP.@constraint(ø, o1 <= cut_expr)
        end
        JuMP.@objective(ø, Max, o1 + o2 + o3)
        vldt = true
    end
    @optimise() # Linear Program
    @assert status == JuMP.OPTIMAL
    po = JuMP.value(po)
    @assert po > ϵ/2 "Gurobi's err"
    o1 = R2 == 0 ? Inf : JuMP.value(o1)
    primVal_2 = JuMP.value(o2) + JuMP.value(o3)
    pu = jv(pu)
    pv = jv(pv)
    px = jv(px)
    pY = jv(pY)
    return o1, primVal_2, po, pu, pv, px, pY
end
function try_gen_vio_lag_cut(yM, Z, ofvΓ, uΓ, vΓ, xΓ, YΓ) # 🥑 use suffix "Γ" to avoid clash
    function pushTrial(ofv, u, v, x, Y)
        push!(ℸ2["ofv"], ofv)
        push!(ℸ2["u"], u)
        push!(ℸ2["v"], v)
        push!(ℸ2["x"], x)
        push!(ℸ2["Y"], Y)
    end
    [empty!(ℸ2[k]) for k in keys(ℸ2)] # ∵ param Z may vary
    if ofvΓ == -Inf
        cn = lag_subproblem(yM, Z, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ))[1]
        return cn, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ)
    end
    PATIENCE = 0.9
    while true
        o1, primVal_2, po, pu, pv, px, pY = argmaxppo_master(ofvΓ, uΓ, vΓ, xΓ, YΓ) # trial slope <-- hat points
        dualBnd = o1 + primVal_2
        dualBnd < cϵ && return -Inf, 1., zero(uΓ), zero(vΓ), zero(xΓ), zero(YΓ)
        cn, (ofv, u, v, x, Y) = lag_subproblem(yM, Z, po, pu, pv, px, pY) # trial cut generation
        cn > o1 + 5e-5 && @warn "final try_gen_vio_lag_cut cn=$cn | $o1=o1"
        primVal = cn + primVal_2
        primVal > PATIENCE * dualBnd && return cn, po, pu, pv, px, pY
        pushTrial(ofv, u, v, x, Y)
    end
end
function lag_subproblem(yM, Z, po, pu, pv, px, pY) # 🩳 the core of "gen cut for ofv"
    ø = JumpModel(0)
    @stage1feas_code()
    @addMatVarViaCopy(Y, pY)
    tmp = [(u, pu), (v, pv), (x, px), (Y, pY)]
    JuMP.@expression(ø, o1, mapreduce(t -> ip(t[1], t[2]), +, tmp))
    JuMP.drop_zeros!(o1)
    NY = size(yM, 3)
    JuMP.@variable(ø, ly[i = 1:NY], Bin)
    JuMP.@constraint(ø, sum(ly) == 1)
    JuMP.@constraint(ø, sum(ly[i] * yM[:, :, i] for i in 1:NY) .== Y)
    @primobj_code() # 'Z' is fixed
    JuMP.@objective(ø, Min, o1 + po * primobj)
    @optimise()
    @assert status == JuMP.OPTIMAL
    cn = JuMP.objective_value(ø)
    ofv = JuMP.value(primobj)
    u = jv(u)
    v = jv(v)
    x = jv(x)
    Y = jv(Y)
    return cn, (ofv, u, v, x, Y) # [1] RHS of lag cut [2] ℸ requisites  
end
function gencut_ofv_uvxY(yM, Z, ofv, u, v, x, Y) # lag's format to Bens'
    cn, po, pu, pv, px, pY = try_gen_vio_lag_cut(yM, Z, ofv, u, v, x, Y)
    cn =  cn / po
    pu = -pu / po
    pv = -pv / po
    px = -px / po
    pY = -pY / po
    return cn, pu, pv, px, pY
end
function gencut_f_uvxY(yM, Z, ofv, u, v, x, Y) # append ofc
    cn, pu, pv, px, pY = gencut_ofv_uvxY(yM, Z, ofv, u, v, x, Y)
    cn += ip(CL, Z)
    return cn, pu, pv, px, pY
end
function gencut_f_uvxY(yM, i, Z, ofv, x) # a sheer wrapper for input style
    u, v = decode_uv_from_x(x)
    Y = yM[:, :, i]
    return gencut_f_uvxY(yM, Z, ofv, u, v, x, Y)
end
function gencut_ℶ2(yM, i, Z, ofv, x)
    cn, pu, pv, px, pY = gencut_f_uvxY(yM, i, Z, ofv, x)
    pβ2 = -Z
    return cn, pu, pv, px, pY, pβ2
end
function tryPush_ℶ2(yM, i, Z, ofv, x)
    cn, pu, pv, px, pY, pβ2 = gencut_ℶ2(yM, i, Z, ofv, x)
    cn == -Inf && return false # no cut can be generated
    push!(ℶ2["st"], true)
    push!(ℶ2["cn"], cn)
    push!(ℶ2["pu"], pu)
    push!(ℶ2["pv"], pv)
    push!(ℶ2["px"], px)
    push!(ℶ2["pY"], pY)
    push!(ℶ2["pβ"], pβ2)
    return true
end
function tryPush_ℶ2(yM, i, Z, oℶ2, x, β2)::Bool # 👍 use this directly
    ofv = oℶ2 + ip(β2, Z) - ip(CL, Z)
    return success_flag = tryPush_ℶ2(yM, i, Z, ofv, x)
end# 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄
_, x, β1, _, oℶ1 = master(ℶ1)
iY = argmaxindY(Δ2, x, β1, yM)
β2, oℶ2 = get_trial_β2_oℶ2(ℶ2, x, yM, iY)
Z = argmaxZ(x, yM, iY, β2)
gCnt = [0]
termination_flag = falses(1)
lbV = [-Inf] # to draw pictures
ubV, xV, β1V = [Inf], [x, x], [β1, β1] # current best value is ubV[1], current best solution is xV[2], β1V[2]
oℶ1V = [oℶ1]
iYV = [iY]
β2V, oℶ2V = [β2], [oℶ2]
ZV = [Z]
tV = [t1]
c1p2V = [Inf]
from_t1V = trues(1) # used in t2f
vldtV = falses(1)
function xbo1(is_premain::Bool)
    vldtV[1], xV[1], β1V[1], c1p2V[1], oℶ1V[1] = vldt, x, β1, cost1plus2, oℶ1 = master(ℶ1)
    if vldt
        @info "▶▶ master's vldt is true"
        return true
    end
    (from_t1V[1] = true; tV[1] = t2f)
    return false
end
function xbo1()
    vldtV[1], xV[1], β1V[1], c1p2V[1], oℶ1V[1] = vldt, x, β1, cost1plus2, oℶ1 = master(ℶ1)
    (from_t1V[1] = true; tV[1] = t2f)
end
function ybo2(yM)
    x, β1 = xV[1], β1V[1] 
    iYV[1] = iY = argmaxindY(Δ2, x, β1, yM) # NOTE: this iY may NOT be optimal w.r.t (x, β1) Nonetheless the resultant upper_bound_value is indeed valid
    if from_t1V[1] # 💻 concluding session
        cost1plus2, oℶ1, oℶ1_ub = c1p2V[1], oℶ1V[1], ub_φ1(Δ2, x, β1, yM, iY)
        @assert oℶ1 <= oℶ1_ub + 1e-6
        (lb = cost1plus2 + oℶ1; push!(lbV, lb))
        ub = cost1plus2 + oℶ1_ub # 1️⃣ this trial upper bound could deteriorate
        if ub < ubV[1] # 🥑 update candidate solution
            xV[2], β1V[2], ubV[1] = x, β1, ub # 🥑 if I take the 1st stage in xV[2], then the resultant DR cost will be no more than ubV[1]
        end
        ub = ubV[1] # 2️⃣ this upper bound is the best so far
        gap = gap_lu(lb, ub)
        lb, ub = rd6(lb), rd6(ub)
        str = "t1:[g$(gCnt[1])]($(vldtV[1]))lb $lb | $ub ub, gap $gap"
        @info str
        if gap < 8/1000
            @info " 😊 gap < 8/1000, thus terminate at next t3 "
            termination_flag[1] = true
        end
    end
    β2V[1], oℶ2V[1] = get_trial_β2_oℶ2(ℶ2, x, yM, iY)
    tV[1] = t3
end
function zb2d2(yM)
    x, iY, β2, oℶ2 = xV[1], iYV[1], β2V[1], oℶ2V[1]
    ZV[1] = Z = argmaxZ(x, yM, iY, β2)
    termination_flag[1] && return true
    tryPush_ℶ2(yM, iY, Z, oℶ2, x, β2) || @warn "t3: ℶ2 saturation"
    evalPush_Δ2(β2, x, yM, iY, Z)
    gCnt[1] += 1
    tV[1] = t2b
    return false
end
function yb1d1(yM)
    x, β1, oℶ1, iY = xV[1], β1V[1], oℶ1V[1], iYV[1] # ✅ reuse Y in FWD because it's faster in practice
    tryPush_ℶ1(yM, iY, oℶ1, x, β1) ? (tV[1] = t1) : (from_t1V[1] = false; tV[1] = t2f)
end
function main(yM, is_premain::Bool)
    while true # 🌸 to make master has lower_bound even if β1 is free
        t = tV[1]
        if t == t1
            xbo1(is_premain) && break
        elseif t == t2f
            ybo2(yM)
        elseif t == t3
            zb2d2(yM)
        elseif t == t2b
            yb1d1(yM)
        end
    end
end
function main(yM)
    while true # 🌸 to make master has lower_bound even if β1 is free
        t = tV[1]
        if t == t1
            xbo1()
        elseif t == t2f
            ybo2(yM)
        elseif t == t3
            zb2d2(yM) && break
        elseif t == t2b
            yb1d1(yM)
        end
    end
end

@time main(yM, true)
@time main(yM)

popfirst!(lbV)
using CairoMakie
fi = Figure();
axs = Axis.([fi[i...] for i in Iterators.product([1,2],[1,2])]; aspect = 1);
lines!(axs[1], eachindex(lbV), lbV; color = :navy)

function argmaxZ(u, v, x, Y, β2, gos1234) # ❌ This heuristic is not very effective! The format might be useful though
    function my_callback_function(cb_data, cb_where::Cint)
        vfun(x) = JuMP.callback_value(cb_data, x)
        function readcb(cb_data, cb_where)
            bstbndP = Ref{Cdouble}()
            feasvalP = Ref{Cdouble}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJBND, bstbndP)
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJ, feasvalP)
            return bstbndP[], feasvalP[]
        end
        cb_where == Gurobi.GRB_CB_MIPSOL || return
        go5p6ub, feasgo5p6 = readcb(cb_data, cb_where)
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        if gos1234 + feasgo5p6 > ubV[1] + 0.05
            go56_ubV[1] = go5p6ub
            feas_Z[1] = vfun.(Z)
            Gurobi.GRBterminate(JuMP.backend(ø))
        end
    end
    ø = JumpModel(2)
    JuMP.@variable(ø, 0 <= Z[t = 1:T, l = 1:L] <= LM[l])
    @dualobj_code()
    JuMP.@objective(ø, Max, ip(Z, CL .- β2) + dualobj)
        feas_Z, go56_ubV = [MZ], [Inf] # written by callback
        JuMP.MOI.set(ø, Gurobi.CallbackFunction(), my_callback_function)
        JuMP.unset_silent(ø)
    @optimise()
    if status == JuMP.INTERRUPTED
        go56ub = go56_ubV[1] # a value ≥ φ2(\hat_x2, \hat_β2) such that can be written into Δ2
        return go56ub, feas_Z[1] # a hazardous scene
    elseif status == JuMP.OPTIMAL
        return jv(Z) # the most hazardous scene
    end
    error("in argmaxZ: status = $status")
end


macro prim_skeleton()
    return esc(
        quote 
            JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
            JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
            JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
            JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
            JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
            JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
            JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
            JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
            JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2))
        end
    )
end
macro dual_skeleton()
    return esc(
        quote
            JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
            JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
            JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
            JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
            JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
            JuMP.@expression(ø, pslCom[t = 1:T], CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
            JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
            JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
            JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
            JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t])  >= 0.)
            JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t])           >= 0.)
            JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] >= 0.)
            JuMP.@expression(ø, dualobj,
                -ip(Y, Dvp) - ip(Z, Dzt)
                + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            )
        end
    )
end
macro prim_Quad()
    return esc(
        quote
            JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
            JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
            JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
            JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
            JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
            JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
            JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
            JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
            JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
            JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
            JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
            JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
            JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
            JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
            JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
            JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe))
        end
    )
end
macro dual_Quad()
    return esc(
        quote
            JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
            JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
            JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
            JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
            JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
            JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
            JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
            JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
            JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
            JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
            JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
            JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
            JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
            JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
            JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
            JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
            JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
            JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t])    >= 0.)
            JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t])             >= 0.)
            JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t]   >= 0.)
            JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
                -ip(Y, Dvp) - ip(Z, Dzt) 
                + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            )
        end
    )
end
macro prim_QuadBf()
    return esc(quote
        JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
        JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
        JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
        JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
        JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
        JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
        JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
        ) # 🌸
        JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
        JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
        JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
        JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
        JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe)) # okay
    end)
end
macro dual_QuadBf()
    return esc(quote
        JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
        JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
        JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
        JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
        JuMP.@constraint(ø, p1[g = 1:G],            pCom[1, g] == 0.) # 🍀
        JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] == 0.) # 🍀
        JuMP.@constraint(ø, pT[g = 1:G],            pCom[T, g] == 0.) # 🍀
        JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
        JuMP.@constraint(ø, psl1,                   pslCom[1] == 0.)  # 🍀slack
        JuMP.@constraint(ø, psl2[t = 2:T-1],        pslCom[t] == 0.)  # 🍀slack
        JuMP.@constraint(ø, pslT,                   pslCom[T] == 0.)  # 🍀slack
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
            -ip(Y, Dvp) - ip(Z, Dzt) 
            + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
        )
    end)
end
macro prim_code()
    return esc(quote
        JuMP.@variable(ø, p[t = 1:T, g = 1:G+1])
        JuMP.@variable(ø, ϱ[t = 1:T, g = 1:G] >= 0.) # G+1 @ ϱsl
        JuMP.@variable(ø, ϖ[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, ζ[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, pp[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, pe[t = 1:T, g = 1:G+1] >= 0.) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [pp[t, g] + 1, pp[t, g] - 1, 2 * p[t, g]] in JuMP.SecondOrderCone()) # 🍟 okay
        JuMP.@constraint(ø, De[t = 1:T, g = 1:G+1], pe[t, g] >= (C2[g] * pp[t, g] + C1[g] * p[t, g] + C0[g]) - EM[g] * (1 - x[t, g])) # 🍟
        JuMP.@expression(ø, ϱsl[t = 1:T], sum(ζ[t, :]) - sum(ϖ[t, :]) - sum(ϱ[t, g] for g in 1:G)) # 🍀 ϱ[t, G+1]
        JuMP.@constraint(ø, Dϱl[t = 1:T], ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dϱu[t = 1:T], p[t, G+1] - ϱsl[t] >= 0.) # 🍀
        JuMP.@constraint(ø, Dvp[t = 1:T, w = 1:W], Y[t, w] >= ϖ[t, w])
        JuMP.@constraint(ø, Dzt[t = 1:T, l = 1:L], Z[t, l] >= ζ[t, l])
        JuMP.@constraint(ø, Dvr[t = 1:T, g = 1:G], p[t, g] >= ϱ[t, g])
        JuMP.@constraint(ø, Dpi[t = 1:T, g = 1:G+1], p[t, g] >= PI[g] * x[t, g])
        JuMP.@constraint(ø, Dps[t = 1:T, g = 1:G+1], PS[g] * x[t, g] >= p[t, g])
        JuMP.@constraint(ø, Dd1[g = 1:G+1], p[1, g] - ZP[g] >= -RD[g] * x[1, g] - SD[g] * v[1, g])              # 🧊
        JuMP.@constraint(ø, Du1[g = 1:G+1], RU[g] * ZS[g] + SU[g] * u[1, g] >= p[1, g] - ZP[g])                 # 🧊
        JuMP.@constraint(ø, Dd[t = 2:T, g = 1:G+1], p[t, g] - p[t-1, g] >= -RD[g] * x[t, g] - SD[g] * v[t, g])  # 🧊
        JuMP.@constraint(ø, Du[t = 2:T, g = 1:G+1], RU[g] * x[t-1, g] + SU[g] * u[t, g] >= p[t, g] - p[t-1, g]) # 🧊
        JuMP.@expression(ø, bf[t = 1:T, b = 1:B],
            sum(FM[b, NG[g]] * ϱ[t, g] for g in 1:G)
            + sum(FM[b, NW[w]] * ϖ[t, w] for w in 1:W)
            - sum(FM[b, NL[l]] * ζ[t, l] for l in 1:L)
        ) # 🌸
        JuMP.@constraint(ø, Dbl[t = 1:T, b = 1:B], bf[t, b] >= -BC[b]) # 🌸
        JuMP.@constraint(ø, Dbr[t = 1:T, b = 1:B], BC[b] >= bf[t, b])  # 🌸
        JuMP.@expression(ø, lscost_2, -ip(CL, ζ))
        JuMP.@expression(ø, gccost_1, sum(CG[g]   * (p[t, g]   - ϱ[t, g]) for t in 1:T, g in 1:G))
        JuMP.@expression(ø, gccost_2, sum(CG[G+1] * (p[t, G+1] - ϱsl[t])  for t in 1:T))
        JuMP.@expression(ø, primobj, lscost_2 + (gccost_1 + gccost_2) + sum(pe))
    end)
end
macro dual_code()
    return esc(quote
        JuMP.@variable(ø, 0. <= De[t = 1:T, g = 1:G+1] <= 1.) # 🍟 ub is due to sum(pe)
        JuMP.@variable(ø, D1[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D2[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@variable(ø, D3[t = 1:T, g = 1:G+1]) # 🍟
        JuMP.@constraint(ø, [t = 1:T, g = 1:G+1], [D1[t, g], D2[t, g], D3[t, g]] in JuMP.SecondOrderCone()) # 🍟
        JuMP.@variable(ø, Dϱl[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dϱu[t = 1:T] >= 0.) # 🍀
        JuMP.@variable(ø, Dvp[t = 1:T, w = 1:W] >= 0.)
        JuMP.@variable(ø, Dzt[t = 1:T, l = 1:L] >= 0.)
        JuMP.@variable(ø, Dvr[t = 1:T, g = 1:G] >= 0.)
        JuMP.@variable(ø, Dpi[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dps[t = 1:T, g = 1:G+1] >= 0.)
        JuMP.@variable(ø, Dd1[g = 1:G+1] >= 0.)         # 🧊
        JuMP.@variable(ø, Du1[g = 1:G+1] >= 0.)         # 🧊
        JuMP.@variable(ø, Dd[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
        JuMP.@variable(ø, Du[t = 2:T, g = 1:G+1] >= 0.) # 🧊        
        JuMP.@variable(ø, Dbl[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@variable(ø, Dbr[t = 1:T, b = 1:B] >= 0.) # 🌸
        JuMP.@constraint(ø, pp[t = 1:T, g = 1:G+1], De[t, g] * C2[g] - D1[t, g] - D2[t, g] == 0.) # 🍟
        JuMP.@expression(ø, pCom[t = 1:T, g = 1:G], De[t, g] * C1[g] - 2 * D3[t, g] + CG[g] + Dps[t, g] - Dpi[t, g] - Dvr[t, g])
        JuMP.@constraint(ø, p1[g = 1:G], pCom[1, g] + Du1[g] - Dd1[g] + Dd[1+1, g] - Du[1+1, g] == 0.) # 🍀
        JuMP.@constraint(ø, p2[t = 2:T-1, g = 1:G], pCom[t, g] + Du[t, g] - Dd[t, g] + Dd[t+1, g] - Du[t+1, g] == 0.) # 🍀
        JuMP.@constraint(ø, pT[g = 1:G], pCom[T, g] + Du[T, g] - Dd[T, g] == 0.) # 🍀
        JuMP.@expression(ø, pslCom[t = 1:T], De[t, G+1] * C1[G+1] - 2 * D3[t, G+1] + CG[G+1] + Dps[t, G+1] - Dpi[t, G+1] - Dϱu[t])
        JuMP.@constraint(ø, psl1, pslCom[1] + Du1[G+1] - Dd1[G+1] + Dd[1+1, G+1] - Du[1+1, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, psl2[t = 2:T-1], pslCom[t] + Du[t, G+1] - Dd[t, G+1] + Dd[t+1, G+1] - Du[t+1, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, pslT, pslCom[T] + Du[T, G+1] - Dd[T, G+1] == 0.) # 🍀slack
        JuMP.@constraint(ø, ϱ[t = 1:T, g = 1:G], -CG[g] + Dvr[t, g] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NG[g]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ϖ[t = 1:T, w = 1:W], Dvp[t, w] + CG[G+1] - (Dϱu[t] - Dϱl[t]) + sum(FM[b, NW[w]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@constraint(ø, ζ[t = 1:T, l = 1:L], -CL[t, l] + Dzt[t, l] - CG[G+1] + Dϱu[t] - Dϱl[t] - sum(FM[b, NL[l]] * (Dbr[t, b] - Dbl[t, b]) for b in 1:B) >= 0.)
        JuMP.@expression(ø, dualobj, sum(D2 .- D1) + sum(De[t, g] * (C0[g] - EM[g] * (1 - x[t, g])) for t in 1:T, g in 1:G+1)
            -ip(Y, Dvp) - ip(Z, Dzt) + ip(Dd1 .- Du1, ZP)
            + sum((PI[g] * Dpi[t, g] - PS[g] * Dps[t, g]) * x[t, g] for t in 1:T, g in 1:G+1)
            - sum(BC[b] * (Dbl[t, b] + Dbr[t, b]) for t in 1:T, b in 1:B)
            - ip(Du1, RU .* ZS) - sum(Du[t, g] * RU[g] * x[t-1, g] for t in 2:T, g in 1:G+1)
            - sum((Du1[g] * u[1, g] + sum(Du[t, g] * u[t, g] for t in 2:T)) * SU[g] for g in 1:G+1)
            - sum((Dd1[g] * v[1, g] + sum(Dd[t, g] * v[t, g] for t in 2:T)) * SD[g] for g in 1:G+1)
            - sum((Dd1[g] * x[1, g] + sum(Dd[t, g] * x[t, g] for t in 2:T)) * RD[g] for g in 1:G+1)
        )
    end)
end


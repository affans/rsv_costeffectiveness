## model for running cost-effectiveness of RSV. 
using Distributions, StatsBase, Random, StaticArrays
using DelimitedFiles
using IterTools
using Base.Filesystem
using CSV, DataFrames
using Match
using Logging, ProgressMeter
import LatinHypercubeSampling.randomLHC
using ParallelDataTransfer

# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    age::Int64 = 0   # in years. don't really need this but left it incase needed later
    ageindays::Int64 = 0 # for new borns, we want to stratify it between 0:30 days, 30:180, 
    newborn::Bool = false # whether person is new born 
    monthborn::Int64 = 0 # month of the born baby
    activeaging::Bool = true # if active, the agent ages every month (this is to prevent newborns that have not been born yet to age)
    preterm::Bool = false
    gestation::Int64 = 0 # 1 = <29, 2 = 29-32 weeks, 3 = 33-35 weeks... only if preterm is true
    houseid::Int64 = 0
    sibling::Bool = false # we want to tag newborns that have a sibling <5.
    comorbidity::Int64 = 0 # comorbidity 1 = CHD, 2 = CLD (congenital heart disease, chronic lung disease)
    rsvpositive::Int64 = 0 
    rsvmonth::Vector{Int64} = Int64[]
    rsvage::Vector{Int64} = Int64[]
    rsvtype::Vector{Int64} = Int64[] # 1=MA, 2=NMA
    rsvhospitalized::Bool = false # a flag to see if person is hospitalized
    vac_lama::Bool = false # flags for if a person gets LAMA or is maternally immunized 
    vac_mat::Bool = false 
    eff_outpatient::Vector{Float64} = zeros(Float64, 24)
    eff_hosp::Vector{Float64} = zeros(Float64, 24)
    eff_icu::Vector{Float64} = zeros(Float64, 24)
    qalys::Dict{String, Float64} = Dict("symptomatic"  => 0.0, "pedward" => 0.0, "icu" => 0.0, "wheezing" => 0.0, "nonrsv" => 0.0)
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters    ## use @with_kw from Parameters
    popsize::Int64 = 100000
    modeltime::Int64 = 300
    numofsims::Int64 = 1000
    lhs_inpatient_preterm1::Matrix{Float64} = zeros(Float64, numofsims, 12) 
    lhs_inpatient_preterm2::Matrix{Float64} = zeros(Float64, numofsims, 12) 
    lhs_inpatient_preterm3::Matrix{Float64} = zeros(Float64, numofsims, 12) 
    lhs_inpatient_fullterm::Matrix{Float64} = zeros(Float64, numofsims, 12)
    lhs_mortality::Matrix{Float64} = zeros(Float64, numofsims, 6)
    l1_l4_coverage::Float64 = 0.90 # coverage for L1 to L4 
    #l5_coverage::Float64 = 0.80 # coverage for L5
    mat_coverage::Float64 = 0.60 # coverage for materal immunization 
    vaccine_eff_scenaro::String = "sigmoidal" # "sigmoidal" or "fixed" 
end

# constant variables
const vax_rng = MersenneTwister(422) # create a random number object... this is only for use in the vaccine functions
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
const ON_POP = 13448500 # 13448495 from ontario database, but their numbers don't add up
pc(x) = Int(round(x / ON_POP * 100000)) # 13_448_495: max population in ontario 
export humans

# setup fixed distributions for hosptialization and mortality, 
# these will get populated for each simulation by `set_outcome_probs`
const prob_inpatient_fullterm  = zeros(Float64, 12)
const prob_inpatient_preterm_1 = zeros(Float64, 12)
const prob_inpatient_preterm_2 = zeros(Float64, 12)
const prob_inpatient_preterm_3 = zeros(Float64, 12)
const prob_inpatient_preterm = [prob_inpatient_preterm_1, prob_inpatient_preterm_2, prob_inpatient_preterm_3]
const prob_death_rate = zeros(Float64, 6) #elements 1:3 are for preterm, element 4 is fullterm, element 5 and 6 are CHD/CLD#

function debug(sc, simulate_id)
    logger = NullLogger()
    global_logger(logger)
    reset_params()  # reset the parameters for the simulation scenario    
   
    p.vaccine_eff_scenaro = "fixed" 
    p.l1_l4_coverage = 1.0# 0.35 # 1.0 # 0.90 # coverage for L1 to L4 
    p.mat_coverage = 0.60  # coverage for materal immunization 
    
    Random.seed!(158) # start all simulations by setting a seed - ensures reproducibilty
    assign_inpatient_death_probs_to_modelparams() 

    # should be less than the number of simulations
    #simulate_id = 460 ## THIS WILL SIMULATE WITH THIS SEED
    Random.seed!(vax_rng, 5 * simulate_id)  # set the seed randomly for the vaccine functions
    Random.seed!(158 * simulate_id) 
    
    set_inpatient_and_death_probs(simulate_id)
    nb = demographics(); println("total newborns: $nb")
    chd, cld = apply_comorbidities()
    apply_qalys();
    tf = incidence(); println("total episodes sampled: $tf")
    vc = vaccine_scenarios(sc); println("vaccine cnts: $vc")
    nbs, ql, qd, rc, dc, inpats, outpats, non_ma, hospdays = outcome_analysis(simulate_id)
    println("total deaths: $dc")
    println("total qalys: $ql")
    return ql
end


# function to simulate a specific vaccine scenario
simulate_scenario(sc) = simulate_scenario(sc, 158)
function simulate_scenario(sc, _seed)
    #println("running simulation $(sc) on core id: $(myid())")
    sc_data = zeros(Float64, p.numofsims, 11)

    # dont run unneccessary simulations
    if sc ∉ [:s0, :s1, :s2, :s3, :s4, :s5, :s10]
        return sc_data
    end
    
    for i = 1:p.numofsims
        # set relevant seeds for each simulation 
        Random.seed!(vax_rng, 5 * i) # set the seed for vaccine function
        Random.seed!(_seed * i) # Set GLOBAL RNG SEED for each simulation so that the same number of newborns/preterms/incidence are sampled! 
        
        # set the inpatient/death probabilities (which are sampled through LHS)
        set_inpatient_and_death_probs(i)

        # run model functions
        nb = demographics() # initialize the human population 
        chd, cld = apply_comorbidities()
        apply_qalys()
        tf = incidence() # sample the total number of infected individuals
        lc, mc = vaccine_scenarios(sc) # apply vaccine dynamics
        nbs, ql, qd, rc, dc, inpats, outpats, non_ma, hospdays = outcome_analysis(i) # perform outcome analysis, pass in simulation number to set internal seed

        # store simulation data
        sc_data[i,  :] .= [nbs, lc, mc, rc, dc, ql, qd, inpats, outpats, non_ma, hospdays]
    end
    return sc_data 
end

# main run function -- if distributed procs are added, the scenarios are split over the cores
# each scenario resets the population on its core 
# I don't know if its possible to run simulations in parallel, but I don't think so
# The @everywhere does nothing if no procs are added so we don't need to separate code for the distributed version
# setup to use distributed 
# using Distributed; addprocs(N); @everywhere include("rsv.jl")

function get_file_name() 
    ef = p.vaccine_eff_scenaro 
    co = p.l1_l4_coverage == 0.80 ? "basecase" : "100"
    return "sims_$(ef)_$(co).csv"
end

function simulations() 
    println("starting simulations on core id: $(myid())")
    #return 
    @everywhere reset_params()  # reset the model parameters 

    # get a log file for simulations only -- save all the @info statement to files
    # io = open("log.txt", "w")
    # logger = SimpleLogger(io)
    # global_logger(logger)
    @everywhere logger = NullLogger()
    @everywhere global_logger(logger)

    @everywhere p.numofsims = 1000
    @everywhere p.vaccine_eff_scenaro = "fixed" 
    @everywhere p.l1_l4_coverage = 0.80 # 1.0 # 0.90 # coverage for L1 to L4 
    @everywhere p.mat_coverage = 0.60  # coverage for materal immunization 
    fname = get_file_name()

    # many of these scenarios are not used anymore in revisions, but we keep the file structure the same so Seyed doesn't have to adjust plotting scripts
    scenarios = [:s0, :s1, :s2, :s3, :s4, :s5, :s6, :s7, :s8, :s9, :s10]
    
    Random.seed!(158) # start simulations by setting a seed - ensures reproducibilty
    assign_inpatient_death_probs_to_modelparams() # for inpatient/death, generate samples of probabilities using LHS

    all_data = pmap(scenarios) do x 
        simulate_scenario(x, 158)
    end 
    
    # generate colnames for Seyed's preferred format
    colnames = ["_newborns", "_lama_cnt", "_mat_cnt", "_rsvcost", "_deathcost", "_totalqalys", "qalylost_death", "_inpatients", "_outpatients", "_non_ma", "_hospdays"]
    scnames = string.(scenarios)
    dfnames = vcat([sc .* colnames for sc in scnames]...)
    
    df = DataFrame(hcat(all_data...), dfnames)

    if gethostname() == "hpc"
        CSV.write("/data/$(fname)", df)
    else
        CSV.write("$(fname)", df)
    end 
    return df
end

function vaccine_scenarios(scenario) 
    lama_cnt = 0 
    mat_cnt = 0 

    # adjusted scenarios for revisions... the s10 scenario is a combination of LAMA(s1) + MAT
    if scenario in (:s1, :s2, :s3, :s4)
        _, cnt = lama_vaccine(scenario) 
        lama_cnt = cnt
        mat_cnt = 0 
    elseif scenario == :s5
        _, cnt = maternal_vaccine()
        lama_cnt = 0 
        mat_cnt = cnt 
    elseif scenario == :s10  # combination LAMA + MAT
        _, mat_cnt = maternal_vaccine()
        _, lama_cnt = lama_vaccine(:s1)
    end
    return lama_cnt, mat_cnt
end

#l# Iniltialization Functions 
reset_params() = reset_params(ModelParameters())
function reset_params(ip::ModelParameters)
    # the p is a global const
    # the ip is an incoming different instance of parameters 
    # copy the values from ip to p. 
    ip.popsize == 0 && error("no population size given")
    for x in propertynames(p)
        setfield!(p, x, getfield(ip, x))
    end
    # resize the human array to change population size
    resize!(humans, p.popsize)
end
export reset_params

function demographics() 
    # reset all agents to their default values
    for i = 1:length(humans)
        humans[i] = Human() 
    end

    # demographic data 
    # ontario census data: https://www11.statcan.gc.ca/census-recensement/1016/dp-pd/prof/details/Page.cfm?Lang=E&Geo1=PR&Code1=35&Geo1=&Code1=&Data=Count&SearchText=Ontario&Sear
    # convert to per 100,000 
    ages_per_ag = @SVector [0:4, 5:19, 20:64, 65:99]
    size_per_ag = pc.([697360, 2322285, 8177200, 2251655]) # from census data 
    sum(size_per_ag) != ON_POP && "error -- age groups don't add up to total population size"
    @info "pop size per capita (from census data):" size_per_ag, sum(size_per_ag)

    # data on babies/newborn, order is June to May multiplied twice since we are doing two years
    babies_per_month = pc.(@SVector [12291, 13052, 13127, 13618, 13331, 13249, 12624, 11864, 11376, 11242, 11095, 12830])
    preterms_per_month = pc.(@SVector [983, 1044, 1050, 1089, 1067, 1060, 1010, 949, 910, 899, 888, 1026])
    prop_pre_terms = preterms_per_month ./ babies_per_month # we do it this way to generate some stochasticity instead of fixing the number of preterms per month
    @info "newborns distribution: $babies_per_month, sum: $(sum(babies_per_month)), times two: $(sum(babies_per_month) *2)"

#    println("$babies_per_month")    
    # from the first age group, remove n agents to be added in as newborns later (we want to maintain 100000)
    # basically these are the agents we don't really care about -- have no affect in the model
    size_per_ag[1] = size_per_ag[1] - (sum(babies_per_month) * 2) # x2 because we are doing two years worth of babies 
    @info "adjusted pop sizes (removed newborns from first age group: $size_per_ag, sum: $(sum(size_per_ag))"
    
    # initialize the non-newborns first
    sz_non_newborns = sum(size_per_ag)
    human_ages = rand.(inverse_rle(ages_per_ag, size_per_ag)) # vector of ages for each non-newborn human
    shuffle!(human_ages)
    @info ("proportion of 0:4s that are 0 or 1 (but not marked as newborns): $(length(findall(x -> x <= 1, human_ages)))")
    idx = 0
    for i in 1:sz_non_newborns
        idx += 1 # we need to know when the loop ends so we know where to begin the next one for the babies
        #humans[idx] = Human()  # keep everything else as default
        x = humans[idx]
        x.idx = idx # we can use the i 
        x.age = human_ages[idx]  
        x.ageindays = 731 # 730 = two years
        x.monthborn = 0 # we don't really care about the monthborn, because we are not tracking these babies
        if x.age >=2 
            x.ageindays = 731 
        else # a baby is 0 or 1 year old but they are not newborns that are being trakced (perhaps immigration or movement)
            x.ageindays = rand(30:730) # but lets give them an age in days because we want them to be included in buckets
            # if its 1:730, the assumption is that there are some newborns that are not part of the "count" seyed gave me (immigration, etc)
            # if its 30:730, that means there are newborns from before the simulation started -- this doesnt even matter, since 
        end
    end
    
    # make sure no one has a monthassigned for being born
    newborns = findall(x -> x.monthborn >= 1, humans)
    @info "sanity check: humans with a monthborn property assigned (should be zero): $(length(newborns))"
 
    # probability of preterm 
    prob_gestation = Categorical([0.07, 0.17, 0.76]) # 1 = <29, 2 = 29-32 weeks, 3 = 33-36 weeks

    # next initialize the newborns (2 years worth) -- use two loops to do two years
    sz_newborns = sum(babies_per_month)  # this is the total newborns PER YEAR, so run the loop twice
    for y in [:y1, :y2]
        month_idx = inverse_rle(1:12, babies_per_month) # for each baby, assign them a birth month 
        @info "year: $y, distribution of newborns birth months $(countmap(month_idx)) "
        for i in 1:sz_newborns
            idx += 1 # we need to know when the loop ends so we know where to begin the next one for the babies
            humans[idx] = Human() 
            x = humans[idx]
            x.idx = idx # we can use the i 
            x.age = 0 
            x.ageindays = 1 #rand(1) # because its a new born
            if y == :y1  
                x.newborn = true 
                x.monthborn = month_idx[i] 
            else 
                x.newborn = false 
                x.monthborn = month_idx[i] + 12
            end
            x.activeaging = false  # new borns are not aging unless activated 
            if rand() <= prop_pre_terms[month_idx[i]] # month_idx[i] is the birth month 
                x.preterm = true
                x.gestation = rand(prob_gestation) 
            end
        end
    end

    # @info some statistics for debug purposes
    newborns = findall(x -> x.monthborn >= 1, humans)
    newborns_tracked = findall(x -> x.monthborn >= 1 && x.newborn == true, humans)
    preterm = findall(x -> x.monthborn >= 1 && x.preterm == true, humans)
    preterm_tracked = findall(x -> x.monthborn >= 1 && x.preterm == true && x.newborn == true, humans)
    gestperiods = [humans[x].gestation for x in preterm_tracked]
    @info ("total newborns: $(length(newborns)), tracked: $(length(newborns_tracked))")
    @info ("total preterm: $(length(preterm)), tracked: $(length(preterm_tracked))")
    @info ("gestation periods distribution $(countmap(gestperiods)), total: $(length(gestperiods))")
    return length(newborns_tracked)
end

function apply_comorbidities()
    # go through all newborns and apply comorbidities 
    # either 1 = CHD or 2 = CLD  

    # first lets do CHD, we need to sample 1% of the population 
    newborns = findall(x -> x.newborn == true, humans)
    _nl1 = Int(round(0.0123 * length(newborns)))    
    eligible_chd = sample(newborns, _nl1, replace=false)
    for id in eligible_chd 
        x = humans[id] 
        x.comorbidity = 1 
    end

    # lets do chronic lung disease as well 
    # it is possible to select the same newborn that had CHD 
    # but it will get replaced with CLD if that happens
    g1 = findall(x -> x.newborn == true && x.gestation == 1 && x.comorbidity == 0, humans)
    g2 = findall(x -> x.newborn == true && x.gestation == 2 && x.comorbidity == 0, humans)
    g3 = findall(x -> x.newborn == true && x.gestation == 3 && x.comorbidity == 0, humans)
    preterms = [g1, g2, g3]
    
    c1 = Int(round(0.281*length(g1)))
    c2 = Int(round(0.04*length(g2)))
    c3 = Int(round(0.024*length(g3)))
    cnts = [c1, c2, c3]
    eligible_cld = vcat(sample.(preterms, cnts, replace=false)...)
    for id in eligible_cld 
        x = humans[id] 
        x.comorbidity = 2
    end 
    return eligible_chd, eligible_cld
end

function apply_qalys() 
    # this function samples the QALYs for each agent 
    # this is to ensure that the same number of qalys are gained/loss 
    # over each scenario since the seed should be fixed  
    # constant (disutility) distributions for QALY calculations
    qaly_symptomatic = Beta(53.6, 281.4)
    qaly_nonrsv = Beta(19.2, 364.6)
    qaly_pediatricward = Beta(109.7, 157.9)
    qaly_icu = Beta(159.4, 106.2)
    qaly_wheezing = Beta(14.1, 338.4)

    newborns = findall(x -> x.newborn == true, humans)
    for id in newborns 
        x = humans[id] 
        x.qalys["symptomatic"] = 1 - rand(qaly_symptomatic)
        x.qalys["pedward"] = 1 - rand(qaly_pediatricward)
        x.qalys["icu"] = 1 - rand(qaly_icu)
        x.qalys["wheezing"] = 1 - rand(qaly_wheezing)
        x.qalys["nonrsv"] = 1 - rand(qaly_nonrsv)
    end
end

function household() 
    # function assigns each agent in the population to a household,
    # paramterized by Ontario census data 
    for x in humans 
        x.houseid = 0 # reset houseids for everyone 
    end
    adult_indices = shuffle!(findall(x -> x.age > 19 && x.age <= 65, humans))
    grand_indices = shuffle!(findall(x -> x.age > 65, humans))
    child_indices = shuffle!(findall(x -> x.age <= 19, humans))

    # get the total number of individuals in 1,1,3,4,5-person households
    household_sizes = [0, 1_815_000, 844545, 791435, 330560] # 1, 1, 3, 4, 5+ (fill in the 1 size household in the next step)
    single_adults = ON_POP - sum(household_sizes .* [1, 1, 3, 4, 5]) # calculate the 1-person household size by using the total population 
    household_sizes[1] = single_adults 
    prop_household_sizes = household_sizes ./ sum(household_sizes) 

    @info ("total adults: $(length(adult_indices)), grandparents: $(length(grand_indices)), children: $(length(child_indices))")
    @info ("population (size) in households $(household_sizes .* [1, 1, 3, 4, 5]), total sum: $(sum(household_sizes .* [1, 1, 3, 4, 5]))")
    
    # all data from Ontario census
    # go through children and determine whether they have two parents (a couple) or a lone parent 
    families_with_1_child = (658305 + 386415) # includes couples and lone parents
    families_with_1_child = (738710 + 186140) # includes couples and lone parents
    families_with_3_child = (311975 + 71310)  # includes couples and lone parents, 
   
    _totalfamilies = (644975 + 1708995) # from census data... however we add the above manually to ensure sum = 1 (data is off by 5 count)
    totalfamilies = families_with_1_child + families_with_1_child + families_with_3_child

    couples_with_1_child = 658305
    couples_with_1_child = 738710 
    couples_with_3_child = 311975

    prop_families_with_1_child = round(families_with_1_child / totalfamilies, digits=3)  # from census data 
    prop_families_with_1_child = round(families_with_1_child / totalfamilies, digits=3)  # from census data 
    prop_families_with_3_child = round(families_with_3_child / totalfamilies, digits=3)  # from census data 

    @info ("total families from census: $_totalfamilies")
    @info ("fam with 1 child: $families_with_1_child ($prop_families_with_1_child%), 1: $families_with_1_child ($prop_families_with_1_child%), 3: $families_with_3_child ($prop_families_with_3_child %) => total families: $totalfamilies")
  
    prop_couples_with_1_child = couples_with_1_child / families_with_1_child
    prop_couples_with_1_child = couples_with_1_child / families_with_1_child
    prop_couples_with_3_child = couples_with_3_child / families_with_3_child
    prop_couples_with_child = [prop_couples_with_1_child, prop_couples_with_1_child, prop_couples_with_3_child]
    
    childsize_probvec = Categorical([prop_families_with_1_child, prop_families_with_1_child, prop_families_with_3_child])
    
    houseid = 1 # initialize houseid 
    ctr_two_parent = 0 
    while length(child_indices) > 0   # have to really use a for loop here... splitting a vector up into partitions according to a distribution is known as the subset sum problem
        _childsize = rand(childsize_probvec)
        childsize = min(_childsize, length(child_indices)) # at the end there might not be enough children left 
        for i = 1:childsize
            child_idx = pop!(child_indices)
            humans[child_idx].houseid = houseid
        end
        parent1_idx = pop!(adult_indices)
        humans[parent1_idx].houseid = houseid
        if rand() < prop_couples_with_child[childsize]
            parent1_idx = pop!(adult_indices)
            humans[parent1_idx].houseid = houseid
            ctr_two_parent += 1
        end
        houseid += 1
    end
   
    # ## step 2: go through the adults that are supposed to be in single households, no children/grandparents
    # println("value of houseid at step 1: $houseid ") # making sure this isn't reset 
    # for adult_idx in adult_indices[1:one_person_households] # only select this many adults
    #     adult_idx = pop!(adult_indices)
    #     humans[adult_idx].houseid = houseid 
    #     houseid += 1 
    # end
    # println("total adults left without house assigned: $(length(adult_indices))")

    # step 3: distribute the rest according to the household comp distribution 
    #println("value of houseid at step 3: $houseid ") # making sure this isn't reset 
    dist_household_comp = Categorical(prop_household_sizes)
    remaining_adults = vcat(adult_indices, grand_indices)
    while length(remaining_adults) != 0 
        fsize = min(rand(dist_household_comp), length(remaining_adults)) # if there are only 3 adults left, we can't create a family size of 5...
        for _ = 1:fsize 
            p_idx = pop!(remaining_adults) 
            humans[p_idx].houseid = houseid
        end
        houseid +=1 
    end
end

function tag_siblings() 
    # go through every newborns household... determine if there is a newborn in that household, determine if there is another child age of 5 in that household
    newborns = findall(x -> x.newborn == true, humans)  
    println("newborns: $(length(newborns))")
    cnt_siblings = 0 
    for nb in newborns
        x = humans[nb]
        siblings = [y.age for y in humans if (y.houseid == x.houseid && y.idx != x.idx)] ## a sibling could be another newborn
        for s_age in siblings 
            if s_age <= 5 
                x.sibling = true
                cnt_siblings += 1
            end
        end
    end
    println("total newborns with siblings <5: $cnt_siblings ")
    return
end

function generate_household_statistics() 
    # calculate summary statistics 
    _chldnohouse = length(findall(x -> x.houseid == 0 && x.age <= 19, humans))
    _adltnohouse = length(findall(x -> x.houseid == 0 && x.age > 19, humans))
    _allchildidx = findall(x -> x.age <= 19, humans)
    _allchildhouses = [humans[i].houseid for i in _allchildidx]
    _famsize = countmap([humans[i].houseid for i in _allchildidx])
    _famsizevalues = countmap(values(_famsize))
    _famsizeprops = values(_famsizevalues) ./ sum(values(_famsizevalues))
    println("no house (kids): $_chldnohouse, (adults): $_adltnohouse ")
    println("unique households: $(length(unique(_allchildhouses)))")
    println("1, 1, 3 child distribution: $_famsizevalues, props (order of dict): $(round.(_famsizeprops, digits=1)), theoretical prop -- see above")

    all_houseids = countmap([x.houseid for x in humans])
    avg_house_size = mean(values(all_houseids))
    println("average housesize: $avg_house_size")
    # size_each_houseid = [length(findall(x -> x.houseid == i, humans)) for i in all_houseids]

    # generate debug file 
    _data = [(x.idx, x.age, x.houseid) for x in humans]
    writedlm("debug_households.csv", _data, ',')
end

function activate_newborns(mth) 
    # for a given month, this function activates newborns so they can go through the aging process
    # non-activated newborns are not aged 
    nb_idx = findall(x -> x.monthborn == mth, humans)
    for i in nb_idx 
        x = humans[i] 
        x.activeaging = true # turn on aging 
    end
    return length(nb_idx)
end

function within_3_month(x, mth)
    # a condition function to check if agents are eligible for infection
    # max number of infections is 2 and minimum of 3 months between infections
    c1 = x.rsvpositive < 2
    c4 = true
    if length(x.rsvmonth) > 0 
        c4 = (mth - x.rsvmonth[end]) >= 4 
    end
    return c1 && c4 
end

function increase_age() 
    # increases age by 30 days for those in the 0 - 730 day age bracket (i.e. 0 or 1 year of age)
    # have to be careful not to age the babies that are not born yet
    all_baby_idx = findall(x -> x.activeaging == true, humans)  # we don't have to filter by x.newborn or x.age because activeaging is false for everyone except those who we are tracking
    for idx in all_baby_idx
        x = humans[idx]
        x.ageindays = min(x.ageindays + 30, 731) 
    end
end

function get_monthly_agegroup_infcnt() 
    # function samples the yearly counts for both MA (medically attended) and N-MA (symptomatic, non medical) 
    # splits it over age groups AND over months 
    # (function should run once per year) 
    # returns the total sampled counts of MA and non-MA

    # get medically attended (MA) counts per year/season
    rates_per_agegroup = @SVector [0.051, 0.147, 0.111, 0.119, 0.166] # <19 d, 19–89 d, 90d to <6mo, 6mo to <1yr, 1 to <2 yr
    yearly_incidence = rand(Uniform(1001, 2439)) # yearly incidence sampled from a range -- this is for ALL age groups
    yearly_incidence_per_ag = yearly_incidence * rates_per_agegroup # split the yearly counts to the age groups we are interested in 
    sum_incidence_per_ag = sum(yearly_incidence_per_ag) # add up incidence for selected age groups only 
    @info ("yearly sampled MA: $yearly_incidence \nyearly MA per age group:\n $yearly_incidence_per_ag, \ntotal incidence (over 5 age grps): $(sum_incidence_per_ag)")
    
    # get non-medically (N-MA) (symptomatic) counts -- it's basically the MA increased by some percentage
    nma_range = rand(Uniform(0.15, 0.30)) # percentage of symptomatic (non-medical attended)
    nma_incidence = sum_incidence_per_ag * nma_range # because symptomatic = MA * (1 + x%)
    nma_rates_per_agegroup = @SVector [0, 0.011, 0.043, 0.185, 0.761] # split the non-medically attended counts over age groups - given by Seyed 
    yearly_nma_per_ag = nma_incidence * nma_rates_per_agegroup 
    
    @info ("total sampled N-MA: $nma_incidence \nyearly N-MA per age group:\n $yearly_nma_per_ag, \ntotal incidence (over 5 age grps): $(sum(yearly_nma_per_ag))")
    
    # as a check, see how many infants we have and compare against how many rsv infections we need to distribute
    _cr = length(findall(x -> x.ageindays <= 730, humans)) 
    @info "sanity checks \ntotal children at risk: $_cr, \ntotal incidence to distribute: $(sum(yearly_incidence_per_ag) + sum(yearly_nma_per_ag))"

    # the distribution of yearly cases to monthly cases (starting at January) 
    # this works for both MA and N-MA cases
    # this is split between the age groups:  #<19 d, 19–89 d, 90d to <6mo, 6mo to <1yr, 
    # order APRIL TO MARCH
    month_distr_ag1 = @SVector [0.0845, 0.0563, 0.0141, 0.0141, 0.0181, 0.0563, 0.0845, 0.1127, 0.1409, 0.1549, 0.1408, 0.1127]
    month_distr_ag2 = @SVector [0.0845, 0.0563, 0.0141, 0.0141, 0.0181, 0.0563, 0.0845, 0.1127, 0.1409, 0.1549, 0.1408, 0.1127]
    month_distr_ag3 = @SVector [0.0615, 0.0417, 0.0108, 0.0108, 0.0417, 0.0615, 0.0833, 0.1150, 0.1667, 0.1667, 0.115, 0.0833]
    month_distr_ag4 = @SVector [0.0714, 0.0357, 0.0357, 0.0357, 0.0357, 0.0714, 0.1071, 0.1419, 0.1430, 0.1419, 0.1071, 0.0714]
    month_distr_ag5 = @SVector [0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.1111, 0.1663, 0.1667, 0.1111, 0.0556]
    month_distr = [month_distr_ag1, month_distr_ag2, month_distr_ag3, month_distr_ag4, month_distr_ag5]

    # a matrix of age groups x monthly incidence (dim: 5 x 11)
    _monthly_inc = @. (yearly_incidence_per_ag * month_distr) 
    monthly_agegroup_incidence = Int.(ceil.(transpose(hcat(_monthly_inc...))))
    _monthly_nma = @. (yearly_nma_per_ag * month_distr)
    monthly_agegroup_non_ma = Int.(ceil.(transpose(hcat(_monthly_nma...))))
    @info "monthly incidence (MA and N-MA)" monthly_agegroup_incidence monthly_agegroup_non_ma

    @info "return values \nmonthly incidence (MA, N-MA) across age groups:" round.(sum(monthly_agegroup_incidence, dims=1)) round.(sum(monthly_agegroup_non_ma, dims=1))
    return monthly_agegroup_incidence, monthly_agegroup_non_ma
end 

function incidence() 
    # function simulations two seasons of RSV, uses helper functions above to sampl;e incidence numbers

    totalsick = 0  # ctr to keep track of total infections (including repeats) over 1 seasons

    # sample the monthly MA/N-MA infection counts...
    # since the function returns counts for 11 months only, do it twice, and concatenate the results so we have 14 months of data
    # benefit here is that both seasons have their own stochasticity
    # after the concatenation, the matrix is 5x14 (5 age groups x 14 months)
    yr1_monthly_agegroup_incidence, yr1_monthly_agegroup_non_ma = get_monthly_agegroup_infcnt()
    yr2_monthly_agegroup_incidence, yr2_monthly_agegroup_non_ma = get_monthly_agegroup_infcnt()
    monthly_agegroup_incidence = hcat(yr1_monthly_agegroup_incidence, yr2_monthly_agegroup_incidence)
    monthly_agegroup_non_ma = hcat(yr1_monthly_agegroup_non_ma, yr2_monthly_agegroup_non_ma)
    @info "\ntotal infections to distribute over y1" sum(yr1_monthly_agegroup_incidence)+sum(yr1_monthly_agegroup_non_ma) 
    @info "total infections to distribute over y2" sum(yr2_monthly_agegroup_incidence)+sum(yr2_monthly_agegroup_non_ma) 

    @info "total infections to distribute over ag1/y1" sum(yr1_monthly_agegroup_incidence[1, :])+sum(yr1_monthly_agegroup_non_ma[1, :]) 
    @info "total infections to distribute over ag1/y2" sum(yr2_monthly_agegroup_incidence[1, :])+sum(yr2_monthly_agegroup_non_ma[1, :]) 
    @info "total infections to distribute over 24 months" sum(monthly_agegroup_incidence)+sum(monthly_agegroup_non_ma) 
 
    # loop through months: Order is April => 1, May => 2, etc 
    # MUST VERIFY that the vectors used to inject babies and sample monthly newborns are in the right order!
    for mth in 1:24
        _acnb = activate_newborns(mth) 
        @info "simulating month: $mth, activating: $_acnb newborns"

        # eligble children (0 - 730 days of age) split into 5 age groups (because the counts are sampled over these five age groups)
        # ag1_idx includes all the newborns since their ageindays = 1, but need to make sure not to get babies that are not born yet
        # ag1_idx should not include 'non-newborns' -- everyone should be >30 ageindays
        ag1_idx = shuffle!(findall(x -> x.ageindays > 0  && x.ageindays <= 30 && within_3_month(x, mth) && x.monthborn == mth, humans))
        ag2_idx = shuffle!(findall(x -> x.ageindays > 30  && x.ageindays <= 90 && within_3_month(x, mth), humans))
        ag3_idx = shuffle!(findall(x -> x.ageindays > 90  && x.ageindays <= 180 && within_3_month(x, mth), humans))
        ag4_idx = shuffle!(findall(x -> x.ageindays > 180 && x.ageindays <= 365 && within_3_month(x, mth), humans))
        ag5_idx = shuffle!(findall(x -> x.ageindays > 365 && x.ageindays <= 730 && within_3_month(x, mth), humans))
        ag_idx = [ag1_idx, ag2_idx, ag3_idx, ag4_idx, ag5_idx]

        # get the month specific infection counts for all age groups
        ma_to_distr = monthly_agegroup_incidence[:, mth]
        nma_to_distr = monthly_agegroup_non_ma[:, mth]

        @info "incidence cnts to be sampled: \nMA: $(ma_to_distr)  \nN-MA: $nma_to_distr \nnum of babies in our system: $(length.(ag_idx))"
        
        # this statement samples from each agegroup the number of MA/N-MA individuals, this then creates an array of array.
        # i.e. sampled_idx_to_be_sick[1] is the indices of people from ag1 that will be sick 
        # i.e. sampled_idx_to_be_sick[2] is the indices of people from ag2 that will be sick
        # we combine the MA and N-MA together but then split it later according to the counts
        # understand the sample function by running this command: sample.([[1, 1, 3, 4, 5], [11, 11, 13, 14, 15]], [1, 3], replace=false)
        sampled_idx_to_be_sick = sample.(ag_idx, ma_to_distr .+ nma_to_distr, replace=false)

        # go through each age group, and within each )
        for (idx, ag_sick_ids) in enumerate(sampled_idx_to_be_sick)
            # ag_sick_ids is the array of indices of agents to make sick (but they are combined between MA and N-MA)
            # idx is the age group from 1 to 5 
            # generate a pre-list of whether individual is MA = 1 or N-MA = 2 based on the distribution sampled from above
            sick_type_lst = inverse_rle([1, 2], [ma_to_distr[idx], nma_to_distr[idx]])
            for i in ag_sick_ids 
                h = humans[i]
                h.rsvpositive = h.rsvpositive + 1
                push!(h.rsvmonth, mth)
                push!(h.rsvage, h.ageindays)
                push!(h.rsvtype, pop!(sick_type_lst))
                totalsick += 1
            end
        end       
        # increase everyones age
        increase_age()
    end
    # error check, all newborns should be activated
    @info "newborns not activated (should be zero): $(length(findall(x -> x.newborn == true && x.activeaging == false, humans)))"
    @info "actual total sick inserted into the system: $(totalsick)"
    return totalsick
end    

function maternal_vaccine() 
    # all newborns get maternal vaccine with some coverage
    newborns = findall(x -> x.newborn == true && rand(vax_rng) < p.mat_coverage, humans) # for each newborn baby, determine their vaccine efficacy for each month of the simulation. 
    total_cost = 0 # we don't need to track the costs, Seyed is handling on his side.
    
    # old efficacies
    # define efficacy values over 12 months from time of administration
    # eff_outp = [72, 63, 49, 34, 23, 16, 13, 11, 10, 0, 0, 0] ./ 100
    # eff_hosp = [93, 82, 69, 56, 43, 32, 24, 19, 15, 0, 0, 0] ./ 100
    # eff_icu = [93, 82, 69, 56, 43, 32, 24, 19, 15, 0, 0, 0] ./ 100

    # new efficacies for the revisions 
    if p.vaccine_eff_scenaro == "sigmoidal" 
        # sigmoidal efficacy
        eff_outp = [0.748, 0.667, 0.520, 0.329, 0.169, 0.076, 0.032, 0.015, 0.008, 0.005, 0.000, 0.000]
        eff_hosp = [0.906, 0.835, 0.724, 0.577, 0.413, 0.265, 0.153, 0.080, 0.036, 0.012, 0.000, 0.000]
        eff_icu = [0.906, 0.835, 0.724, 0.577, 0.413, 0.265, 0.153, 0.080, 0.036, 0.012, 0.000, 0.000]
    elseif p.vaccine_eff_scenaro == "fixed" 
        # fixed efficacy
        eff_outp = [0.571, 0.571, 0.571, 0.571, 0.489, 0.408, 0.326, 0.245, 0.163, 0.082, 0.000, 0.000]
        eff_hosp = [0.818, 0.818, 0.818, 0.818, 0.701, 0.584, 0.467, 0.351, 0.234, 0.117, 0.000, 0.000]
        eff_icu = [0.818, 0.818, 0.818, 0.818, 0.701, 0.584, 0.467, 0.351, 0.234, 0.117, 0.000, 0.000]
    else 
        error("invalid vaccine efficacy setting")
    end

    for (i, h) in enumerate(newborns)          
        x = humans[h]
        mb = x.monthborn
        # a quick error check
        mb > 12 && error("error in newborns/vaccine, someone is being tracked over 12 months of age") 
        fm = mb 
        em = fm + 11
        x.eff_outpatient[fm:em] .= eff_outp
        x.eff_hosp[fm:em] .= eff_hosp
        x.eff_icu[fm:em] .= eff_icu
        x.vac_mat = true # set flag to true to indicate newborn is maternally vaccinated
    end
    @info "number of newborns for MI and cost" length(newborns), total_cost
    return total_cost, length(newborns)
end


function lama_eligible(sc) 
    # LAMA vaccine is incremental. 
    # so L2 should include all the selected people from L1 
    # and L3 should add on top of L2, and so on
    sc ∉ (:s1, :s2, :s3, :s4) && error("invalid LAMA strategy")
   
    # find all comorbid infants
    nb_co = findall(x -> x.newborn == true && x.comorbidity > 0, humans)
    #nb_co = []
    # find infants based on gestation/monthborn
    nb_s1 = union(nb_co, findall(x -> x.newborn == true && x.gestation in (1, 2), humans))
    nb_s2 = union(nb_co, findall(x -> x.newborn == true && x.preterm == true, humans))
    nb_s3 = union(nb_co, findall(x -> x.newborn == true && (x.preterm == true || x.monthborn in 7:12), humans))
    nb_s4 = union(nb_co, findall(x -> x.newborn == true, humans))
    

    # take the set differences to get the incremental infants added after each step
    elig_s1 = nb_s1 #setdiff(nb_s1, Set([])) # doesn't do anything -- setdiff with empty set is not type safe 
    elig_s2 = setdiff(nb_s2, nb_s1)
    elig_s3 = setdiff(nb_s3, nb_s2, nb_s1) # essentiall all infants in 7:12, but not preterms (since thats s2)
    elig_s4 = setdiff(nb_s4, nb_s3, nb_s2, nb_s1) # essentiall all infants but not those in previous categories

    # # convert to vectors for sampling
    _vecs = collect.([elig_s1, elig_s2, elig_s3, elig_s4])
    # @info ("length vecs: $(length.(_vecs))")
    
    coverage = p.l1_l4_coverage
    covlengths = Int.(round.(coverage .* length.(_vecs))) 
    eligible_per_strategy = sample.(vax_rng, _vecs, covlengths, replace=false)

    ##idxs = [humans[x].idx for x in eligible_per_strategy]
    ##isitin = 98108 ∈ idxs 
    #println("is it in: $isitin")
        
    total_eligible = @match string(sc) begin 
        "s1" => [eligible_per_strategy[1]...]
        "s2" => [eligible_per_strategy[1]..., eligible_per_strategy[2]...]
        "s3" => [eligible_per_strategy[1]..., eligible_per_strategy[2]..., eligible_per_strategy[3]...]
        "s4" => [eligible_per_strategy[1]..., eligible_per_strategy[2]..., eligible_per_strategy[3]..., eligible_per_strategy[4]...]
        _ =>  error("wrong strategy for lama vaccination")
    end
    return total_eligible
end

function lama_vaccine(strategy) 
    # REVISION SCENARIOS: 
    # L1 Infants ≤32 wGA, and infants with CLD or CHD 
    # L2 Infants ≤36 wGA, and infants with CLD or CHD 
    # L3 Entire birth cohort
    # MI Pregnant women 
    # LMI Infants ≤32 wGA, and infants with CLD or CHD  and pregnant women

    # OLD SCENARIOS 
    # (i) vaccination of preterm infants under 32 wGA with 90% coverage (S1); 
    # (ii) vaccination of all preterm infants with 90% coverage  !!! in addition to (i); 
    # (iii) vaccination of infants born in month 7:12 with 90% coverage  !!! in addition to (ii); 
    # (iv)  vaccination of all infants !!! in addition to (iii); 

    # get the IDs of newborns based on strategy 
    # `lama_eligible` builds the list of newborns incrementally
    newborns = lama_eligible(strategy)
    total_costs = 0 # we don't need to track the costs, Seyed is handling on his side.
    # old efficacies
    # eff_outp = [100, 96, 87, 70, 50, 33, 23, 19, 17, 0, 0, 0] ./ 100
    # eff_hosp = [100, 95, 84, 67, 47, 31, 22, 18, 16, 0, 0, 0] ./ 100
    # eff_icu = [100, 97, 91, 80, 64, 47, 33, 24, 20, 0, 0, 0] ./ 100

    # new efficacies for the revisions 
    if p.vaccine_eff_scenaro == "sigmoidal" 
        # sigmoidal efficacies
        eff_outp = [0.867, 0.858, 0.839, 0.805, 0.743, 0.643, 0.501, 0.337, 0.185, 0.070, 0.000, 0.000]
        eff_hosp = [0.857, 0.832, 0.792, 0.731, 0.646, 0.537, 0.410, 0.282, 0.167, 0.077, 0.000, 0.000]
        eff_icu = [0.920, 0.901, 0.870, 0.822, 0.750, 0.648, 0.518, 0.372, 0.227, 0.101, 0.000, 0.000]
    elseif p.vaccine_eff_scenaro == "fixed" 
        # fixed efficacies
        eff_outp = [0.795, 0.795, 0.795, 0.795, 0.795, 0.795, 0.636, 0.477, 0.318, 0.159, 0.000, 0.000]
        eff_hosp = [0.773, 0.773, 0.773, 0.773, 0.773, 0.773, 0.618, 0.464, 0.309, 0.155, 0.000, 0.000]
        eff_icu = [0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.688, 0.516, 0.344, 0.172, 0.000, 0.000]
    else 
        error("invalid vaccine efficacy setting")
    end

    # assign the efficacies to infants based on when they are born and when the vaccine is administered
    for (i, h) in enumerate(newborns)    
        x = humans[h]
        
        mb = x.monthborn
        # a quick error check
        mb > 12 && error("error in newborns/vaccine, someone is being tracked over 12 months of age") 
        fm = max(7, mb)
        em = fm + 11
        
        x.eff_outpatient[fm:em] .= eff_outp
        x.eff_hosp[fm:em] .= eff_hosp
        x.eff_icu[fm:em] .= eff_icu
        x.vac_lama = true # set flag to true to indicate newborn is vaccinated by lama
    end
    @info "number of newborns for LAMA and cost" length(newborns), total_costs
    return total_costs, length(newborns)
end

function _generate_hosp_lhc(lows, highs)
    # generate LHC given a range lows and high vectors
    @assert length(lows) == 12 
    @assert length(highs) == 12 
    res = zeros(Float64, p.numofsims, 12)
    lhc = randomLHC(Random.GLOBAL_RNG, p.numofsims, 12) ./ p.numofsims # 4 dimensions for preterm1,2, 3 (fullterm is same as 3) + mortality
    for mth in 1:12
        res[:, mth] .= @. lows[mth] + lhc[:, mth]*(highs[mth] - lows[mth])
    end
    return res
end

function generate_hospitalization_rates()   
    # function generates the hosptialization (in-patient) probabilities 
    # uses Latin hypercube sampling for each preterm and fullterm  
    # the return value is a matrix of num_of_simulations x 12 (mo∏nths)
    # so for each simulation, we pick the corresponding row.
    # if `means = true` then all simulation indices are just populated with the mean

    # range lows and highs for preterm1, preterm2, preterm3, and fullterm
    p1l = [5.28, 9.33, 6.66, 4.68, 4.05, 2.46, 2.10, 2.37, 1.77, 1.95, 1.77, 1.47] 
    p1h = [6.84, 13.11, 7.26, 5.22, 4.50, 3.24, 2.88, 2.85, 2.16, 2.52, 1.89, 1.77]
    p2l = [3.52, 6.22, 4.44, 3.12, 2.70, 1.64, 1.40, 1.58, 1.18, 1.30, 1.18, 0.98]
    p2h = [4.56, 8.74, 4.84, 3.48, 3.00, 2.16, 1.92, 1.90, 1.44, 1.68, 1.26, 1.18]
    p3l = [1.76, 3.11, 2.22, 1.56, 1.35, 0.82, 0.70, 0.79, 0.59, 0.65, 0.59, 0.49]
    p3h = [2.28, 4.37, 2.42, 1.74, 1.50, 1.08, 0.96, 0.95, 0.72, 0.84, 0.63, 0.59]
    fl = [1.76, 3.11, 2.22, 1.56, 1.35, 0.82, 0.70, 0.79, 0.59, 0.65, 0.59, 0.49]
    fh = [2.28, 4.37, 2.42, 1.74, 1.50, 1.08, 0.96, 0.95, 0.72, 0.84, 0.63, 0.59]

    preterm1 = _generate_hosp_lhc(p1l, p1h) ./ 100
    preterm2 = _generate_hosp_lhc(p2l, p2h) ./ 100 
    preterm3 = _generate_hosp_lhc(p3l, p3h) ./ 100 
    fullterm = _generate_hosp_lhc(fl, fh) ./ 100

    return preterm1, preterm2, preterm3, fullterm
end

function generate_mortality_rates()   
    p1 = (low=0.0036, high=0.033)
    p2 = (low=0.0036, high=0.033)
    p3 = (low=0.0002,  high=0.0182)
    ft = (low=0.0002, high=0.01)
    hd = (low=0.034, high=0.053) # heart disease
    ld = (low=0.035, high=0.051) # lung disease
  
    res = zeros(Float64, p.numofsims, 6) # 6 columns (1,2,3 for preterm, 4 for full term, 5/6 for chd/cld)
    lhc = randomLHC(Random.GLOBAL_RNG, p.numofsims, 6) ./ p.numofsims

    # elements 1:3 are for preterm, element 4 is fullterm, element 5 and 6 are CHD/CLD
    res[:, 1] = @. p1.low + lhc[:, 1]*(p1.high - p1.low)
    res[:, 2] = @. p2.low + lhc[:, 2]*(p2.high - p2.low)
    res[:, 3] = @. p3.low + lhc[:, 3]*(p3.high - p3.low)
    res[:, 4] = @. ft.low + lhc[:, 4]*(ft.high - ft.low)  
    res[:, 5] = @. hd.low + lhc[:, 5]*(hd.high - hd.low)  
    res[:, 6] = @. ld.low + lhc[:, 6]*(ld.high - ld.low)  
    return res
end

function assign_inpatient_death_probs_to_modelparams() 
    # function uses LHS to sample inpatient/mortality matrices 
    # and assigns it to the ModelParameters for use later on 
    # function is only run once before simulations start -- 
   
    p1, p2, p3, ft = generate_hospitalization_rates() 
    mr = generate_mortality_rates()
    
    # need to assign to workers 
    sendto(workers(), p1=p1, p2=p2, p3=p3, ft=ft, mr=mr)
    @everywhere p.lhs_inpatient_preterm1 = $p1
    @everywhere p.lhs_inpatient_preterm2 = $p2
    @everywhere p.lhs_inpatient_preterm3 = $p3
    @everywhere p.lhs_inpatient_fullterm = $ft
    @everywhere p.lhs_mortality = $mr

    # p.lhs_inpatient_preterm1 = p1
    # p.lhs_inpatient_preterm2 = p2 
    # p.lhs_inpatient_preterm3 = p3 
    # p.lhs_inpatient_fullterm = ft
    # p.lhs_mortality = mr
    return p1, p2, p3, ft, mr 
end

function set_inpatient_and_death_probs(sim_number) 
    # assign the simulation specific hospitalization, mortality rates to the constant variables
    # this requires the parameters in ModelParameters set properly (use sample_inpatient_and_death_probs)
    # if sim_number == 0, just apply the means ... this is good for testing
    if sim_number == 0 
        # mean values 
        p1m = [6.06, 11.22, 6.96, 4.95, 4.275, 2.85, 2.49, 2.61, 1.965, 2.235, 1.83, 1.62] ./ 100 
        p2m = [4.04, 7.48, 4.64, 3.3, 2.85, 1.9, 1.66, 1.74, 1.31, 1.49, 1.22, 1.08] ./ 100 
        p3m = [2.02, 3.74, 2.32, 1.65, 1.425, 0.95, 0.83, 0.87, 0.655, 0.745, 0.61, 0.54] ./ 100 
        fm = [2.02, 3.74, 2.32, 1.65, 1.425, 0.95, 0.83, 0.87, 0.655, 0.745, 0.61, 0.54] ./ 100
        prob_inpatient_preterm_1 .= p1m
        prob_inpatient_preterm_2 .= p2m
        prob_inpatient_preterm_3 .= p3m
        prob_inpatient_fullterm .= fm
        prob_death_rate .= p.lhs_mortality[1, :] # we don't have average mortality rates? just pick the first LHS one
    else 
        prob_inpatient_preterm_1 .= p.lhs_inpatient_preterm1[sim_number, :]
        prob_inpatient_preterm_2 .= p.lhs_inpatient_preterm2[sim_number, :]
        prob_inpatient_preterm_3 .= p.lhs_inpatient_preterm3[sim_number, :]
        prob_inpatient_fullterm .=  p.lhs_inpatient_fullterm[sim_number, :]
        prob_death_rate .= p.lhs_mortality[sim_number, :]
    end
end

function outcome_flow(x, sim_id) 
    # this function determines the outcomes of rsv infants
    #  -- argument x: should be an initialized agent with *atleast ONE INFECTION*

    # for all probability checks (for outcomes), use an agent-specific RNG 
    # this is because we want the outcomes to be same for each scenario (but inpatient/outpatient probabilities change due to vaccine) 
    outcomes_RNG = MersenneTwister(x.idx * sim_id)

    # Random.seed!(x.idx)

    # create empty array to store the flow of outcomes
    flow = String[]
   
    # check if second episode happens within 12 months (we already know that the first episode is happening for person x in argument)
    infcnt_max = 1 # we know there is minimum one episode 
    if x.rsvpositive == 2 # if two symptomatic episodes, check whether the second episode happens within 12 months 
        diff = x.rsvmonth[2] - x.monthborn[1]
        infcnt_max = diff <= 11 ? 2 : 1
    end

    # A dictionary to hold the sampled number of days for each outcome 
    # for some parameters, the values are binary to indiciate event happened or not
    sampled_days = Dict(
        "prior_to_infection" => 0., 
        "after_infection" => 0., 
        "symptomatic" => 0., 
        "pediatricward" => 0., 
        "icu" => 0., 
        "wheezing" => 0., 
        "emergencydept" => 0.,  # binary 0/1 these two only really used for cost purposes... not for QALYs
        "office_consultation" => 0, # binary 0/1
        "death" => 0 # binary 0/1
    )
    
    for ic in 1:infcnt_max    
        if sampled_days["death"] > 0  # if the person was sampled to be dead, skip the loop. logic is that sampled_death = 0 for ic = 1 (initial infection)
            continue 
        end
        push!(flow, "INF$ic")
        push!(flow, "symptomatic")

        # pre-sample the coin tosses to check against probabilities of each outcome 
        # we do it here to maintain sequence of random numbers. 
        rn_outpatient = rand(outcomes_RNG)
        rn_inpatient = rand(outcomes_RNG)
        rn_icu = rand(outcomes_RNG)
        rn_death = rand(outcomes_RNG)
        rn_wheezing = rand(outcomes_RNG)
        rn_emergency = rand(outcomes_RNG)
        
        # store rsvtype and rsvmonth used to adjust outcome flow probabilities 
        rt = x.rsvtype[ic] 
        rm = x.rsvmonth[ic]
        !(rt in (1, 2)) && error("RSV type incorrect, required 1 or 2... given: $rt") # quick sanity check

        # calculate probabilies of the outcomes 
        prob_emergencydept = rand(outcomes_RNG, Uniform(0.4, 0.5))
        
        # probability of inpatient 
        # adjusted by vaccine and adjusted by efficacy.
        # prob_inpatient_preterm, prob_inpatient_fullterm are defined globally

        # range lows and highs for preterm1, preterm2, preterm3, and fullterm
        ft = [(1.76, 2.28), (3.11, 4.37), (2.22, 2.42), (1.56, 1.74), (1.35, 1.5), (0.82, 1.08), (0.7, 0.96), (0.79, 0.95), (0.59, 0.72), (0.65, 0.84), (0.59, 0.63), (0.49, 0.59)]
        p3 = [(1.76, 2.28), (3.11, 4.37), (2.22, 2.42), (1.56, 1.74), (1.35, 1.5), (0.82, 1.08), (0.7, 0.96), (0.79, 0.95), (0.59, 0.72), (0.65, 0.84), (0.59, 0.63), (0.49, 0.59)]
        p2 = [(3.52, 4.56), (6.22, 8.74), (4.44, 4.84), (3.12, 3.48), (2.7, 3.0), (1.64, 2.16), (1.4, 1.92), (1.58, 1.9), (1.18, 1.44), (1.3, 1.68), (1.18, 1.26), (0.98, 1.18)]
        p1 = [(5.28, 6.84), (9.33, 13.11), (6.66, 7.26), (4.68, 5.22), (4.05, 4.5), (2.46, 3.24), (2.1, 2.88), (2.37, 2.85), (1.77, 2.16), (1.95, 2.52), (1.77, 1.89), (1.47, 1.77)]
        _prob_inpatient_preterm = [p1, p2, p3]
        _prob_inpatient_fullterm = ft
        distr = x.preterm ? _prob_inpatient_preterm[x.gestation] : _prob_inpatient_fullterm
        diff = (rm - x.monthborn) + 1 
        _pb = rand(outcomes_RNG, distr[diff]) / 100 # get the month specific probability of inpatient
        # adjust inpatient probability by comorbidity
        if x.comorbidity > 0  
            OR = x.comorbidity == 1 ? 1.9 : 2.2 # Odds Ratio
            _pb = (OR * _pb) / (1 + (OR*_pb) - _pb) # https://stats.stackexchange.com/questions/324410/converting-odds-ratio-to-percentage-increase-reduction
        end
        _pb = _pb * (1 - x.eff_hosp[rm])    
        prob_inpatient = _pb
  
        # probability of ICU is sampled from Uniform distributions depending on the gestation period 
        if x.gestation in (1, 2) 
            prob_icu = rand(outcomes_RNG, Uniform(41.3, 62.1)) ./ 100 
        elseif x.gestation == 3 
            prob_icu = rand(outcomes_RNG, Uniform(13.1, 53.6)) ./ 100 
        else 
            prob_icu = rand(outcomes_RNG, Uniform(5.4, 30.0)) ./ 100 
        end
        prob_icu = prob_icu * (1 - x.eff_icu[rm]) # adjust icu by vaccine efficacy
        
        # probability of wheezing
        prob_wheezing = 0.31 

        # probability of mortality
        p1 = rand(outcomes_RNG, Uniform(0.0036, 0.033))
        p2 = rand(outcomes_RNG, Uniform(0.0036, 0.033))
        p3 = rand(outcomes_RNG, Uniform(0.0002, 0.0182))
        ft = rand(outcomes_RNG, Uniform(0.0002, 0.01))
        hd = rand(outcomes_RNG, Uniform(0.034, 0.053)) # heart disease
        ld = rand(outcomes_RNG, Uniform(0.035, 0.051)) # lung disease
        _prob_death_rate = [p1, p2, p3, ft, hd, ld]
        prob_death = x.preterm ? _prob_death_rate[x.gestation] : _prob_death_rate[4] # the 4th element of `prob_death_rate` corresponds to full term mortality rate
        if x.comorbidity > 0
            _aidx = x.comorbidity + 4 # comorbidity is either 1 or 2 (chd or cld) which corresponds to elements 5 and 6 of prob_death_rate
            prob_death = _prob_death_rate[_aidx] 
        end    

        # for every single rsv episode, sample the number of symptomatic days 
        # reduce the number of symptomatic days by 60% if person is NON-MA (or becomes NON MA due to vaccine)
        days_symptomatic = rand(outcomes_RNG, Uniform(5, 8))
        sampled_days["symptomatic"] = days_symptomatic
      
        # return sampled_days, flow
        # first efficacy endpoint, against outpatient/inpatient (i.e. MA)
        # i.e., essentially turns a MA infant to a N-MA infant (the loop will not run) 
        outpatient_ct = rn_outpatient < x.eff_outpatient[rm] 
        
        # # 97792, 97847, 97859, 97860, 97906, 97910, 97919, 97931, 97947, 97948, 97953, 97989, 97992, 98017, 98063, 98109, 98111, 98115, 98143, 98151, 98169, 98175, 98206, 98219, 98235, 98246, 98264, 98313, 98316, 98321, 98340, 98351, 98361, 98362, 98394, 98396, 98405, 98410, 98433, 98434, 98442, 98447, 98448, 98456, 98467, 98470, 98486, 98507, 98513, 98533, 98571, 98579, 98586, 98626, 98656, 98672, 98677, 98699, 98721, 98733, 98746, 98751, 98765, 98790, 98802, 98812, 98824, 98836, 98844, 98848, 98876, 98879]
        # testsubject = 97947
        # if x.idx == testsubject

        #     println("monthborn $(x.monthborn) rsvmonth: $rm type: $(rt), vax? $(x.vac_lama)")
        #     println("will turn non-ma: $outpatient_ct, eff: $(x.eff_outpatient[rm])")
        #     println("prob of inpatient before vaccine: $_tmp, after vaccine: $(prob_inpatient)")
        #     println("will inpatient: $(rn_inpatient < prob_inpatient) coin flip: $rn_inpatient, prob_inpatient: $prob_inpatient, eff: $(x.eff_icu[rm]) ")
        #     #println("will die: $(rn_recovery < prob_recovery)")
        # end

        # for non-ma episodes (either by sampled or by vaccine), ignore the remaining code
        if rt == 2 || outpatient_ct
            push!(flow, "nonma")
            continue
        end
        
        # at this point the infant is MA and will either be an outpatient or inpatient 
        # outcome flows for inpatient/outpatient, icu/ward, wheezing, and recovery/death
        if rn_inpatient < prob_inpatient 
            #x.rsvhospitalized = true
            push!(flow, "inpatient")
            if rn_icu < prob_icu 
                push!(flow, "icu")
                sampled_days["icu"] += x.gestation in (1, 2) ? rand(Gamma(20.22, 0.47)) : rand(Gamma(12.38, 0.42))
            else # pediatric ward
                push!(flow, "pediatric ward")
                sampled_days["pediatricward"] += x.gestation in (1, 2) ? rand(Gamma(12.71, 0.48)) : rand(Gamma(6.08, 0.64)) 
            end
            if rn_death < prob_death  # after ICU or PedWard, check if recovered or death 
                push!(flow, "death")
                sampled_days["death"] += ic # by assigning death as ic (1 or 2) it gives us an indication of when death happened
            else 
                push!(flow, "recovery")
                if rn_wheezing < prob_wheezing # wheezing episode
                    push!(flow, "wheezing")
                    sampled_days["wheezing"] +=  rand(Uniform(5.2, 9.8))
                end
            end
        else # is either office ot ED
            push!(flow, "outpatient")
            if rn_emergency < prob_emergencydept
                push!(flow, "ED")
                sampled_days["emergencydept"] += 1
            else 
                push!(flow, "Office")
                sampled_days["office_consultation"] += 1
            end
        end
    end

    # add the remaining days (prior, after infection) -- might use this in calculations of QALYs
    sampled_days["prior_to_infection"] += (x.rsvmonth[1] - x.monthborn) * 30 
    sampled_days["after_infection"] += 365 - sum(values(sampled_days))

    return sampled_days, flow
end

function calculate_qaly(x, sampled_days) 
    # calculate the QALYs due to type of RSV episode
    q_symp = sampled_days["symptomatic"] / 365 * x.qalys["symptomatic"]
    q_pediatricward = sampled_days["pediatricward"] / 365 * x.qalys["pedward"]
    q_icu = sampled_days["icu"] / 365 * x.qalys["icu"]
    q_wheezing = sampled_days["wheezing"] / 365 * x.qalys["wheezing"]
    
    # if a infant is dead, they have a loss of qaly from the time they die 
    daysdead = 0
    if sampled_days["death"] > 0 # since death can be 1 or 2
        infcnt = sampled_days["death"] # will be either 1 or 2 
        daysdead = (12 - mod(x.rsvmonth[infcnt] - x.monthborn, 12)) * 30 
    end
    
    # calculate the non-RSV QALY 
    non_rsv_days = (365 - sampled_days["symptomatic"] - sampled_days["pediatricward"] - sampled_days["icu"]  - sampled_days["wheezing"] - daysdead) / 365
    non_rsv_qaly = non_rsv_days * x.qalys["nonrsv"]
    
    # calculate the total QALY for the infant (non rsv + rsv)
    totalqalys = non_rsv_qaly + q_symp + q_pediatricward + q_icu + q_wheezing

    q_loss_due_to_death = sampled_days["death"] > 0 ? 45.3 : 0 
    return totalqalys, q_loss_due_to_death
end

function calculate_costs(x, sampled_days) 
   # A dict to hold costs as they are being calculated
   costs = Dict(
    "cost_in_icu" => 0., 
    "cost_in_ward" => 0., 
    "cost_hosp_followup" => 0., 
    "cost_wheezing" => 0., 
    "cost_outpatient" => 0.
    )
    costs["cost_in_icu"] = 3638 * sampled_days["icu"] # 10, 2 symptomatics
    costs["cost_in_ward"] = 1491 * sampled_days["pediatricward"]
    costs["cost_hosp_followup"] = begin 
        cost_followup = 0 
        if (sampled_days["icu"] + sampled_days["pediatricward"]) > 0 
            diff1 = x.rsvmonth[1] - x.monthborn
            if diff1 == 0 
                cost_followup += 1791 
            elseif diff1 in (1, 2)  
                cost_followup += 1261
            elseif diff1 in (3, 4, 5) 
                cost_followup += 423 
            else 
                cost_followup += 374
            end
            
            if x.rsvpositive == 2 
                diff2 = x.rsvmonth[2] - x.monthborn
                if diff2 == 0 
                    cost_followup += 1791 
                elseif diff2 in (1, 2)  
                    cost_followup += 1261
                elseif diff2 in (3, 4, 5) 
                    cost_followup += 423 
                else 
                    cost_followup += 374
                end
            end
        end
        cost_followup
    end
    costs["cost_wheezing"] = begin 
        cw = 0 
        if sampled_days["wheezing"] > 0
            cw = sampled_days["wheezing"] > 10 ? (229 * 2) : 229 # a little trick to see if there were two wheezing episodes (since max of one episode is 9.8)
        end
        cw
    end
    costs["cost_outpatient"] = (sampled_days["emergencydept"] * 342) + (sampled_days["office_consultation"] * 229) 
    cost_due_to_death = sampled_days["death"] > 0 ? 2292572 : 0
    return sum(values(costs)), cost_due_to_death
end

function outcome_analysis(sim_id)
    # this function performs outcome analysis on all infants with RSV episodes 
   
    # find all newborns that had atleast one symptomatic episode in their first year of life
    newborns = shuffle!(findall(x -> x.newborn == true && x.rsvpositive > 0 && (x.rsvmonth[1] - x.monthborn) <= 11, humans))

    # sanity check: there are some infants that will have an infection outside their first year of life... lets print out the difference
    _nbpos = findall(x -> x.newborn == true && x.rsvpositive > 0, humans) 
    @info "\ntotal rsv positive newborns: $(length(_nbpos)) \nthose with first infection in first year:$(length(newborns))"

    # initialize empty vectors to save all outcome data
    # data = [] # -- use if saving all the individual level outcome data -- not really neccessary  
    totalqalys = 0
    qalyslost_death = 0 
    totalcosts = 0
    totalcosts_death = 0
    total_inpatients = 0 
    total_outpatients = 0
    total_non_ma = 0 
    total_hosp_days = 0 

    for (i, h) in enumerate(newborns) 
        x = humans[h]   
        sampled_days, flow = outcome_flow(x, sim_id)  # simulate outcomes for the RSV infant
        @info "flow chart for id $i: $(join(flow, " => "))" 
        
        # add the total hospital days 
        total_hosp_days += sampled_days["icu"] + sampled_days["pediatricward"]

        # calculate qalys and costs
        tq, ql = calculate_qaly(x, sampled_days) 
        tc, cl = calculate_costs(x, sampled_days)
        totalqalys += tq
        totalcosts += tc
        qalyslost_death += ql 
        totalcosts_death += cl

        # calculate number of inpatient/outpatient/ non-MA episodes
        total_inpatients += length(findall(x -> x == "inpatient", flow))
        total_outpatients += length(findall(x -> x == "outpatient", flow))
        total_non_ma += length(findall(x -> x == "nonma", flow))

        # save all of the individual level information as a named tuple -- will be turned into a dataframe to store as a CSV 
        # flowend = flow[end]
        # push!(data, (;x.idx, x.monthborn, x.preterm, x.vac_lama, x.vac_mat, infection_tup(x)..., tq, ql, flowend))
    end
    # save simulation specific data as a csv file for debug purposes later
    # df = DataFrame(data) 
    # CSV.write("./output/sim_$(randstring(5)).csv", df)
    #return qalyslost

    return length(newborns), totalqalys, qalyslost_death, totalcosts, totalcosts_death, total_inpatients, total_outpatients, total_non_ma, total_hosp_days
end

function infection_tup(x) 
    # this function returns a named n-tuple of rsvmonth and rsvtype for both episodes (-99 if there is no second episode)
    # the argument expects atleast a single infection, otherwise will crash. 
    rsvmonth1 = x.rsvmonth[1]
    rsvtype1 = x.rsvtype[1]
    rsvmonth2 = -99
    rsvtype2 = -99
    if x.rsvpositive == 2 # if 2 symptomatic episodes, check whether it happens within 12 months 
        diff = x.rsvmonth[2] - x.monthborn
        if diff <= 11 
            rsvmonth2 = x.rsvmonth[2] 
            rsvtype2 =  x.rsvtype[2] 
        end
    end
    (; rsvmonth1, rsvtype1, rsvmonth2, rsvtype2)
end

# function test_qaly_distributions() 
#     # plots the PDFs of the QALY distributions 
#     a = hist(rand(qaly_prior_to_infection, 10000), bs=0.005)
#     b = hist(rand(qaly_symptomatic, 10000), bs=0.005)
#     c = hist(rand(qaly_pediatricward, 10000), bs=0.005)
#     d = hist(rand(qaly_icu, 10000), bs=0.005)
#     e = hist(rand(qaly_wheezing, 10000), bs=0.005)
#     dd = [a, b, c, d, e]
#     #h = hist(e, bs = 0.005)
#     @gp "reset" 
#     @gp :- a.bins a.counts "with boxes title 'non-rsv'" :- 
#     @gp :- b.bins b.counts "with boxes title 'symp'" :- 
#     @gp :- c.bins c.counts "with boxes title 'ward'" :- 
#     @gp :- d.bins d.counts "with boxes title 'icu'" :- 
#     @gp :- e.bins e.counts "with boxes title 'wheezing'" :- 
#     display(@gp)
#     #@gp "reset"
#     #@gp :- h.bins h.counts 
# end

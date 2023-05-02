## model for running cost-effectiveness of RSV. 
using Distributions, StatsBase, Random, StaticArrays
using DelimitedFiles
using IterTools


# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    age::Int64 = 0   # in years. don't really need this but left it incase needed later
    ageindays::Int64 = 0 # for new borns, we want to stratify it between 0:30 days, 30:180, 
    newborn::Bool = false # whether person is new born 
    monthborn::Int64 = 0 # month of the born baby
    activeaging::Bool = true # if active, the agent ages every month (this is to prevent newborns that have not been born yet to age)
    preterm::Bool = false
    gestation::Int64 = 0
    houseid::Int64 = 0
    sibling::Bool = false # we want to tag newborns that have a sibling <5.
    rsvpositive::Int64 = 0 
    rsvmonth::Vector{Int64} = Int64[]
    rsvage::Vector{Int64} = Int64[]
    rsvtype::Vector{Int64} = Int64[]
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters    ## use @with_kw from Parameters
    popsize::Int64 = 100000
    modeltime::Int64 = 300
end

const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
const ON_POP = 13448500 # 13448495 from ontario database, but their numbers don't add up
export humans

function main(ip::ModelParameters) 
    println("running simulation")
end

function main() 
    println("running simulations - testing revise")
    reset_params()  # reset the parameters for the simulation scenario    
    initialize() 
    household() 
    tag_siblings() 
end
export main

## Initialization Functions 
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

pc(x) = Int(round(x / ON_POP * 100000)) # 13_448_495: max population in ontario 

function initialize() 
    #reset to defauls 
    for i = 1:length(humans)
        humans[i] = Human() 
    end

    # demographic data 
    # ontario census data: https://www11.statcan.gc.ca/census-recensement/1016/dp-pd/prof/details/Page.cfm?Lang=E&Geo1=PR&Code1=35&Geo1=&Code1=&Data=Count&SearchText=Ontario&Sear
    # convert to per 100,000 
    ages_per_ag = @SVector [0:4, 5:19, 20:64, 65:99]
    size_per_ag = pc.([697360, 2322285, 8177200, 2251655]) # from census data 
    sum(size_per_ag) != ON_POP && "error -- age groups don't add up to total population size"
    println("pop sizes from census data per capita: $size_per_ag, sum: $(sum(size_per_ag))")
    
    # data on babies/newborn, order is June to May multiplied twice since we are doing two years
    babies_per_month = pc.(@SVector [11830, 13331, 13149, 13051, 13117, 13618, 11376, 11191, 11141, 11614, 11864, 11095])  
    preterms_per_month = pc.(@SVector [899, 888, 1016, 983, 1044, 1050, 1089, 1067, 1060, 1010, 949, 910])
    prop_pre_terms = preterms_per_month ./ babies_per_month # we do it this way to generate some stochasticity instead of fixing the number of preterms per month
    println("newborns distribution: $babies_per_month, sum: $(sum(babies_per_month)), times two: $(sum(babies_per_month) *2)")

#    println("$babies_per_month")    
    # from the first age group, remove n agents to be added in as newborns later
    size_per_ag[1] = size_per_ag[1] - (sum(babies_per_month) * 2) # x2 because we are doing two years worth of babies 
    println("adjusted pop sizes (removed newborns from first age group: $size_per_ag, sum: $(sum(size_per_ag))")
    # initialize the non-newborns first
    sz_non_newborns = sum(size_per_ag)
    human_ages = rand.(inverse_rle(ages_per_ag, size_per_ag)) # vector of ages for each human
    shuffle!(human_ages)
    println("proportion of 0:4s that are 0 or 1: $(length(findall(x -> x <= 1, human_ages)))")
    idx = 0
    for i in 1:sz_non_newborns
        idx += 1 # we need to know when the loop ends so we know where to begin the next one for the babies
        #humans[idx] = Human()  # keep everything else as default
        x = humans[idx]
        x.idx = idx # we can use the i 
        x.age = human_ages[idx]  
        x.ageindays = 731 # 730 = two years
        if x.age >=2 
            x.ageindays = 731 
        else # a baby is 0 or 1 year old but they are not newborns that are being trakced (perhaps immigration or movement)
            x.monthborn = 0 # we don't really care about the monthborn, because we are not tracking these babies
            x.ageindays = rand(30:730) # but lets give them an age in days because we want them to be included in buckets
            # if x.ageindays <= 30 
            #    x.monthborn = 1
            # end
            # if its 1:730, the assumption is that there are some newborns that are not part of the "count" seyed gave me (immigration, etc)
            # if its 30:730, that means there are newborns from before the simulation started -- this doesnt even matter, since 
        end
    end
    println("pop assigned ageindays between 30-730 days: $(length(findall(x -> x.ageindays <= 730 && x.ageindays > 0, humans)))")
    
    # make sure no one has a monthassigned for being born
    newborns = findall(x -> x.monthborn >= 1, humans)
    println("agents with month assigned: $(length(newborns)) -- should be zero.")
 
    # probability of preterm 
    prob_gestation = Categorical([0.07, 0.17, 0.76]) # 1 = <29, 2 = 29-32 weeks, 3 = 33-35 weeks

    # next initialize the newborns (2 years worth) -- use two loops to do two years
    sz_newborns = sum(babies_per_month)
    for y in [:y1, :y2]
        month_idx = inverse_rle(1:12, babies_per_month) # for each baby, assign them a birth month 
        println(countmap(month_idx))
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

    # monthborn should only be for newborns that we are tracking

    newborns = findall(x -> x.monthborn >= 1, humans)
    newborns_tracked = findall(x -> x.monthborn >= 1 && x.newborn == true, humans)
    println("total newborns: $(length(newborns)), tracked (should be half): $(length(newborns_tracked))")
    return idx
end


function age_statistics() 
    # statistics
    _wg = length(findall(x -> x.ageindays < 999, humans))
    println("number of individuals with an age assigned: $_wg")

    _wg = length(findall(x -> x.newborn == true, humans))
    println("number ofnew born tracking: $_wg")


    # print/save demographics statistics 
    nbpm = countmap([x.monthborn for x in humans if x.newborn == true])
    println("distribution of newborns (month => cnt): $nbpm")
end

function household() 
    # reset houseids for everyone 
    for x in humans 
        x.houseid = 0
    end
    adult_indices = shuffle!(findall(x -> x.age > 19 && x.age <= 65, humans))
    grand_indices = shuffle!(findall(x -> x.age > 65, humans))
    child_indices = shuffle!(findall(x -> x.age <= 19, humans))

    # get the total number of individuals in 1,1,3,4,5-person households
    household_sizes = [0, 1_815_000, 844545, 791435, 330560] # 1, 1, 3, 4, 5+ (fill in the 1 size household in the next step)
    single_adults = ON_POP - sum(household_sizes .* [1, 1, 3, 4, 5]) # calculate the 1-person household size by using the total population 
    household_sizes[1] = single_adults 
    prop_household_sizes = household_sizes ./ sum(household_sizes) 

    println("total adults: $(length(adult_indices)), grandparents: $(length(grand_indices)), children: $(length(child_indices))")
    println("population (size) in households $(household_sizes .* [1, 1, 3, 4, 5]), total sum: $(sum(household_sizes .* [1, 1, 3, 4, 5]))")
    
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

    println("total families from census: $_totalfamilies")
    println("fam with 1 child: $families_with_1_child ($prop_families_with_1_child%), 1: $families_with_1_child ($prop_families_with_1_child%), 3: $families_with_3_child ($prop_families_with_3_child %) => total families: $totalfamilies")
  
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
   
    # ## step 1: go through the adults that are supposed to be in single households, no children/grandparents
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
    # for a given month, "activate" the newborns so they can go through the aging process
    # do it this way to prevent aging of babies that are not born yet
    @info "activating newborns"
    nb_idx = findall(x -> x.monthborn == mth, humans) 
    for i in nb_idx 
        x = humans[i] 
        x.activeaging = true # turn on aging 
    end
end

function within_3_month(x, mth)
    # a condition function to check if agents are eligible for infection
    # max number of infections is 1 and minimum 3 months between infections
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
    all_baby_idx = findall(x -> x.activeaging == true, humans)
    for idx in all_baby_idx
        x = humans[idx]
        x.ageindays = min(x.ageindays + 30, 731) 
    end
end

function get_monthly_agegroup_infcnt() 
    # function samples the yearly counts for both MA (medically attended) and N-MA (symptomatic, non medical) 
    # splits it over age groups
    # and splits it over months 
    # run this function twice if you need for two years. 

    # rate? of infection from Seyed (see Excel sheet)
    #<19 d, 19–89 d, 90d to <6mo, 6mo to <1yr, 1 to < 1 yr, 1–5 yr, 6–17 yr, 18–49 yr, 50–64 yr, 65–79 yr, ≥ 80 yr
    # _rates_per_age = @SVector [0.051, 0.147, 0.111, 0.119, 0.166, 0.145, 0.016, 0.046, 0.053, 0.064, 0.071]

    # get medically attended (MA) counts per year/season
    rates_per_agegroup = @SVector [0.051, 0.147, 0.111, 0.119, 0.166] # <19 d, 19–89 d, 90d to <6mo, 6mo to <1yr, 1 to < 1 yr
    yearly_incidence = rand(Uniform(1001, 2439)) # yearly incidence sampled from a range -- this is for ALL age groups
    yearly_incidence_per_ag = yearly_incidence * rates_per_agegroup # split the yearly counts to the age groups we are interested in 
    sum_incidence_per_ag = sum(yearly_incidence_per_ag) # add up incidence for selected age groups only 
    @info ("yearly sampled MA: $yearly_incidence")
    @info ("yearly MA per age group:\n $yearly_incidence_per_ag, \ntotal incidence (over 5 age grps): $(sum_incidence_per_ag)")
    
    # get non-medically (N-MA) (symptomatic) counts -- it's basically the MA increased by some percentage
    nma_range = rand(Uniform(0.15, 0.30)) # percentage of symptomatic (non-medical attended)
    nma_incidence = sum_incidence_per_ag * nma_range # because symptomatic = MA * (1 + x%)
    nma_rates_per_agegroup = @SVector [0, 0.011, 0.043, 0.185, 0.761] # split the non-medically attended counts over age groups - given by Seyed 
    yearly_nma_per_ag = nma_incidence * nma_rates_per_agegroup 
    
    @info ("total sampled N-MA: $nma_incidence")
    @info ("yearly N-MA per age group:\n $yearly_nma_per_ag, \ntotal incidence (over 5 age grps): $(sum(yearly_nma_per_ag))")
    
    # as a check, see how many babies we have
    _cr = length(findall(x -> x.ageindays <= 730, humans)) 
    @info ("sanity check: total children at risk: $_cr, total incidence to distribute: $(sum(yearly_incidence_per_ag) + sum(yearly_nma_per_ag))")

    # the distribution of yearly cases to monthly cases (starting at January) 
    # this works for both MA and N-MA cases
    # this is split between the age groups:  #<19 d, 19–89 d, 90d to <6mo, 6mo to <1yr, 
    # order june to may
    month_distr_ag1 = @SVector [0.0141, 0.0141, 0.0181, 0.0563, 0.0845, 0.1117, 0.1409, 0.1549, 0.1408, 0.1117, 0.0845, 0.0563]
    month_distr_ag1 = @SVector [0.0141, 0.0141, 0.0181, 0.0563, 0.0845, 0.1117, 0.1409, 0.1549, 0.1408, 0.1117, 0.0845, 0.0563]
    month_distr_ag3 = @SVector [0.0108, 0.0108, 0.0417, 0.0615, 0.0833, 0.115, 0.1667, 0.1667, 0.115, 0.0833, 0.0615, 0.0417]
    month_distr_ag4 = @SVector [0.0357, 0.0357, 0.0357, 0.0714, 0.1071, 0.1419, 0.1430, 0.1419, 0.1071, 0.0714, 0.0714, 0.0357]
    month_distr_ag5 = @SVector [0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.1111, 0.1663, 0.1667, 0.1111, 0.0556, 0.0556, 0.0556]
    month_distr = [month_distr_ag1, month_distr_ag1, month_distr_ag3, month_distr_ag4, month_distr_ag5]

    # a matrix of age groups x monthly incidence (dim: 5 x 11)
    _monthly_inc = @. (yearly_incidence_per_ag * month_distr) 
    monthly_agegroup_incidence = Int.(ceil.(transpose(hcat(_monthly_inc...))))
    _monthly_nma = @. (yearly_nma_per_ag * month_distr)
    monthly_agegroup_non_ma = Int.(ceil.(transpose(hcat(_monthly_nma...))))
    display(monthly_agegroup_incidence)

    println("yearly incidence (N-MA) split into months per age group:")
    display(monthly_agegroup_non_ma)

    # to do: 
    println("total (theoretical) monthly incidence (MA) across age groups: \n$(round.(sum(monthly_agegroup_incidence, dims=1)))")
    println("total (theoretical) monthly incidence (N-MA) across age groups: \n$(round.(sum(monthly_agegroup_non_ma, dims=1)))")
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
    yr1_monthly_agegroup_incidence, yr1_monthly_agegroup_non_ma = get_monthly_agegroup_infcnt()    
    monthly_agegroup_incidence = hcat(yr1_monthly_agegroup_incidence, yr1_monthly_agegroup_incidence)
    monthly_agegroup_non_ma = hcat(yr1_monthly_agegroup_non_ma,yr1_monthly_agegroup_non_ma)
    
    # loop through months: Order is June => 1, July => 1, etc (verify that the vectors used to inject babies and sample monthly counts are in the right order!) 
    for mth in 1:24
        @info "simulating month: $mth"
        activate_newborns(mth) 

        # eligble children (0 - 730 days of age) split into 5 age groups (because the counts are sampled over these five age groups)
        ag1_idx = findall(x -> x.ageindays > 0  && x.ageindays <= 30 && within_3_month(x, mth) && x.monthborn == mth, humans) 
        ag2_idx = findall(x -> x.ageindays > 30  && x.ageindays <= 90 && within_3_month(x, mth), humans)
        ag3_idx = findall(x -> x.ageindays > 90  && x.ageindays <= 180 && within_3_month(x, mth), humans)
        ag4_idx = findall(x -> x.ageindays > 180 && x.ageindays <= 365 && within_3_month(x, mth), humans)
        ag5_idx = findall(x -> x.ageindays > 365 && x.ageindays <= 730 && within_3_month(x, mth), humans)
        ag_idx = [ag1_idx, ag2_idx, ag3_idx, ag4_idx, ag5_idx]

        # get the month specific infection counts for all age groups
        ma_to_distr = monthly_agegroup_incidence[:, mth]
        nma_to_distr = monthly_agegroup_non_ma[:, mth]

        @info """ incidence cnts to be sampled--  MA: $ma_to_distr, N-MA: $nma_to_distr
        num of babies in our system: $(length.(ag_idx))
        """
        
        # to understand whats happening here: run this command: sample.([[1, 1, 3, 4, 5], [11, 11, 13, 14, 15]], [1, 3], replace=false)
        # when we sample, we combine the MA and N-MA together but then split it later according to the counts
        sampled_idx_to_be_sick = sample.(ag_idx, ma_to_distr .+ nma_to_distr, replace=false)
        #ag_sick = vcat(_ag_sick...)     # now we list of IDs that need to be infected, we don't care about the structure (which at this stage is an array or arrays)
        #println("total (sampled) monthly infected across age groups: $(length(ag_sick))")
        
        ## at this point, drop all ag_sick indices from ag1_idx

        for (idx, ag_sick_ids) in enumerate(sampled_idx_to_be_sick)
            # ag_sick is the array of indices of agents to make sick (but they are combined between MA and N-MA)
            # ag_idx is the age group from 1 to 5 
            # generate a pre-list of whether individual is MA = 1 or N-MA = 2
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
    println("newborns not activated (should be zero): $(length(findall(x -> x.newborn == true && x.activeaging == false, humans)))")
    return totalsick
end    

function outcome_flow(x) 
    # sample the potential outcomes for sick newborn

    prob_inpatient_fullterm = [2.02, 3.74, 2.32, 1.65, 1.425, 0.95, 0.83, 0.87, 0.655, 0.745, 0.61, 0.54] ./ 100
    prob_inpatient_preterm_1 = [6.06, 11.22, 6.96, 4.95, 4.275, 2.85, 2.49, 2.61, 1.965, 2.235, 1.83, 1.62] ./ 100 
    prob_inpatient_preterm_2 = [4.04, 7.48, 4.64, 3.3, 2.85, 1.9, 1.66, 1.74, 1.31, 1.49, 1.22, 1.08] ./ 100
    prob_inpatient_preterm_3 = [2.02, 3.74, 2.32, 1.65, 1.425, 0.95, 0.83, 0.87, 0.655, 0.745, 0.61, 0.54] ./ 100
    prob_inpatient_preterm = [prob_inpatient_preterm_1, prob_inpatient_preterm_2, prob_inpatient_preterm_3]
    
    prob_inpatient = 0 # not defined... gets defined in the loop since it depends on the infection count.
    prob_icu = 0.13 # otherwise pedward 
    prob_recovery = 1 - 0.005 # other recovery 
    prob_wheezing = 0.31 # will have to sample duration for cost-effectiveness 
    prob_emergencydept = rand(Uniform(0.4, 0.5))

    days_symptomatic_distr = Uniform(5.68, 6.63) 

    # create empty array to store the flow of outcomes
    flow = String[]
   
    # check if second infection happens within 12 months
    infcnt_max = 1 # we know there is one infection minimum
    if x.rsvpositive == 2 # if 2 symptomatic episodes, check whether it happens within 12 months 
        diff = x.rsvmonth[2] - x.monthborn[1]
        infcnt_max = diff <= 11 ? 2 : 1
    end
    println("infcnt: $infcnt_max")

    # A dict to hold sampled values as they are being calculated
    sampled_days = Dict(
        "prior_to_infection" => 0., 
        "after_infection" => 0., 
        "symptomatic" => 0., 
        "pediatricward" => 0., 
        "icu" => 0., 
        "wheezing" => 0., 
        "emergencydept" => 0.,  # these two only really used for cost purposes... not for QALYs
        "office_consultation" => 0
    )

    # insert the number of days prior to first infection
    sampled_days["prior_to_infection"] += (x.rsvmonth[1] - x.monthborn) * 30
    
    for ic in 1:infcnt_max 
        
        # for each infection, determine their probability of inpatient based on when infection happens (i.e, how many months after birth)
        distr = x.preterm ? prob_inpatient_preterm[x.gestation] : prob_inpatient_fullterm
        diff = (x.rsvmonth[ic] - x.monthborn) + 1 # the prob of inpatient depends on when infection happens after birth
        prob_inpatient = distr[diff]

        days_symptomatic = rand(days_symptomatic_distr)
        sampled_days["symptomatic"] += days_symptomatic

        if rand() < prob_inpatient && x.rsvtype[ic] == 1  ## a non-medical attended person can not be inpatient... only symptomatic
            push!(flow, "inpatient")
            if rand() < prob_icu 
                push!(flow, "icu")
                sampled_days["icu"] += x.preterm ? rand(Gamma(20.22, 0.47)) : rand(Gamma(12.38, 0.42))
            else # pediatric ward
                push!(flow, "pediatric ward")
                sampled_days["pediatricward"] += x.preterm ? rand(Gamma(12.71, 0.48)) : rand(Gamma(6.08, 0.64)) 
            end
            if rand() < prob_recovery  # after ICU or PedWard, check if recovered or death 
                push!(flow, "recovery")
                if rand() < prob_wheezing # wheezing episode
                    push!(flow, "wheezing")
                    sampled_days["wheezing"] +=  rand(Uniform(5.2, 9.8))
                end
            else 
                push!(flow, "death")
            end
        else # is either office ot ED
            push!(flow, "outpatient")  
            if rand() < prob_emergencydept
                push!(flow, "ED")
                sampled_days["emergencydept"] += 1
            else 
                push!(flow, "office")
                sampled_days["office_consultation"] += 1
            end
        end
    end
    # add the remaining days  
    sampled_days["after_infection"] += 365 - sum(values(sampled_days))

    # print statistics
    println("")
    println(join(flow, " => "))
    println("")
    display(sampled_days)
    println("total days: $(sum(values(sampled_days)))")

    return sampled_days # return the number of days in each outcome state
end

function f2()
    # find all newborns that had symptomatic epideo
    newborns = findall(x -> x.newborn == true && x.rsvpositive > 0 && (x.rsvmonth[1] - x.monthborn) <= 11, humans)
    println("\ntotal rsv positive newborns: $(length(newborns))")

    # create the QALY distributions (disutility)
    qaly_prior_to_infection = Beta(19.2, 364.6)
    qaly_symptomatic = Beta(53.6, 281.4)
    qaly_pediatricward = Beta(109.7, 157.9)
    qaly_icu = Beta(159.4, 106.2)
    qaly_wheezing = Beta(14.1, 338.4)

    for (i, h) in enumerate(newborns) 
        x = humans[h]   
        println(h)
        break
        dump(x)
        sampled_days = outcome_flow(x)
        # A dict to hold qaly values as they are being calculated
        qalys = Dict(
            "prior_to_infection" => 0., 
            "after_infection" => 0., 
            "symptomatic" => 0., 
            "pediatricward" => 0., 
            "icu" => 0., 
            "wheezing" => 0., 
        )
        sm = rand(qaly_prior_to_infection)   
        qalys["prior_to_infection"] =  sampled_days["prior_to_infection"] / 365 * sm 
        qalys["after_infection"] = sampled_days["after_infection"] / 365 * sm
        qalys["symptomatic"] = sampled_days["symptomatic"] / 365 * rand(qaly_symptomatic)
        qalys["pediatricward"] = sampled_days["pediatricward"] / 365 * rand(qaly_pediatricward)
        qalys["icu"] = sampled_days["icu"] / 365 * rand(qaly_icu)
        qalys["wheezing"] = sampled_days["wheezing"] / 365 * rand(qaly_wheezing)
        
        #display(qalys)
        println("Total QALY decrement: $(sum(values(qalys))), QALY decrement (RSV): $(qalys["after_infection"] + qalys["symptomatic"] + qalys["pediatricward"] + qalys["icu"] + qalys["wheezing"]) QALY: $(1 - sum(values(qalys))) ")
    
        cost_in_icu = 3638 * sampled_days["icu"] # 10, 2 symptomatics
        cost_in_ward = 1491 * sampled_days["pediatricward"]
        cost_hosp_followup = begin 
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
        cost_wheezing = begin 
            cw = 0 
            if sampled_days["wheezing"] > 0
                cw = sampled_days["wheezing"] > 10 ? (229 * 2) : 229
            end
            cw
        end
        cost_outpatient = (sampled_days["emergencydept"] * 342) + (sampled_days["office_consultation"] * 229) 
    
        println("cost_in_icu: $cost_in_icu, 
        cost_in_ward: $cost_in_ward, \
        hosp followup: $cost_hosp_followup, 
        cost_wheezing: $cost_wheezing,  
        cost outpatient: $cost_outpatient")

        # SAVE EVERYTHING AS TUPLE 
        (;x.idx, x.monthborn, x.preterm, split_infections..., )
    end
   

    # save everything to a dataframe
    
    #     nb_data[i] = (;x.idx, x.monthborn, x.preterm, rsvmonth1, rsvage1, rsvtype1, rsvmonth2, rsvage2, rsvtype2)
  
end

function split_infections(x) 
    rsvmonth1 = x.rsvmonth[1]
    rsvage1 = x.rsvage[1]
    rsvtype1 = x.rsvtype[1]

    rsvmonth2 = -99
    rsvage2 = -99
    rsvtype2 = -99
    if x.rsvpositive == 2 # if 2 symptomatic episodes, check whether it happens within 12 months 
        diff = x.rsvmonth[2] - x.monthborn
        if diff <= 11 
            rsvmonth2 = x.rsvmonth[2] 
            rsvage2 = x.rsvage[2]
            rsvtype2 =  x.rsvtype[2] 
        end
    end
    (; rsvmonth1, rsvage1, rsvtype1, rsvmonth2, rsvage2, rsvtype2)
end

# function save_simulation() 
#     nb = findall(x -> x.newborn == true && x.rsvpositive > 0, humans)
#     nb_data = Array{NamedTuple{(:idx, :monthborn, :preterm, :rsvmonth1, :rsvage1, :rsvtype1, :rsvmonth2, :rsvage2, :rsvtype2), Tuple{Int64, Int64, Bool, Int64, Int64, Int64, Int64, Int64, Int64}}}(undef, length(nb))

#     println("nb sick: $(length(nb))")

#     for (i, hid) in enumerate(nb)
#         x = humans[hid] 

#         # test, make sure activeaging == true 
#         rsvmonth1 = x.rsvmonth[1]
#         rsvage1 = x.rsvage[1]
#         rsvtype1 = x.rsvtype[1]

#         rsvmonth2 = -99
#         rsvage2 = -99
#         rsvtype2 = -99
#         if x.rsvpositive == 2 # if 2 symptomatic episodes, check whether it happens within 12 months 
#             diff = x.rsvmonth[2] - x.rsvmonth[1]
#             if diff <= 11 
#                 rsvmonth2 = x.rsvmonth[2] 
#                 rsvage2 = x.rsvage[2]
#                 rsvtype2 =  x.rsvtype[2] 
#             else    
#                 @info "id: $hid - two infections > 12 months"
#             end
#         end
#         nb_data[i] = (;x.idx, x.monthborn, x.preterm, rsvmonth1, rsvage1, rsvtype1, rsvmonth2, rsvage2, rsvtype2)
#         #
#     end
#     df = DataFrame(nb_data) 
#     println(countmap(df.monthborn))
#     CSV.write("rsv_sim_data.csv", df)
#     return df
# end

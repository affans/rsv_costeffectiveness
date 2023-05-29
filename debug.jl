# file to run debug functions for model correctness  
# run main() before running these functions.

function vaccine_debug() 
    # debug -- save data in a file to check for bugs 
    newborns = findall(x -> x.newborn == true, humans)
    
    aa = [humans[x].monthborn for x in newborns]
    ty = hcat([humans[x].vac_mat + humans[x].vac_lama for x in newborns]...)
    eo = hcat([humans[x].eff_outpatient for x in newborns]...)
    eh = hcat([humans[x].eff_hosp for x in newborns]...)
    ei = hcat([humans[x].eff_icu for x in newborns]...)
    cc = [aa ty' eo']
    #writedlm("vaccine_test.csv", cc, ",")
    cc
end

function test_infection_monthborn() 
    # function tests that infection always occurs after the monthborn 
    mb = findall(x -> x.newborn == true && x.rsvpositive > 0, humans) 
    aa = [humans[x].monthborn for x in mb]
    bb = [humans[x].rsvmonth[1] for x in mb]
    for (a, b) in zip(aa, bb)
        b < a && println("TEST FAILED")
    end
    println("TEST PASSED")
    cc = [aa bb]
end


function test_outcome_flows() 
    # edits a newborn manually for testing purposes 
    x = humans[97826] 
    x.newborn = true
    x.monthborn = 1
    x.activeaging = true 
    x.preterm = false 
    x.gestation = 0
    x.houseid = 0 
    x.sibling = 0 
    x.rsvpositive = 2 
    x.rsvmonth = [2, 8]
    x.rsvage = [0, 0]
    x.rsvtype = [1, 1]
    x.vac_lama = false
    x.vac_mat = false
    x.eff_outpatient = zeros(Float64, 24)
    x.eff_hosp = zeros(Float64, 24)
    x.eff_icu = zeros(Float64, 24) #zeros(Float64, 24)
    dump(x)
    s, f = outcome_flow(x)
    calculate_qaly(x, s)
end

function check_s5() 
    totalvaccinated = findall(x -> sum(x.eff_outpatient) > 0, humans)

    totalg1g2 = findall(x -> x.newborn == true && x.gestation in (1, 2), humans)
    totalg1g2_v = findall(x -> x.newborn == true && x.gestation in (1, 2) && sum(x.eff_outpatient) > 0, humans)

    totalcomorbid_g3 = findall(x -> x.newborn == true && x.gestation == 3 && x.comorbidity > 0, humans)
    totalcomorbid_g3_v = findall(x -> x.newborn == true && x.gestation == 3 && x.comorbidity > 0 && sum(x.eff_outpatient) > 0, humans)

    totalcomorbid_ft = findall(x -> x.newborn == true && x.preterm == false && x.comorbidity > 0, humans)
    totalcomorbid_ft_v = findall(x -> x.newborn == true && x.preterm == false && x.comorbidity > 0 && sum(x.eff_outpatient) > 0, humans)

    n_tv = length(totalvaccinated)
    n_g12 = length(totalg1g2)
    n_g3 = length(totalcomorbid_g3)
    n_ft = length(totalcomorbid_ft)

    println("""
        total vaccinated: $(n_tv),

        total g1/g2: $(n_g12), vaccinated: $(length(totalg1g2_v))
        total g3: $(n_g3), vaccinated: $(length(totalcomorbid_g3_v))
        total n_ft: $(n_ft), vaccinated: $(length(totalcomorbid_ft_v))
    """)
end
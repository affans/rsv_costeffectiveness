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
    outcome_flow(x)
end
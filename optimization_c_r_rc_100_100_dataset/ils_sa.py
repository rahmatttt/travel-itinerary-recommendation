#HYBRID ITERATED LOCAL SEARCH - SIMULATED ANNEALING
import random
import copy
import numpy as np
from optimization_c_r_rc_100_100_dataset.base import BaseILS

class ILS_SA(BaseILS):
    def __init__(self,stepsize = 0.1,strength=0.5,temperature=1000,cooling_rate=0.95,stopping_temperature=0.0002,max_iter = 100,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.stepsize = stepsize #step size in local search process
        self.strength = strength #perturbation strength (to escape the local optimum)
        self.temperature = temperature #initial temperature for SA process
        self.stopping_temperature = stopping_temperature #minimum temperature to continue iteration for SA process
        self.cooling_rate = cooling_rate #cooling rate for SA process
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
    
    def local_search(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(solution)))
        temperature = self.temperature
        
        while temperature >= self.stopping_temperature:
            #generate neighbor
            candidate = solution + np.random.uniform(-self.stepsize,self.stepsize, size=len(solution))
            candidate_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(candidate)))
            
            #calculate acceptance probability
            acceptance_prob = np.exp((candidate_fitness - fitness) / temperature)
            
            #accept if better
            if candidate_fitness > fitness or random.uniform(0,1) < acceptance_prob:
                solution = copy.deepcopy(candidate)
                fitness = candidate_fitness
            
            temperature = self.cooling_rate * temperature
        
        return solution,fitness
            
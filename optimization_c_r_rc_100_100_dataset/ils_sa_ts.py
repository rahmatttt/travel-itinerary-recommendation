#HYBRID ITERATED LOCAL SEARCH - SIMULATED ANNEALING
import random
import copy
import numpy as np
from optimization_c_r_rc_100_100_dataset.base import BaseILS

class ILS_SA_TS(BaseILS):
    def __init__(self,stepsize = 0.1,strength=0.5,max_tabu_size = 10,num_neighbor = 10,temperature=1000,cooling_rate=0.95,stopping_temperature=0.0002,max_iter = 100,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.stepsize = stepsize #step size in local search process
        self.strength = strength #perturbation strength (to escape the local optimum)
        self.max_tabu_size = max_tabu_size #maximum size of tabu list
        self.num_neighbor = num_neighbor #the number of neighbor in neighborhood for new solution candidates in TS process
        self.temperature = temperature #initial temperature for SA process
        self.stopping_temperature = stopping_temperature #minimum temperature to continue iteration for SA process
        self.cooling_rate = cooling_rate #cooling rate for SA process
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
    
    def local_search_sa(self,solution_nodes):
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
    
    def local_search_ts(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(solution)))
        
        tabu_list = []
        
        idem_counter = 0
        for i in range(self.max_iter):
            add_idem = True
            #generate neighborhood
            for j in range(self.num_neighbor):
                addition = np.random.uniform(-self.stepsize, self.stepsize, size=len(solution))
                if len(tabu_list) > 0:
                    if np.any(np.all(addition == tabu_list, axis=1)) == False:
                        neighbor = solution + addition
                        neighbor_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor)))
                        tabu_list.append(addition)
                else:
                    neighbor = solution + addition
                    neighbor_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor)))
                    tabu_list.append(addition)
                
                if neighbor_fitness > fitness:
                    solution = copy.deepcopy(neighbor)
                    fitness = neighbor_fitness
                    add_idem = False
            
            #check tabu list
            if len(tabu_list) > self.max_tabu_size:
                tabu_list = tabu_list[1:]
            
            if add_idem == True:
                idem_counter += 1
            else:
                idem_counter = 0
            
            if idem_counter > self.max_idem:
                return solution,fitness
        
        return solution,fitness
    
    def hybrid_local_search(self,solution_nodes):
        sol_sa,fitness_sa = self.local_search_sa(solution_nodes)
        sol_ts,fitness_ts = self.local_search_ts(solution_nodes)
        
        if fitness_sa > fitness_ts:
            return sol_sa,fitness_sa
        else:
            return sol_ts,fitness_ts
    
    def construct_solution(self):
        idem_counter = 0
        
        best_solution,best_fitness = self.hybrid_local_search(self.generate_init_solution())
        
        for i in range(self.max_iter):
            #perturbation
            candidate = self.perturbation(best_solution)
            
            #local search
            candidate,candidate_fitness = self.hybrid_local_search(candidate)
            
            if candidate_fitness > best_fitness:
                best_solution = copy.deepcopy(candidate)
                best_fitness = candidate_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))
                    return best_solution,best_fitness
        
        best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))
        return best_solution,best_fitness
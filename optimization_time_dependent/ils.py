#ITERATED LOCAL SEARCH
import copy
import numpy as np
from optimization_time_dependent.base import BaseILS

class ILS_VRP_TD(BaseILS):
    def __init__(self,stepsize = 0.1,strength=0.5,max_iter = 100,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state,seed_numpy = True)

        # parameter setting
        self.stepsize = stepsize #step size in local search process
        self.strength = strength #perturbation strength (to escape the local optimum)
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        #degree of interest (DOI for MAUT) setting
        self.degree_waktu = None
        self.degree_tarif = None
        self.degree_rating = None
    
    def local_search(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(solution))))
        
        for i in range(self.max_iter):
            #generate neighbor
            candidate = solution + np.random.uniform(-self.stepsize,self.stepsize, size=len(solution))
            candidate_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(candidate))))
            
            #accept if better
            if candidate_fitness > fitness:
                solution = copy.deepcopy(candidate)
                fitness = candidate_fitness
        
        return solution,fitness
            
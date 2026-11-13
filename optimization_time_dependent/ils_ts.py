#HYBRID ITERATED LOCAL SEARCH - TABU SEARCH
import copy
import numpy as np
from optimization_time_dependent.base import BaseILS

class ILS_TS_VRP_TD(BaseILS):
    def __init__(self,stepsize = 0.1,strength=0.5,max_tabu_size = 10,num_neighbor = 10,max_iter = 100,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state,seed_numpy = True)

        # parameter setting
        self.stepsize = stepsize #step size in local search process
        self.strength = strength #perturbation strength (to escape the local optimum)
        self.max_tabu_size = max_tabu_size #maximum size of tabu list
        self.num_neighbor = num_neighbor #the number of neighbor in neighborhood for new solution candidates in TS process
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        #degree of interest (DOI for MAUT) setting
        self.degree_waktu = None
        self.degree_tarif = None
        self.degree_rating = None
    
    def local_search(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(solution))))
        
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
                        neighbor_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor))))
                        tabu_list.append(addition)
                else:
                    neighbor = solution + addition
                    neighbor_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor))))
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
            
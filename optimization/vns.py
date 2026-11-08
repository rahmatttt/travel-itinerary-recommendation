# VARIABLE NEIGHBORHOOD SEARCH 
import random
import copy
import datetime
from optimization.base import BaseOptimizer

class VNS_VRP(BaseOptimizer):
    def __init__(self,max_iter = 200,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state,seed_numpy = True)

        # parameter setting
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
    
    def set_model(self,tour,hotel,timematrix,init_solution=[],travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        else:
            self.init_solution = self.generate_init_solution()
    
    def n1(self,solution):
        sol = copy.deepcopy(solution)
        pos1,pos2 = sorted(random.sample(range(len(sol)),2))
        sol[pos1:pos2] = reversed(sol[pos1:pos2])
        return sol
    
    def n2(self,solution):
        sol = copy.deepcopy(solution)
        pos1,pos2,pos3 = sorted(random.sample(range(len(sol)),3))
        sol[pos1:pos2] = reversed(sol[pos1:pos2])
        sol[pos1:pos3] = reversed(sol[pos1:pos3])
        sol[pos2:pos3] = reversed(sol[pos2:pos3])
        return sol
    
    def n3(self,solution):
        sol = copy.deepcopy(solution)
        pos1,pos2 = random.sample(range(len(sol)),2)
        sol[pos1],sol[pos2] = sol[pos2],sol[pos1]
        return sol
    
    def construct_solution(self):
        best_solution = copy.deepcopy(self.init_solution)
        best_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(best_solution)))
        
        best_found_solution = copy.deepcopy(best_solution)
        best_found_fitness = best_fitness
        
        idem_counter = 0
        
        for i in range(self.max_iter):
            k = 1
            while k<=3:
                if k == 1:
                    solution = self.n1(best_found_solution)
                elif k == 2:
                    solution = self.n2(best_found_solution)
                else:
                    solution = self.n3(best_found_solution)
                fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(solution)))
                if fitness > best_fitness:
                    best_found_solution = solution
                    best_found_fitness = fitness
                
                k += 1
            
            if best_found_fitness > best_fitness:
                best_solution = copy.deepcopy(best_found_solution)
                best_fitness = best_found_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return self.solution_list_of_nodes_to_dict(self.split_itinerary(best_solution)),best_fitness
            
        return self.solution_list_of_nodes_to_dict(self.split_itinerary(best_solution)),best_fitness
import random
import copy
import datetime
import numpy as np
from optimization_time_dependent.base import BaseOptimizer

class SA_VRP_TD(BaseOptimizer):
    def __init__(self,temperature = 15000,cooling_rate=0.99,stopping_temperature=0.0002,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.temperature = temperature #temperature of SA
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        
        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
    
    def set_model(self,tour,hotel,timematrix,init_solution=[],start_day = 5,travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,start_day = start_day,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        self.init_solution = init_solution
    
    def swap_operation(self,solution):
        sol = copy.deepcopy(solution)
        pos1,pos2 = random.sample(range(len(sol)),2)
        sol[pos1],sol[pos2] = sol[pos2],sol[pos1]
        return sol
    
    def construct_solution(self):
        solution = random.sample(self.tour,len(self.tour)) if len(self.init_solution) <= 0 else copy.deepcopy(sum(self.init_solution,[])+self.get_rest_nodes_from_solution(self.init_solution))
        solution_dict = self.solution_list_of_nodes_to_dict(self.split_itinerary(solution))
        fitness = self.MAUT(solution_dict)
        while self.temperature >= self.stopping_temperature:
            #generate new solution
            new_solution = self.swap_operation(solution)
            new_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(new_solution)))
            
            if new_fitness > fitness:
                solution = new_solution
                fitness = new_fitness
            else:
                #calculate acceptance probability
                prob = np.exp(-(fitness - new_fitness)/self.temperature)
                if random.uniform(0,1) < prob:
                    solution = new_solution
                    fitness = new_fitness
            
            self.temperature = self.cooling_rate*self.temperature
        
        solution_dict = self.solution_list_of_nodes_to_dict(self.split_itinerary(solution))
        fitness = self.MAUT(solution_dict)
        return solution_dict,fitness
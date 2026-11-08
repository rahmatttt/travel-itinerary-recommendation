# SIMULATED ANNEALING
import random
import copy
import numpy as np
from optimization_mctoptw_solomon_dataset_maut.base import BaseOptimizer

class SA(BaseOptimizer):
    def __init__(self,temperature = 15000,cooling_rate=0.99,stopping_temperature=0.0002,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.temperature = temperature #temperature of SA
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        
        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
    
    def set_model(self,nodes,depot,num_vehicle=3,init_solution=[],degree_waktu = 1,degree_tarif = 1,degree_profit = 1):
        super().set_model(nodes,depot,num_vehicle,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_profit = degree_profit)
        self.init_solution = init_solution

    def swap_operation(self,solution):
        sol = copy.deepcopy(solution)
        pos1,pos2 = random.sample(range(len(sol)),2)
        sol[pos1],sol[pos2] = sol[pos2],sol[pos1]
        return sol
    
    def construct_solution(self):
        solution = random.sample(self.nodes,len(self.nodes)) if len(self.init_solution) <= 0 else copy.deepcopy(sum(self.init_solution,[])+self.get_rest_nodes_from_solution(self.init_solution))
        fitness = self.fitness(self.split_itinerary(solution))
        while self.temperature >= self.stopping_temperature:
            #generate new solution
            new_solution = self.swap_operation(solution)
            new_fitness = self.fitness(self.split_itinerary(new_solution))
            
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
            
        return self.split_itinerary(solution),self.fitness(self.split_itinerary(solution))
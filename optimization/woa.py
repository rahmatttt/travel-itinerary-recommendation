# WHALE OPTIMIZATION ALGORITHM 
import random
import numpy as np
from optimization.base import BaseOptimizer

class WOA_VRP(BaseOptimizer):
    def __init__(self,population = 50,b = 1,max_iter = 1000,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state,seed_numpy = True)

        # parameter setting
        self.population = population #population size
        self.b = b #logarithmic spiral
        self.a = 2 #a value set to 2 and will gradually decrease throughout the iterations until it reaches 0
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
    
    def generate_population(self):
        population_ls = [np.array([random.uniform(0,10) for i in range(len(self.tour))]) for j in range(self.population)]
        return population_ls
    
    def encircling_prey(self,best_solution,A,D):
        return best_solution - (A*D)
    
    def bubble_net_attacking(self,best_solution,A,D):
        l = random.uniform(-1,1)
        return D * np.exp(self.b*l) * np.cos(2*np.pi*l) + best_solution
    
    def search_for_prey(self,other_solution,A,D):
        return self.encircling_prey(other_solution,A,D)
    
    def construct_solution(self):
        #generate population
        population_ls = self.generate_population()
        population_ls = sorted(population_ls,
                               key=lambda x: self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(x)))),
                              reverse=True)
        
        best_solution = population_ls[0]
        best_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))))
        
        idem_counter = 0
        
        for i in range(self.max_iter):
            new_population_ls = []
            for solution in population_ls:
                r = random.uniform(0,1)
                A = (2*self.a*r)-self.a
                C = 2*r
                p = random.uniform(0,1)
                if p < 0.5:
                    if np.abs(A) < 1:
                        D = np.abs((C*best_solution)-solution)
                        new_solution = self.encircling_prey(best_solution,A,D)
                    else:
                        other_solution = population_ls[random.randint(0,self.population-1)]
                        D = np.abs((C*other_solution)-solution)
                        new_solution = self.search_for_prey(other_solution,A,D)
                else:
                    D = np.abs((C*best_solution)-solution)
                    new_solution = self.bubble_net_attacking(best_solution,A,D)
                new_solution[new_solution<0] = 0
                new_solution[new_solution>10] = 10
                new_population_ls.append(new_solution)
            
            population_ls = population_ls + new_population_ls
            
            population_ls = sorted(population_ls,
                               key=lambda x: self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(x)))),
                              reverse=True)
            
            population_ls = population_ls[:self.population]
            
            best_found_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(population_ls[0]))))
            
            self.a = 2*(1-((i+1)/self.max_iter))
            
            if best_found_fitness > best_fitness:
                best_solution = population_ls[0]
                best_fitness = best_found_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))),best_fitness
            
        return self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))),best_fitness
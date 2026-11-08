# GENETIC ALGORITHM
import random
import copy
import numpy as np
from optimization_c_r_rc_100_100_dataset.base import BaseOptimizer

class GA(BaseOptimizer):
    def __init__(self,population_size = 50,crossover_rate=0.8,mutation_rate=0.2,max_iter = 300,max_idem = 15,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.population_size = population_size #max size of population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
            
    def partially_mapped_crossover(self,parent1,parent2): #PMX
        start,end = sorted(random.sample(range(len(parent1)), 2))
        child1 = [None] * len(parent1)
        child2 = copy.deepcopy(child1)
        mapping1 = {parent1[i]["id"]:parent2[i] for i in range(start,end+1)}
        mapping2 = {parent2[i]["id"]:parent1[i] for i in range(start,end+1)}

        for i in range(len(parent1)):
            if start <= i <= end:
                child1[i],child2[i] = parent1[i],parent2[i] 
            else:
                gene1,gene2 = parent2[i],parent1[i]
                while gene1["id"] in mapping1:
                    gene1 = mapping1[gene1["id"]]
                while gene2["id"] in mapping2:
                    gene2 = mapping2[gene2["id"]]
                child1[i],child2[i] = gene1,gene2

        return child1,child2
    
    def swap_mutation(self,individual):
        individu = copy.deepcopy(individual)
        pos1,pos2 = random.sample(range(len(individu)),2)
        individu[pos1],individu[pos2] = individu[pos2],individu[pos1]
        return individu
    
    def construct_solution(self):
        solution = []
        fitness = 0
        
        idem_counter = 0
        population = [random.sample(self.nodes,len(self.nodes)) for i in range(self.population_size)]
        for i in range(self.max_iter):
            
            #crossover and mutation
            offspring = []
            for ind in range(0,self.population_size,2):
                parent1,parent2 = population[ind],population[ind+1]
                
                #crossover
                if random.uniform(0,1) < self.crossover_rate:
                    child1,child2 = self.partially_mapped_crossover(parent1,parent2)
                else:
                    child1,child2 = copy.deepcopy(parent1),copy.deepcopy(parent2)
                
                #mutation
                if random.uniform(0,1) < self.mutation_rate:
                    child1 = self.swap_mutation(child1)
                if random.uniform(0,1) < self.mutation_rate:
                    child2 = self.swap_mutation(child2)
                
                offspring.append(child1)
                offspring.append(child2)
            
            #selection roulette wheel
            all_individuals = population + offspring
            fitness_values = [self.fitness(self.split_itinerary(individual)) for individual in all_individuals]
                                    
            population = random.choices(all_individuals,weights=np.array(fitness_values),k=self.population_size)
            
            best_solution = self.split_itinerary(max(population,key = lambda x:self.fitness(self.split_itinerary(x))))
            best_fitness = self.fitness(best_solution)
            
            if fitness < best_fitness:
                fitness = best_fitness
                solution = best_solution
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return solution,fitness
        
        return solution,fitness
                
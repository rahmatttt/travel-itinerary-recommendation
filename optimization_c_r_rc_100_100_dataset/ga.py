# GENETIC ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class GA(object):
    def __init__(self,population_size = 50,crossover_rate=0.8,mutation_rate=0.2,max_iter = 300,max_idem = 15,random_state=None):
        # parameter setting
        self.population_size = population_size #max size of population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # data model setting
        self.nodes = None #node yang dipilih oleh user untuk dikunjungi
        self.depot = None #depot: starting and end point
        self.max_travel_time = None
        self.num_vehicle = None
    
        #set random seed
        if random_state != None:
            random.seed(random_state)
            
    def set_model(self,nodes,depot,num_vehicle=3):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]
    
    def fitness(self,solution):
        return sum([sol["S"] for sol in sum(solution,[])])
    
    def euclidean(self,node1,node2):
        return np.sqrt(((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2))
    
    def split_itinerary(self,solution):
        routes = []
        
        vehicle = 1
        tabu_nodes = []
        
        while vehicle <= self.num_vehicle:
            current_loc = self.depot
            current_time = self.depot["O"]
            route = []
            next_node_candidates = [node for node in solution if node["id"] not in tabu_nodes]
            for next_node in next_node_candidates:
                travel_time = self.euclidean(current_loc,next_node)
                arrival_time = current_time + travel_time
                finish_time = max(arrival_time,next_node["O"])+next_node["d"]
                return_to_depot_time = self.euclidean(next_node,self.depot)
                if (arrival_time <= next_node["C"]) and (finish_time + return_to_depot_time <= self.max_travel_time):
                    route.append(next_node)
                    tabu_nodes.append(next_node["id"])
                    current_loc = next_node
                    current_time = finish_time
                else:
                    continue
            
            if len(route) > 0:
                routes.append(route)
            
            if len(tabu_nodes) == len(self.nodes):
                break
            
            vehicle += 1
        
        return routes
    
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
                
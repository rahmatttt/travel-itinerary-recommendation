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

        #degree of interest (DOI for MAUT) setting
        self.degree_waktu = 1
        self.degree_tarif = 1
        self.degree_profit = 1
        self.degree_poi = 1
        self.degree_poi_penalty = 1
        self.degree_time_penalty = 1
        
        #scaler setting
        self.min_profit = None
        self.max_profit = None
        self.min_tarif = None
        self.max_tarif = None
        self.min_waktu = None
        self.max_waktu = None
        self.min_poi = None
        self.max_poi = None
        self.min_poi_penalty = None
        self.max_poi_penalty = None
        self.min_time_penalty = None
        self.max_time_penalty = None
    
        #set random seed
        if random_state != None:
            random.seed(random_state)
            
    def set_model(self,nodes,depot,num_vehicle=3,degree_waktu = 1,degree_tarif = 1,degree_profit = 1):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]

        self.degree_waktu = degree_waktu
        self.degree_tarif = degree_tarif
        self.degree_profit = degree_profit
        
        self.min_profit = min([node["S"] for node in self.nodes])
        self.max_profit = sum([node["S"] for node in self.nodes])
        self.min_tarif = min([node["b"] for node in self.nodes])
        self.max_tarif = sum([node["b"] for node in self.nodes])
        self.min_waktu = 0
        self.max_waktu = self.max_travel_time*self.num_vehicle
        self.min_poi = 0
        self.max_poi = len(self.nodes)
        self.min_poi_penalty = 0
        self.max_poi_penalty = len(self.nodes)
        self.min_time_penalty = 0
        self.max_time_penalty = max(self.euclidean(node,self.depot) for node in self.nodes) * self.num_vehicle
    
    def min_max_scaler(self,min_value,max_value,value):
        if max_value-min_value == 0:
            return 0
        else:
            return (value-min_value)/(max_value-min_value)

    def fitness(self,solution):
        sum_time = 0
        sum_profit = 0
        sum_tarif = 0
        sum_time_penalty = 0

        #loop
        for route in solution:
            route_time = 0
            route_profit = 0
            route_tarif = 0
            route_time_penalty = 0
            current_node = self.depot
            current_time = 0
            
            for node in route:
                travel_time = self.euclidean(current_node,node)
                arrival_time = current_time + travel_time
                finish_time = max(arrival_time,node["O"])+node["d"]
                current_time = finish_time
                current_node = node

                route_time = finish_time
                route_profit += node["S"]
                route_tarif += node["b"]

            return_to_depot_time = self.euclidean(current_node,self.depot)
            route_time += return_to_depot_time

            route_time_penalty = max(0,route_time - self.max_travel_time)

            sum_time += route_time
            sum_profit += route_profit
            sum_tarif += route_tarif
            sum_time_penalty += route_time_penalty

        num_poi = len(sum(solution,[]))
        num_poi_penalty = self.max_poi - num_poi

        score_time = 1-self.min_max_scaler(self.min_waktu,self.max_waktu,sum_time)*self.degree_waktu
        score_profit = self.min_max_scaler(self.min_profit,self.max_profit,sum_profit)*self.degree_profit
        score_tarif = 1-self.min_max_scaler(self.min_tarif,self.max_tarif,sum_tarif)*self.degree_tarif
        score_poi = self.min_max_scaler(self.min_poi,self.max_poi,num_poi)*self.degree_poi
        score_poi_penalty = 1-self.min_max_scaler(self.min_poi_penalty,self.max_poi_penalty,num_poi_penalty)*self.degree_poi_penalty
        score_time_penalty = 1-self.min_max_scaler(self.min_time_penalty,self.max_time_penalty,sum_time_penalty)*self.degree_time_penalty

        degree_profit = self.degree_profit
        degree_tarif = self.degree_tarif
        degree_waktu = self.degree_waktu
        degree_poi = self.degree_poi
        degree_poi_penalty = self.degree_poi_penalty
        degree_time_penalty = self.degree_time_penalty

        pembilang = score_profit+score_tarif+score_time+score_poi+score_poi_penalty+score_time_penalty
        penyebut = degree_profit+degree_tarif+degree_waktu+degree_poi+degree_poi_penalty+degree_time_penalty
        maut = pembilang/penyebut

        return maut
    
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
                if (arrival_time <= next_node["C"]) and (finish_time <= self.max_travel_time):
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
                
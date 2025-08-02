# GENETIC ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class SA(object):
    def __init__(self,temperature = 15000,cooling_rate=0.99,stopping_temperature=0.0002,random_state=None):
        # parameter setting
        self.temperature = temperature #temperature of SA
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        
        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
        
        # data model setting
        self.nodes = None #node yang dipilih oleh user untuk dikunjungi
        self.depot = None #depot: starting and end point
        self.max_travel_time = None
        self.num_vehicle = None
    
        #set random seed
        if random_state != None:
            random.seed(random_state)
    
    def set_model(self,nodes,depot,num_vehicle=3,init_solution=[]):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]
        self.init_solution = init_solution
    
    def fitness(self,solution):
        return sum([sol["S"] for sol in sum(solution,[])])
    
    def fitness_between_two_nodes(self,current_node,next_node):
        return current_node["S"]+next_node["S"]
    
    def euclidean(self,solution1,solution2):
        return np.sqrt(((solution1["x"] - solution2["x"])**2 + (solution1["y"] - solution2["y"])**2))
    
    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time + return_to_depot_time <= self.max_travel_time):
            return True
        else:
            return False
    
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
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node["id"] for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.nodes if node["id"] not in solution_id]
        return rest_nodes
    
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
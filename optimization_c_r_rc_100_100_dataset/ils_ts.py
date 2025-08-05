#HYBRID ITERATED LOCAL SEARCH - SIMULATED ANNEALING
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class ILS_TS(object):
    def __init__(self,stepsize = 0.1,strength=0.5,max_tabu_size = 10,num_neighbor = 10,max_iter = 100,max_idem = 20,random_state=None):
        # parameter setting
        self.stepsize = stepsize #step size in local search process
        self.strength = strength #perturbation strength (to escape the local optimum)
        self.max_tabu_size = max_tabu_size #maximum size of tabu list
        self.num_neighbor = num_neighbor #the number of neighbor in neighborhood for new solution candidates in TS process
        self.max_iter = max_iter #max iteration
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
    
    def fitness_between_two_nodes(self,current_node,next_node):
        return current_node["S"]+next_node["S"]
    
    def euclidean(self,node1,node2):
        return np.sqrt(((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2))
    
    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time + return_to_depot_time <= self.max_travel_time):
            return True
        else:
            return False
    
    def solution_list_of_float_to_nodes(self,solutions):
        index_order = np.flip(np.argsort(solutions))
        return list(np.array(self.nodes)[index_order])
    
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
    
    def generate_init_solution(self):
        return np.random.uniform(-5.12,5.12,size=len(self.nodes))
    
    def local_search(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(solution)))
        
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
                        neighbor_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor)))
                        tabu_list.append(addition)
                else:
                    neighbor = solution + addition
                    neighbor_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor)))
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
            
    def perturbation(self,solution_nodes):
        return solution_nodes + np.random.uniform(-self.strength,self.strength,size=len(solution_nodes)) 
    
    def construct_solution(self):
        idem_counter = 0
        
        best_solution,best_fitness = self.local_search(self.generate_init_solution())
        
        for i in range(self.max_iter):
            #perturbation
            candidate = self.perturbation(best_solution)
            
            #local search
            candidate,candidate_fitness = self.local_search(candidate)
            
            if candidate_fitness > best_fitness:
                best_solution = copy.deepcopy(candidate)
                best_fitness = candidate_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))
                    return best_solution,best_fitness
        
        best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))
        return best_solution,best_fitness
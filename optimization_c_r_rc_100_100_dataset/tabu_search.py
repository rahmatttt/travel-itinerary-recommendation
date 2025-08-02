# GENETIC ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class TS(object):
    def __init__(self,max_tabu_size = 20,max_iter = 7500,max_idem = 7500,random_state=None):
        # parameter setting
        self.max_tabu_size = max_tabu_size #max size of tabu list
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
        self.rest_nodes = [] #1D list of nodes, [node1,node2,node4,node5,....]
        
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
        
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = copy.deepcopy(init_solution)
        else:
            self.init_solution = self.generate_init_solution()
        self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)
    
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
    
    def generate_init_solution(self):
        solution = []
        vehicle = 1
        tabu_nodes = []
        while vehicle<=self.num_vehicle:
            current_node = self.depot
            current_time = self.depot["O"]
            route = []
            
            for pos in range(len(self.nodes)-1):
                #recheck next node candidates (perlu dicek jam sampainya apakah melebihi max time)
                next_node_candidates = [node for node in self.nodes if self.next_node_check(current_node,node,current_time)==True and node["id"] not in tabu_nodes]
                if len(next_node_candidates)>0:
                    max_pos = np.argmax([self.fitness_between_two_nodes(current_node,next_node) for next_node in next_node_candidates])
                    next_node = next_node_candidates[max_pos]
                    route.append(next_node)
                    tabu_nodes.append(next_node["id"])
                    
                    travel_time = self.euclidean(current_node,next_node)
                    arrival_time = current_time + travel_time
                    finish_time = max(arrival_time,next_node["O"])+next_node["d"]
                    current_node = next_node
                    current_time = finish_time
                else:
                    break
            
            if len(route)>0:
                solution.append(route)
            
            if len(tabu_nodes) == len(self.nodes):
                break
            
            vehicle += 1
        
        return solution
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node["id"] for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.nodes if node["id"] not in solution_id]
        return rest_nodes
    
    def construct_solution(self):
        solution = copy.deepcopy(self.init_solution) 
        fitness = self.fitness(solution)
        sol_list = sum(solution,[]) + self.rest_nodes
        
        idem_counter = 0
        tabu_list = []
        
        for i in range(self.max_iter):
            first_pos,second_pos = random.sample(range(len(sol_list)),2)
            first_node = sol_list[first_pos]
            second_node = sol_list[second_pos]
            
            if (first_node["id"],second_node["id"]) in tabu_list or (second_node["id"],first_node["id"]) in tabu_list:
                idem_counter += 1
            else:
                #swap
                sol_list[first_pos],sol_list[second_pos] = sol_list[second_pos],sol_list[first_pos]
                new_sol = self.split_itinerary(sol_list)
                new_fitness = self.fitness(new_sol)
                if new_fitness > fitness:
                    solution = copy.deepcopy(new_sol)
                    fitness = new_fitness
                    self.rest_nodes = self.get_rest_nodes_from_solution(solution)
                    sol_list = sum(solution,[])+self.rest_nodes
                    tabu_list.append((first_node["id"],second_node["id"]))
                    idem_counter = 0
                else:
                    idem_counter += 1
            if idem_counter%5 == 0 and idem_counter>0:
                tabu_list = []
            if len(tabu_list) > self.max_tabu_size:
                tabu_list = tabu_list[1:]
            if idem_counter > self.max_idem:
                return solution,fitness
        
        return solution,fitness
# GENETIC ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class ACS(object):
    def __init__(self,alpha_t = 2,beta = 3,q0 = 0.1,init_pheromone = 0.1,rho = 0.9,alpha = 0.1,num_ant = 30,max_iter = 200,max_idem=30,random_state=None):
        #parameter setting
        self.alpha_t = alpha_t #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in ACS transition rule
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate local pheromone update
        self.alpha = alpha #evaporation rate global pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration ACS
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
        
        # data model setting
        self.nodes = None #node yang dipilih oleh user untuk dikunjungi
        self.depot = None #depot: starting and end point
        self.max_travel_time = None
        self.num_vehicle = None
        self.pheromone_matrix = None
    
        #set random seed
        if random_state != None:
            random.seed(random_state)
    
    def set_model(self,nodes,depot,num_vehicle=3):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]
        self.pheromone_matrix = self.create_pheromone_matrix(nodes,depot)
    
    def create_pheromone_matrix(self,nodes,depot):
        matrix = {}
        for source in [depot]+nodes:
            matrix[source["id"]] = {}
            for dest in [depot]+nodes:
                if dest["id"] != source["id"]:
                    matrix[source["id"]][dest["id"]] = self.init_pheromone
        
        return matrix
    
    def fitness(self,solution):
        return sum([sol["S"] for sol in sum(solution,[])])
    
    def fitness_between_two_nodes(self,current_node,next_node):
        return current_node["S"]+next_node["S"]
    
    def euclidean(self,solution1,solution2):
        return np.sqrt(((solution1["x"] - solution2["x"])**2 + (solution1["y"] - solution2["y"])**2))
    
    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix):
        max_pos = np.argmax([local_pheromone_matrix[current_node["id"]][next_node["id"]]*(self.fitness_between_two_nodes(current_node,next_node)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node
    
    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node["id"]][next_node["id"]]**self.alpha_t
            heuristic_val = self.fitness_between_two_nodes(current_node,next_node)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val
        
        #probability
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node["id"]][next_node["id"]]**self.alpha_t
            heuristic_val = self.fitness_between_two_nodes(current_node,next_node)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            node_prob = 0.0001 if node_prob == 0 else node_prob
            next_node_prob.append(node_prob)
        
        next_node = random.choices(next_node_candidates,next_node_prob,k=1)
        return next_node[0]
    
    def transition_rule(self,current_node,next_node_candidates,local_pheromone_matrix):
        q = random.uniform(0,1)
        if q <= self.q0: #exploitation
            next_node = self.exploitation(current_node,next_node_candidates,local_pheromone_matrix)
        else: #exploration
            next_node = self.exploration(current_node,next_node_candidates,local_pheromone_matrix)
        return next_node
    
    def local_pheromone_update(self,solutions,fitness,local_pheromone_matrix):
        for route in solutions:
            nodes = [self.depot["id"]] + [node["id"] for node in route] + [self.depot["id"]]
            i = 1
            while i < len(nodes):
                pheromone = local_pheromone_matrix[nodes[i-1]][nodes[i]]
                pheromone = ((1-self.rho)*pheromone)+(self.rho*fitness)
                local_pheromone_matrix[nodes[i-1]][nodes[i]] = pheromone
                i += 1
        return local_pheromone_matrix
    
    def global_pheromone_update(self,best_solutions,best_fitness):
        solutions = copy.deepcopy(best_solutions)
        solution_index = []
        for route in solutions:
            node_ids = [node["id"] for node in route]
            route_solution = [self.depot["id"]] + node_ids + [self.depot["id"]]
            route_solution = [(route_solution[i-1],route_solution[i]) for i in range(1,len(node_ids))]
            solution_index = solution_index + route_solution
        
        for node in self.pheromone_matrix:
            for next_node in self.pheromone_matrix[node]:
                pheromone = self.pheromone_matrix[node][next_node]
                if (node,next_node) in solution_index:
                    pheromone = ((1-self.alpha)*pheromone)+(self.alpha*best_fitness)
                else:
                    pheromone = ((1-self.alpha)*pheromone)+0
                self.pheromone_matrix[node][next_node] = pheromone
    
    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time + return_to_depot_time <= self.max_travel_time):
            return True
        else:
            return False
    
    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            best_found_solution = None
            best_found_fitness = 0
            local_pheromone_matrix = copy.deepcopy(self.pheromone_matrix)
            for ant in range(self.num_ant): #step
                ant_solution = []
                vehicle = 1
                tabu_nodes = []
                while vehicle<= self.num_vehicle:
                    current_node = self.depot
                    current_time = self.depot["O"]
                    ant_vehicle_solution = []
                    
                    for pos in range(len(self.nodes)+1):
                        next_node_candidates = [node for node in self.nodes if self.next_node_check(current_node,node,current_time) == True and node["id"] not in tabu_nodes]
                        
                        if len(next_node_candidates) > 0:
                            next_node = self.transition_rule(current_node,next_node_candidates,local_pheromone_matrix)
                            
                            travel_time = self.euclidean(current_node,next_node)
                            arrival_time = current_time + travel_time
                            finish_time = max(arrival_time,next_node["O"])+next_node["d"]
                            
                            current_time = finish_time
                            current_node = next_node
                            ant_vehicle_solution.append(current_node)
                            
                            tabu_nodes.append(current_node["id"])
                        else:
                            break
                    
                    if len(ant_vehicle_solution)>0:
                        ant_solution.append(ant_vehicle_solution)
                    
                    if len(tabu_nodes) == len(self.nodes):
                        break
                    
                    vehicle += 1
                
                fitness = self.fitness(ant_solution)
                if fitness > best_found_fitness:
                    best_found_fitness = fitness
                    best_found_solution = copy.deepcopy(ant_solution)
                
                #local pheromone update
                local_pheromone_matrix = self.local_pheromone_update(ant_solution,fitness,local_pheromone_matrix)
                
            #global pheromone update
            self.global_pheromone_update(best_found_solution,best_found_fitness)
            
            if best_found_fitness > best_fitness:
                best_fitness = best_found_fitness
                best_solution = copy.deepcopy(best_found_solution)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness
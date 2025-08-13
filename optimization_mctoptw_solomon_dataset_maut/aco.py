# ANT COLONY OPTIMIZATION
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class ACO(object):
    def __init__(self,alpha = 1,beta = 4,q0 = 0.6,init_pheromone = 0.3,rho = 0.1,num_ant = 35,max_iter = 100,max_idem=50,random_state=None):
        #parameter setting
        self.alpha = alpha #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in ACO transition rule
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration ACO
        self.max_idem = max_idem
        
        # data model setting
        self.nodes = None #node yang dipilih oleh user untuk dikunjungi
        self.depot = None #depot: starting and end point
        self.max_travel_time = None
        self.num_vehicle = None
        self.pheromone_matrix = None

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
        self.pheromone_matrix = self.create_pheromone_matrix(nodes,depot)

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
    
    def create_pheromone_matrix(self,nodes,depot):
        matrix = {}
        for source in [depot]+nodes:
            matrix[source["id"]] = {}
            for dest in [depot]+nodes:
                if dest["id"] != source["id"]:
                    matrix[source["id"]][dest["id"]] = {"pheromone":self.init_pheromone}
        
        return matrix
    
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
    
    def fitness_between_two_nodes(self,current_node,next_node):
        score_profit = self.degree_profit * self.min_max_scaler(self.min_profit,self.max_profit,next_node["S"])
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node["b"]))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.euclidean(current_node,next_node)))
        maut = (score_profit+score_tarif+score_waktu)/(self.degree_profit+self.degree_tarif+self.degree_waktu)
        return maut
    
    def euclidean(self,node1,node2):
        return np.sqrt(((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2))
    
    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix):
        max_pos = np.argmax([(local_pheromone_matrix[current_node["id"]][next_node["id"]]["pheromone"]**self.alpha)*(self.fitness_between_two_nodes(current_node,next_node)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node
    
    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node["id"]][next_node["id"]]["pheromone"]**self.alpha
            heuristic_val = self.fitness_between_two_nodes(current_node,next_node)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val
        
        #probability
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node["id"]][next_node["id"]]["pheromone"]**self.alpha
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
    
    def pheromone_update(self,pheromone_matrix):
        for node in self.pheromone_matrix:
            for next_node in self.pheromone_matrix[node]:
                pheromone = self.pheromone_matrix[node][next_node]["pheromone"]
                sum_delta = sum(pheromone_matrix[node][next_node]["delta"])
                pheromone = ((1-self.rho)*pheromone)+sum_delta
                self.pheromone_matrix[node][next_node]["pheromone"] = pheromone
    
    def init_delta_to_pheromone_matrix(self,pheromone_matrix):
        for i in pheromone_matrix:
            for j in pheromone_matrix[i]:
                pheromone_matrix[i][j]["delta"] = []
        return pheromone_matrix
    
    def add_delta_to_pheromone_matrix(self,vehicle_solution,pheromone_matrix,fitness):
        nodes = [self.depot["id"]] + [node["id"] for node in vehicle_solution] + [self.depot["id"]]
        node_edges = [(nodes[idx-1],nodes[idx]) for idx in range(1,len(nodes))]
        for i,j in node_edges:
            pheromone_matrix[i][j]["delta"].append(fitness)
        
        return pheromone_matrix
    
    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time <= self.max_travel_time):
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
            local_pheromone_matrix = self.init_delta_to_pheromone_matrix(local_pheromone_matrix)
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
                
                #add delta
                for vehicle_sol in ant_solution:
                    local_pheromone_matrix = self.add_delta_to_pheromone_matrix(vehicle_sol,local_pheromone_matrix,fitness)
                
            #pheromone update
            self.pheromone_update(local_pheromone_matrix)
            
            if best_found_fitness > best_fitness:
                best_fitness = best_found_fitness
                best_solution = copy.deepcopy(best_found_solution)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness
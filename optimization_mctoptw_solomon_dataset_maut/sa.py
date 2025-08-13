# SIMULATED ANNEALING
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
    
    def set_model(self,nodes,depot,num_vehicle=3,init_solution=[],degree_waktu = 1,degree_tarif = 1,degree_profit = 1):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]
        self.init_solution = init_solution

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
    
    def fitness_between_two_nodes(self,current_node,next_node):
        score_profit = self.degree_profit * self.min_max_scaler(self.min_profit,self.max_profit,next_node["S"])
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node["b"]))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.euclidean(current_node,next_node)))
        maut = (score_profit+score_tarif+score_waktu)/(self.degree_profit+self.degree_tarif+self.degree_waktu)
        return maut
    
    def euclidean(self,node1,node2):
        return np.sqrt(((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2))
    
    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time <= self.max_travel_time):
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
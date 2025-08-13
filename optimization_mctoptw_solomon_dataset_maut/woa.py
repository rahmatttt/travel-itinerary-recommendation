# WHALE OPTIMIZATION ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class WOA(object):
    def __init__(self,population = 50,b = 1,max_iter = 1000,max_idem = 20,random_state=None):
        #parameter setting
        self.population = population #population size
        self.b = b #logarithmic spiral
        self.a = 2 #a value set to 2 and will gradually decrease throughout the iterations until it reaches 0
        self.max_iter = max_iter #max iteration
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
    
    def generate_population(self):
        population_ls = [np.array([random.uniform(0,10) for i in range(len(self.nodes))]) for j in range(self.population)]
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
                               key=lambda x: self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(x))),
                              reverse=True)
        
        best_solution = population_ls[0]
        best_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution)))
        
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
                               key=lambda x: self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(x))),
                              reverse=True)
            
            population_ls = population_ls[:self.population]
            
            best_found_fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(population_ls[0])))
            
            self.a = 2*(1-((i+1)/self.max_iter))
            
            if best_found_fitness > best_fitness:
                best_solution = population_ls[0]
                best_fitness = best_found_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution)),best_fitness
            
        return self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution)),best_fitness
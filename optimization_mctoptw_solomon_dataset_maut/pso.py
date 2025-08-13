# PARTICLE SWARM OPTIMIZATION
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class PSO(object):
    def __init__(self,n = 30, w = 0.7,c1 = 1.5,c2 = 1.5,max_iter = 200,max_idem = 20,random_state=None):
        # parameter setting
        self.n = n #number of particle(population)
        self.w = w #inertia weight
        self.c1 = c1 #cognitive coefficient: Strength of pull toward personal best (individual learning)
        self.c2 = c2 #social coefficient: Strength of pull toward global best (social learning)
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
        #the individu (particle) is a list consists of a number [0,1]
        particle_ls = [np.array([random.random() for i in range(len(self.nodes))]) for j in range(self.n)]
        particle_dict = []
        for particle in particle_ls:
            solution = particle
            fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(particle)))
            velocity = np.random.uniform(-1,1,size=len(particle))
            particle_dict.append({
                "solution":solution,
                "velocity":velocity,
                "fitness":fitness,
                "best_fitness":fitness,
                "best_solution":solution
            })
            
        return particle_dict
    
    def update_velocity(self,particle_dict,best_solution):
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        cognitive = self.c1 * r1 * (particle_dict["best_solution"] - particle_dict["solution"])
        social = self.c2 * r2 * (best_solution - particle_dict["solution"])
        particle_dict['velocity'] = self.w * particle_dict['velocity'] + cognitive + social
        return particle_dict
    
    def update_solution(self,particle_dict):
        particle_dict['solution'] = particle_dict['solution']+particle_dict['velocity']
        particle_dict['fitness'] = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(particle_dict['solution'])))
        if particle_dict['fitness'] > particle_dict['best_fitness']:
            particle_dict['best_fitness'] = particle_dict['fitness']
            particle_dict['best_solution'] = particle_dict['solution']
        return particle_dict
    
    def construct_solution(self):
        idem_counter = 0
        
        particle_dict = self.generate_population()
        
        best_particle = sorted(particle_dict,key=lambda x: x['fitness'], reverse=True)[0]
        best_found_particle = copy.deepcopy(best_particle)
        
        for i in range(self.max_iter):
            for j in range(len(particle_dict)):
                #update velocity
                particle_dict[j] = self.update_velocity(particle_dict[j],best_particle['solution'])
                
                #update solution/position
                particle_dict[j] = self.update_solution(particle_dict[j])
                
                if particle_dict[j]['fitness'] > best_found_particle['fitness']:
                    best_found_particle = copy.deepcopy(particle_dict[j])
            
            if best_found_particle['fitness'] > best_particle['fitness']:
                best_particle = copy.deepcopy(best_found_particle)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_particle['best_solution']))
                    best_fitness = best_particle['best_fitness']
                    return best_solution,best_fitness
        
        best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_particle['best_solution']))
        best_fitness = best_particle['best_fitness']
        return best_solution,best_fitness
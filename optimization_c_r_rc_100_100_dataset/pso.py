# GENETIC ALGORITHM
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
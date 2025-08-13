# BRAINSTORM OPTIMIZATION
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class BSO(object):
    def __init__(self,p1 = 0.7,p2 = 0.7,p3 = 0.9,p4 = 0.3,max_iter = 100,max_idem = 15,two_opt_method="best",random_state=None):
        # parameter setting
        self.p1 = p1 #less than: 2-opt, more than: 2-interchange
        self.p2 = p2 #less than: center, more than: random cluster (2-opt)
        self.p3 = p3 #less than: interchange a cluster and rest nodes, more than: interchange between clusters
        self.p4 = p4 #less than: center, more than: random cluster
        
        self.max_iter = max_iter #max iteration of BSO
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
        
        self.two_opt_method = two_opt_method

        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
        self.rest_nodes = [] #1D list of nodes, [node1,node2,node4,node5,....]

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
        
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        else:
            self.init_solution = self.generate_init_solution()
        self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)
        
        if len(self.rest_nodes) == 0:
            self.p3 = 0 #no interchange with rest nodes
            
        if len(self.init_solution)<=1 and len(self.rest_nodes)>0:
            self.p3 = 1 #no interchange between clusters
        elif len(self.init_solution)<=1 and len(self.rest_nodes)==0:
            self.p1 = 1 #no two interchange (only two opt)
    
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
    
    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time <= self.max_travel_time):
            return True
        else:
            return False
    
    def set_init_solution(self,init_solution):
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        
            self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)

            if len(self.rest_nodes) == 0:
                self.p3 = 0 #no interchange with rest nodes

            if len(self.init_solution)<=1 and len(self.rest_nodes)>0:
                self.p3 = 1 #no interchange between clusters
            elif len(self.init_solution)<=1 and len(self.rest_nodes)==0:
                self.p1 = 1 #no two interchange (only two opt)
    
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
    
    def generate_init_solution(self):
        solution = list(copy.deepcopy(self.nodes))
        random.shuffle(solution)
        
        return self.split_itinerary(solution)
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node["id"] for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.nodes if node["id"] not in solution_id]
        return rest_nodes
    
    def random_clustering(self,solution_nodes):
        random.shuffle(solution_nodes)
        cluster_a = solution_nodes[:len(solution_nodes)//2]
        cluster_b = solution_nodes[len(solution_nodes)//2:]
        return cluster_a,cluster_b
    
    def find_center_index(self,cluster_nodes):
        return np.argmax([self.fitness([i]) for i in cluster_nodes])
    
    def route_feasibility_check(self,route):
        #input : list of nodes route (it was changed by some process before)
        #output: if any invalid time then return False
        solution = copy.deepcopy(route)
        
        current_node = self.depot
        current_time = self.depot["O"]
        
        for i in range(len(solution)):
            if self.next_node_check(current_node,solution[i],current_time):
                next_node = copy.deepcopy(solution[i])
                travel_time = self.euclidean(current_node,next_node)
                arrival_time = current_time + travel_time
                finish_time = max(arrival_time,next_node["O"])+next_node["d"]
                
                current_node = solution[i]
                current_time = finish_time
            else:
                return False #invalid
        return True
    
    def two_opt(self,route):
        solution = [self.depot] + route + [self.depot]
        fitness = self.fitness([solution[1:-1]])
        n = len(solution)
        
        if self.two_opt_method == "first": #first improvement
            improved = False
            for i in range(1,n-2):
                for j in range(i+2,n-1):
                    if j-i == 1:
                        continue
                    temp_solution = copy.deepcopy(solution)
                    temp_solution[i+1:j+1] = reversed(temp_solution[i+1:j+1])
                    status = self.route_feasibility_check(temp_solution[1:-1])
                    if status:
                        temp_fitness = self.fitness([temp_solution[1:-1]])
                        if temp_fitness > fitness:
                            solution = copy.deepcopy(temp_solution)
                            fitness = temp_fitness
                            improved = True
                            return solution[1:-1] #first improvement
        else: #best improvement
            improved = True
            while improved:
                improved = False
                for i in range(1,n-2):
                    for j in range(i+2,n-1):
                        if j-i == 1:
                            continue
                        temp_solution = copy.deepcopy(solution)
                        temp_solution[i+1:j+1] = reversed(temp_solution[i+1:j+1])
                        status = self.route_feasibility_check(temp_solution[1:-1])
                        if status:
                            temp_fitness = self.fitness([temp_solution[1:-1]])
                            if temp_fitness > fitness:
                                solution = copy.deepcopy(temp_solution)
                                fitness = temp_fitness
                                improved = True

        return solution[1:-1] #return best or return unchanged
    
    def two_interchange(self,route1,route2, rest_nodes = False):
        operator = [(0,1),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        solution1 = copy.deepcopy(route1)
        solution2 = copy.deepcopy(route2) #rest nodes if rest_nodes == True
        if rest_nodes == False:
            fitness = self.fitness([solution1,solution2])
        else:
            fitness = self.fitness([solution1])
        
        for op in operator:
            if op[0] <= 1 or len(solution1) == 1:
                sol1 = [[node] for node in solution1]
            else:
                sol1 = [[node[i-1],node[i]] for i in range(1,len(solution1))]
                    
            if op[1] <= 1 or len(solution2) == 1:
                sol2 = [[node] for node in solution2]
            else:
                sol2 = [[node[i-1],node[i]] for i in range(1,len(solution2))] #[[1,2],[2,3],[3,4],[4,5]]
            
            for i in range(len(sol1)):
                for j in range(len(sol2)):
                    temp1 = copy.deepcopy(sol1)
                    temp2 = copy.deepcopy(sol2)
                    
                    if op[0] == 0:
                        #shift sol2 to sol1
                        temp1.append(temp2.pop(j))
                        
                    elif op[1] == 0:
                        #shift sol1 to sol2
                        temp2.append(temp1.pop(i))
                    else:
                        #swap
                        temp1[i],temp2[j] = temp2[j],temp1[i]
                    
                    #flatten
                    temp1 = sum(temp1,[])
                    temp2 = sum(temp2,[])

                    # unify the node list if op > 1
                    if op[0]>1:
                        tabu_nodes = []
                        temp_nodes = []
                        for node in temp1:
                            if node["id"] not in tabu_nodes:
                                temp_nodes.append(node)
                                tabu_nodes.append(node["id"])
                        temp1 = copy.deepcopy(temp_nodes)
                        
                    if op[1]>1:
                        tabu_nodes = []
                        temp_nodes = []
                        for node in temp2:
                            if node._id not in tabu_nodes:
                                temp_nodes.append(node)
                                tabu_nodes.append(node["id"])
                        temp2 = copy.deepcopy(temp_nodes)
                    
                    #check feasibility
                    status1 = self.route_feasibility_check(temp1)
                    if rest_nodes == False:
                        status2 = self.route_feasibility_check(temp2)
                    else:
                        status2 = True
                    
                    if status1 and status2:
                        #count fitness
                        if rest_nodes == False and len(temp2) > 0 and len(temp1) > 0:
                            temp_fitness = self.fitness([temp1,temp2])
                        elif rest_nodes == False and len(temp2) > 0:
                            temp_fitness = self.fitness([temp2])
                        elif rest_nodes == False and len(temp1) > 0:
                            temp_fitness = self.fitness([temp1])
                        elif rest_nodes == True and len(temp1) > 0:
                            temp_fitness = self.fitness([temp1])
                        else:
                            temp_fitness = 0
                    
                        if temp_fitness > fitness:
                            return temp1,temp2 #first improvement
            
            return route1,route2 #if no improvement
    
    def construct_solution(self):
        fitness = 0
        solution = copy.deepcopy(self.init_solution)
        idem_counter = 0
        for i in range(self.max_iter):
            if len(solution) > 1:
                #clustering
                clusters = {1:{},2:{}}
                clusters[1]['list'],clusters[2]['list'] = self.random_clustering(solution)

                #find center index
                clusters[1]['center'] = self.find_center_index(clusters[1]['list'])
                clusters[2]['center'] = self.find_center_index(clusters[2]['list'])
            else:
                clusters = {1:{}}
                clusters[1]['list'] = copy.deepcopy(solution)
                clusters[1]['center'] = self.find_center_index(clusters[1]['list'])
            
            #randomization p1
            if random.uniform(0,1) < self.p1 or (len(clusters)==1 and len(self.rest_nodes)==0):
                #2-opt
                cluster_id = random.randint(1,2) if len(solution)>1 else 1
                
                # randomization p2
                if random.uniform(0,1) < self.p2:
                    #center
                    route_id = clusters[cluster_id]['center']
                else:
                    #random route
                    route_id = random.randint(0,len(clusters[cluster_id]['list'])-1)
                clusters[cluster_id]['list'][route_id] = self.two_opt(clusters[cluster_id]['list'][route_id])
            else:
                #2-interchange
                if random.uniform(0,1) < self.p3 or (len(clusters)==1 and len(self.node)>0):
                    #interchange a cluster with rest nodes
                    cluster_id = random.randint(1,2) if len(clusters)>1 else 1
                    
                    #randomization p4
                    if random.uniform(0,1) < self.p4:
                        route_id = clusters[cluster_id]['center']
                    else:
                        route_id = random.randint(0,len(clusters[cluster_id]['list'])-1)
                    clusters[cluster_id]['list'][route_id],self.rest_nodes = self.two_interchange(clusters[cluster_id]['list'][route_id],self.rest_nodes,rest_nodes=True)
                else:
                    #interchange between two clusters
                    #randomization p4
                    if random.uniform(0,1) < self.p4:
                        route_id_1 = clusters[1]['center']
                        route_id_2 = clusters[2]['center']
                    else:
                        route_id_1 = random.randint(0,len(clusters[1]['list'])-1)
                        route_id_2 = random.randint(0,len(clusters[2]['list'])-1)
                    clusters[1]['list'][route_id_1],clusters[2]['list'][route_id_2] = self.two_interchange(clusters[1]['list'][route_id_1],clusters[2]['list'][route_id_2],rest_nodes=False)
            
            #merge cluster
            solution = []
            for cluster_id in clusters:
                solution.extend(clusters[cluster_id]['list'])
            
            solution = [sol for sol in solution if len(sol)>0]
            new_fitness = self.fitness(solution)
            if new_fitness > fitness:
                fitness = new_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return solution,fitness
        
        return solution,fitness
                
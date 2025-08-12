# MODELING TO GENERATE ALTERNATIVE (MGA)
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class MGA(object):
    def __init__(self,population_size = 10,m=5,n=10,const=3,max_iter = 100,max_idem = 20,random_state=None):
        # parameter setting
        self.population_size = population_size #number of individual in the population
        self.m = m #number of candidates returned by the breed function
        self.n = n #number of alternative solutions (archive)
        self.const = const #suitable constant chosen in advance to allow an acceptable deviation from the obj function (to calculate target fitness)
        self.max_iter = max_iter #max iteration of tabu search
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
        
        self.operator_credit = {
            "displacement" : 1,
            "swap" : 1,
            "insertion" : 1,
            "reversal" : 1
        }
        
        self.archive = [] #the archived solution (S)
        self.reference,_ = self.reference_solution()
        self.target_fitness = self.reference["fitness"] - self.const*np.abs(self.reference["fitness"])
        
        self.archive = [self.reference] #the archived solution (S)
    
    def fitness(self,solution):
        return sum([sol["S"] for sol in sum(solution,[])])
    
    def euclidean(self,node1,node2):
        return np.sqrt(((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2))
    
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
    
    def generate_solution(self, init_solution = []):
        solution = self.split_itinerary(random.sample(self.nodes,len(self.nodes))) if len(init_solution) == 0 else self.split_itinerary(init_solution)
        rest_nodes = self.get_rest_nodes_from_solution(solution)
        fitness = self.fitness(solution)
        dissimilarity = self.compute_dissimilarity(solution) if len(self.archive)>0 else 0
        individual = {"solution" : solution,
                      "rest_nodes" : rest_nodes,
                      "fitness" : fitness,
                      "dissimilarity" : dissimilarity}
        return individual
    
    def generate_init_population(self):
        return [self.generate_solution() for i in range(self.population_size)]
    
    def select_mutation_operator(self):
        sum_credit = sum(self.operator_credit.values())
        weights = [credit/sum_credit for credit in self.operator_credit.values()]
        
        operator = random.choices(list(self.operator_credit.items()),weights=weights,k=1)[0][0]
        
        return operator
    
    def displacement(self,solution):
        sol = sum(solution["solution"],[])
        pos1,pos2 = random.sample(range(len(sol)),2)
        pos2_val = sol.pop(pos2)
        sol.insert(pos1,pos2_val)
        return self.generate_solution(sol)
    
    def swap(self,solution):
        sol = sum(solution["solution"],[])
        pos1,pos2 = random.sample(range(len(sol)),2)
        sol[pos1],sol[pos2] = sol[pos2],sol[pos1]
        return self.generate_solution(sol)
    
    def insertion(self,solution):
        if len(solution["rest_nodes"]) > 0:
            sol = sum(solution["solution"],[])
            pos1 = random.sample(range(len(sol)),1)[0]
            rest_pos = random.sample(range(len(solution["rest_nodes"])),1)[0]
            rest_val = solution["rest_nodes"].pop(rest_pos)
            sol.insert(pos1,rest_val)
            return self.generate_solution(sol)
        else:
            return solution
    
    def reversal(self,solution):
        sol = sum(solution["solution"],[])
        pos1,pos2 = random.sample(range(len(sol)),2)
        if pos2 > pos1:
            pos1,pos2 = pos2,pos1
        sol[pos1:pos2] = reversed(sol[pos1:pos2])
        return self.generate_solution(sol)
    
    def mutate(self,solution):
        operator = self.select_mutation_operator()
        if operator == "displacement":
            sol = self.displacement(solution)
        elif operator == "insertion":
            sol = self.insertion(solution)
        elif operator == "swap":
            sol = self.swap(solution)
        else:
            sol = self.reversal(solution)
        
        if sol["fitness"] > solution["fitness"]:
            self.operator_credit[operator]+=1
        
        return sol
    
    def breed(self,population):
        candidates = []
        while len(candidates) <= self.m:
            #roulette wheel
            sum_fitness = sum([individual["fitness"] for individual in population])
            weights = [individual["fitness"]/sum_fitness for individual in population]
            solution = random.choices(population,weights=weights,k=1)[0]
            solution = self.mutate(solution)
            candidates.append(solution)
        return candidates
    
    def survive(self,population):
        pass_target_population = [individual for individual in population if individual["fitness"]>= self.target_fitness]
        
        new_population = []
        
        if len(pass_target_population) > 0:
            pass_target_population = sorted(pass_target_population, key=lambda x: x["dissimilarity"], reverse = True)
            new_population = new_population + pass_target_population
        
        if len(pass_target_population) == 0 or len(new_population) < self.population_size:
            below_target_population = [individual for individual in population if individual["fitness"]< self.target_fitness]
            below_target_population = sorted(below_target_population, key=lambda x: x["fitness"], reverse = True)
            new_population = new_population + below_target_population
        
        new_population = new_population[:self.population_size]
        return new_population
    
    def reference_solution(self):
        population = self.generate_init_population()
        
        best_solution = []
        best_fitness = 0
        
        idem_counter = 0
        for i in range(self.max_iter):
            candidates = self.breed(population)
            population = population + candidates
            population = sorted(population, key = lambda x:x["fitness"], reverse = True)
            population = population[:self.population_size]
            
            solution = population[0]
            if solution["fitness"] > best_fitness:
                best_solution = solution
                best_fitness = solution["fitness"]
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness
    
    def jaccard_distance(self,a, b):
        return 1 - len(set(a) & set(b)) / len(set(a) | set(b))
    
    def compute_dissimilarity(self,solution):
        archive_id = [[i["id"] for i in sum(j["solution"],[])] for j in self.archive]
        sol_id = [i["id"] for i in sum(solution,[])]
        
        scores = [self.jaccard_distance(sol_id,archive_sol_id) for archive_sol_id in archive_id]
        
        return min(scores)
    
    def update_dissimilarity(self,population):
        for i in range(len(population)):
            population[i]["dissimilarity"] = compute_dissimilarity(population[i])
        
        return population
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node["id"] for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.nodes if node["id"] not in solution_id]
        return rest_nodes
    
    def construct_solution(self):
        population = self.generate_init_population()
        
        best_solution = self.reference
        best_fitness = best_solution["fitness"]
        
        while len(self.archive) < self.n:
            idem_counter = 0
            
            best_found_solution = []
            best_found_fitness = 0
            
            for i in range(self.max_iter):
                candidates = self.breed(population)
                population = population + candidates
                population = self.survive(population)
                population = sorted(population, key = lambda x:x["fitness"], reverse = True)
                population = population[:self.population_size]

                solution = population[0]
                if solution["fitness"] > best_found_fitness:
                    best_found_solution = solution
                    best_found_fitness = solution["fitness"]
                    idem_counter = 0
                else:
                    idem_counter += 1
                    if idem_counter > self.max_idem:
                        break
            
            self.archive.append(best_found_solution)
        
        self.archive = sorted(self.archive, key = lambda x:x["fitness"], reverse=True)
        best_solution = self.archive[0]["solution"]
        best_fitness = self.archive[0]["fitness"]
        
        return best_solution,best_fitness
                
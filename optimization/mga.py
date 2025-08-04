# MODELING TO GENERATE ALTERNATIVE (MGA)
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class MGA_VRP(object):
    def __init__(self,population_size = 10,m=5,n=10,const=3,max_iter = 100,max_idem = 20,random_state=None):
        self.db = ConDB()
        
        # parameter setting
        self.population_size = population_size #number of individual in the population
        self.m = m #number of candidates returned by the breed function
        self.n = n #number of alternative solutions (archive)
        self.const = const #suitable constant chosen in advance to allow an acceptable deviation from the obj function (to calculate target fitness)
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # data model setting
        self.tour = None #POI yang dipilih oleh user untuk dikunjungi
        self.hotel = None #hotel yang dipilih oleh user
        self.timematrix = None
        self.max_travel_time = None
        self.travel_days = None
        
        #degree of interest (DOI for MAUT) setting
        self.degree_waktu = 1
        self.degree_tarif = 1
        self.degree_rating = 1
        self.degree_poi = 1
        self.degree_poi_penalty = 1
        self.degree_time_penalty = 1
        
        #scaler setting
        self.min_rating = None
        self.max_rating = None
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
    
    def set_model(self,tour,hotel,timematrix,travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        #initiate model
        self.tour = copy.deepcopy(tour)
        self.hotel = copy.deepcopy(hotel)
        self.travel_days = travel_days
        self.hotel.depart_time = depart_time
        self.max_travel_time = max_travel_time
        self.timematrix = copy.deepcopy(timematrix)
        
        self.degree_waktu = degree_waktu
        self.degree_tarif = degree_tarif
        self.degree_rating = degree_rating
        
        self.min_rating = min([node.rating for node in self.tour])
        self.max_rating = max([node.rating for node in self.tour])
        self.min_tarif = min([node.tarif for node in self.tour])
        self.max_tarif = sum([node.tarif for node in self.tour])
        self.min_waktu = 0
        self.max_waktu = (self.diff_second_between_time(depart_time,max_travel_time))*self.travel_days
        self.min_poi = 0
        self.max_poi = len(self.tour)
        self.min_poi_penalty = 0
        self.max_poi_penalty = len(self.tour)
        self.min_time_penalty = 0
        self.max_time_penalty = ((24*3600)-self.diff_second_between_time(max_travel_time,depart_time))*travel_days
        
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
    
    def set_max_iter(self,max_iter):
        self.max_iter = max_iter
    
    def time_to_second(self,time):
        return (time.hour*3600)+(time.minute*60)+time.second
    
    def second_to_time(self,second):
        second = int(second)
        return datetime.time(second//3600,(second//60)%60,0) #ignore second detail
    
    def diff_second_between_time(self,time_a,time_b):
        #input: time_a and time_b, datetime.time()
        #output: time_b - time_a, seconds (int)
        return self.time_to_second(time_b) - self.time_to_second(time_a)
    
    def min_max_scaler(self,min_value,max_value,value):
        if max_value-min_value == 0:
            return 0
        else:
            return (value-min_value)/(max_value-min_value)
    
    def MAUT(self,solutions,consider_total_poi = True,use_penalty = True):
        #input: optimization solutions, format = [{"index":[],"waktu":[],"rating":[],"tarif":[]},...]
        #output: fitness value calculated using MAUT
        
        #concat all attribute lists (except for waktu)
        index_ls = sum([i['index'] for i in solutions],[])
        rating_ls = sum([i['rating'] for i in solutions],[])
        tarif_ls = sum([i['tarif'] for i in solutions],[])
        
        waktu_ls = [i['waktu'] for i in solutions]
        
        #rating
        avg_rating = sum(rating_ls)/len(rating_ls)
        score_rating = self.min_max_scaler(self.min_rating,self.max_rating,avg_rating)*self.degree_rating
        
        #tarif
        sum_tarif = sum(tarif_ls)
        score_tarif = (1-self.min_max_scaler(self.min_tarif,self.max_tarif,sum_tarif)) * self.degree_tarif
        
        #waktu
        waktu_per_day = [self.diff_second_between_time(i[0],i[-1]) for i in waktu_ls]
        sum_waktu = sum(waktu_per_day)
        score_waktu = (1-self.min_max_scaler(self.min_waktu,self.max_waktu,sum_waktu))*self.degree_waktu
        
        #poi
        count_poi = len(index_ls)
        score_poi = self.min_max_scaler(self.min_poi,self.max_poi,count_poi) if consider_total_poi == True else 0
        
        if use_penalty==True:
            #poi penalty
            penalty_index = [node._id for node in self.tour if node._id not in index_ls]
            count_penalty = len(penalty_index)
            score_poipenalty = (1-self.min_max_scaler(self.min_poi_penalty,self.max_poi_penalty,count_penalty)) * self.degree_poi_penalty
            
            #time penalty
            penalty_per_day = [max(self.diff_second_between_time(self.max_travel_time,i[-1]),0) for i in waktu_ls]
            sum_time_penalty = sum(penalty_per_day)
            score_timepenalty = (1-self.min_max_scaler(self.min_time_penalty,self.max_time_penalty,sum_time_penalty)) * self.degree_time_penalty
        else:
            score_poipenalty = 0
            score_timepenalty = 0
            
        #MAUT
        degree_rating = self.degree_rating
        degree_tarif = self.degree_tarif
        degree_waktu = self.degree_waktu
        degree_poi = self.degree_poi if consider_total_poi == True else 0
        degree_poi_penalty = self.degree_poi_penalty if use_penalty == True else 0
        degree_time_penalty = self.degree_time_penalty if use_penalty == True else 0

        pembilang = score_rating+score_tarif+score_waktu+score_poi+score_poipenalty+score_timepenalty
        penyebut = degree_rating+degree_tarif+degree_waktu+degree_poi+degree_poi_penalty+degree_time_penalty
        maut = pembilang/penyebut
        return maut
    
    def MAUT_between_two_nodes(self,current_node,next_node):
        score_rating = self.degree_rating * self.min_max_scaler(self.min_rating,self.max_rating,next_node.rating)
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.rating))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.timematrix[current_node._id][next_node._id]['waktu']))
        maut = (score_rating+score_tarif+score_waktu)/(self.degree_rating+self.degree_tarif+self.degree_waktu)
        return maut
    
    def next_node_check(self,current_node,next_node):
        time_needed = self.time_to_second(current_node.depart_time)+self.timematrix[current_node._id][next_node._id]["waktu"]+next_node.waktu_kunjungan
        time_limit = self.time_to_second(self.max_travel_time)
        if (time_needed <= time_limit) and (time_needed <= self.time_to_second(next_node.jam_tutup)):
            return True
        else:
            return False
    
    def set_next_node_depart_arrive_time(self,current_node,next_node):
        arrive_time = self.time_to_second(current_node.depart_time)+self.timematrix[current_node._id][next_node._id]["waktu"]
        arrive_time = max([arrive_time,self.time_to_second(next_node.jam_buka)])
        next_node.arrive_time = self.second_to_time(arrive_time)
        if next_node.tipe.lower() != "hotel":
            next_node.depart_time = self.second_to_time(arrive_time+next_node.waktu_kunjungan)
        return next_node
    
    def solution_list_of_nodes_to_dict(self, solutions):
        solution_dict = []
        for day in solutions:
            day_solution = {"index":[],"waktu":[self.hotel.depart_time],"rating":[],"tarif":[]}
            day_solution['index'] = [i._id for i in day]
            day_solution['rating'] = [i.rating for i in day]
            day_solution['tarif'] = [i.tarif for i in day]
            
            last_waktu = self.time_to_second(day[-1].depart_time) + self.timematrix[day[-1]._id][self.hotel._id]['waktu']
            day_solution['waktu'].extend([i.arrive_time for i in day] + [self.second_to_time(last_waktu)])
            
            solution_dict.append(day_solution)
        return solution_dict
    
    def split_itinerary(self,init_itinerary):
        final_solution = [] #2d list of nodes
        day = 1
        tabu_nodes = [] #list of tabu node's id
        while day <= self.travel_days:
            current_node = self.hotel
            day_solution = [] #list of nodes
            next_node_candidates = [node for node in init_itinerary if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                if self.next_node_check(current_node,next_node_candidates[i]):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i])
                    day_solution.append(next_node_candidates[i])
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
            
            if len(day_solution) > 0:
                final_solution.append(day_solution)
            
            if len(tabu_nodes) == len(self.tour):
                break
            
            day += 1
        return final_solution
    
    def generate_solution(self, init_solution = []):
        solution = self.split_itinerary(random.sample(self.tour,len(self.tour))) if len(init_solution) == 0 else self.split_itinerary(init_solution)
        rest_nodes = self.get_rest_nodes_from_solution(solution)
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict(solution))
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
        archive_id = [[i._id for i in sum(j["solution"],[])] for j in self.archive]
        sol_id = [i._id for i in sum(solution,[])]
        
        scores = [self.jaccard_distance(sol_id,archive_sol_id) for archive_sol_id in archive_id]
        
        return min(scores)
    
    def update_dissimilarity(self,population):
        for i in range(len(population)):
            population[i]["dissimilarity"] = compute_dissimilarity(population[i])
        
        return population
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node._id for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.tour if node._id not in solution_id]
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
        best_solution = self.solution_list_of_nodes_to_dict(self.archive[0]["solution"])
        best_fitness = self.archive[0]["fitness"]
        
        return best_solution,best_fitness
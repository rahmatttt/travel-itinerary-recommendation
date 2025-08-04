#HYBRID ITERATED LOCAL SEARCH - SIMULATED ANNEALING - TABU SEARCH
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class ILS_SA_TS_VRP(object):
    def __init__(self,stepsize = 0.1,strength=0.5,max_tabu_size = 10,num_neighbor = 10,temperature=1000,cooling_rate=0.95,stopping_temperature=0.0002,max_iter = 100,max_idem = 20,random_state=None):
        self.db = ConDB()
        
        # parameter setting
        self.stepsize = stepsize #step size in local search process
        self.strength = strength #perturbation strength (to escape the local optimum)
        self.max_tabu_size = max_tabu_size #maximum size of tabu list
        self.num_neighbor = num_neighbor #the number of neighbor in neighborhood for new solution candidates in TS process
        self.temperature = temperature #initial temperature for SA process
        self.stopping_temperature = stopping_temperature #minimum temperature to continue iteration for SA process
        self.cooling_rate = cooling_rate #cooling rate for SA process
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # data model setting
        self.tour = None #POI yang dipilih oleh user untuk dikunjungi
        self.hotel = None #hotel yang dipilih oleh user
        self.timematrix = None
        self.max_travel_time = None
        self.travel_days = None
        
        #degree of interest (DOI for MAUT) setting
        self.degree_waktu = None
        self.degree_tarif = None
        self.degree_rating = None
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
            np.random.seed(random_state)
    
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
    
    def solution_list_of_float_to_nodes(self,solutions):
        index_order = np.flip(np.argsort(solutions))
        return list(np.array(self.tour)[index_order])
    
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
    
    def generate_init_solution(self):
        return np.random.uniform(-5.12,5.12,size=len(self.tour))
    
    def local_search_sa(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(solution))))
        temperature = self.temperature
        
        while temperature >= self.stopping_temperature:
            #generate neighbor
            candidate = solution + np.random.uniform(-self.stepsize,self.stepsize, size=len(solution))
            candidate_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(candidate))))
            
            #calculate acceptance probability
            acceptance_prob = np.exp((candidate_fitness - fitness) / temperature)
            
            #accept if better
            if candidate_fitness > fitness or random.uniform(0,1) < acceptance_prob:
                solution = copy.deepcopy(candidate)
                fitness = candidate_fitness
            
            temperature = self.cooling_rate * temperature
        
        return solution,fitness
    
    def local_search_ts(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(solution))))
        
        tabu_list = []
        
        idem_counter = 0
        for i in range(self.max_iter):
            add_idem = True
            #generate neighborhood
            for j in range(self.num_neighbor):
                addition = np.random.uniform(-self.stepsize, self.stepsize, size=len(solution))
                if len(tabu_list) > 0:
                    if np.any(np.all(addition == tabu_list, axis=1)) == False:
                        neighbor = solution + addition
                        neighbor_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor))))
                        tabu_list.append(addition)
                else:
                    neighbor = solution + addition
                    neighbor_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(neighbor))))
                    tabu_list.append(addition)
                
                if neighbor_fitness > fitness:
                    solution = copy.deepcopy(neighbor)
                    fitness = neighbor_fitness
                    add_idem = False
            
            #check tabu list
            if len(tabu_list) > self.max_tabu_size:
                tabu_list = tabu_list[1:]
            
            if add_idem == True:
                idem_counter += 1
            else:
                idem_counter = 0
            
            if idem_counter > self.max_idem:
                return solution,fitness
        
        return solution,fitness
    
    def hybrid_local_search(self,solution_nodes):
        sol_sa,fitness_sa = self.local_search_sa(solution_nodes)
        sol_ts,fitness_ts = self.local_search_ts(solution_nodes)
        
        if fitness_sa > fitness_ts:
            return sol_sa,fitness_sa
        else:
            return sol_ts,fitness_ts
            
    def perturbation(self,solution_nodes):
        return solution_nodes + np.random.uniform(-self.strength,self.strength,size=len(solution_nodes))
    
    def construct_solution(self):
        idem_counter = 0
        
        best_solution,best_fitness = self.hybrid_local_search(self.generate_init_solution())
        
        for i in range(self.max_iter):
            #perturbation
            candidate = self.perturbation(best_solution)
            
            #local search
            candidate,candidate_fitness = self.hybrid_local_search(candidate)
            
            if candidate_fitness > best_fitness:
                best_solution = copy.deepcopy(candidate)
                best_fitness = candidate_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    best_solution = self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution)))
                    return best_solution,best_fitness
        
        best_solution = self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution)))
        return best_solution,best_fitness
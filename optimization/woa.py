# WHALE OPTIMIZATION ALGORITHM 
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class WOA_VRP(object):
    def __init__(self,population = 50,b = 1,max_iter = 1000,max_idem = 20,random_state=None):
        self.db = ConDB()
        
        # parameter setting
        self.population = population #population size
        self.b = b #logarithmic spiral
        self.a = 2 #a value set to 2 and will gradually decrease throughout the iterations until it reaches 0
        self.max_iter = max_iter #max iteration
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
        
    def generate_population(self):
        population_ls = [np.array([random.uniform(0,10) for i in range(len(self.tour))]) for j in range(self.population)]
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
                               key=lambda x: self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(x)))),
                              reverse=True)
        
        best_solution = population_ls[0]
        best_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))))
        
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
                               key=lambda x: self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(x)))),
                              reverse=True)
            
            population_ls = population_ls[:self.population]
            
            best_found_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(population_ls[0]))))
            
            self.a = 2*(1-((i+1)/self.max_iter))
            
            if best_found_fitness > best_fitness:
                best_solution = population_ls[0]
                best_fitness = best_found_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))),best_fitness
            
        return self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))),best_fitness
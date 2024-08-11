from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class TS_VRP(object):
    def __init__(self,max_tabu_size = 20,max_iter = 7500,max_idem = 7500,random_state=None):
        self.db = ConDB()
        
        # parameter setting
        self.max_tabu_size = max_tabu_size #max size of tabu list
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
        self.rest_nodes = [] #1D list of nodes, [node1,node2,node4,node5,....]

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
    
    def set_model(self,tour,hotel,timematrix,init_solution=[],travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
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
        
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        else:
            self.init_solution = self.generate_init_solution()
        self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)
    
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
    
    def generate_init_solution(self):
        solution = []
        day = 1
        tabu_nodes = []
        while day<=self.travel_days:
            current_node = self.hotel
            day_solution = []
            
            for pos in range(len(self.tour)-1):
                #recheck next node candidates (perlu dicek jam sampainya apakah melebihi max time)
                next_node_candidates = [node for node in self.tour if self.next_node_check(current_node,node)==True and node._id not in tabu_nodes]
                if len(next_node_candidates)>0:
                    max_pos = np.argmax([self.MAUT_between_two_nodes(current_node,next_node) for next_node in next_node_candidates])
                    next_node = next_node_candidates[max_pos]
                    next_node = self.set_next_node_depart_arrive_time(current_node,next_node)
                    day_solution.append(next_node)
                    tabu_nodes.append(next_node._id)
                    current_node = next_node
                else:
                    break
            
            if len(day_solution)>0:
                solution.append(day_solution)
            
            if len(tabu_nodes) == len(self.tour):
                break
            
            day += 1
        
        return solution
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node._id for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.tour if node._id not in solution_id]
        return rest_nodes
    
    def reset_depart_arrive_time(self,itinerary):
        #input : list of nodes itinerary
        #output: updated depart and arrive time, if any unvalid time then return False
        solution = copy.deepcopy(itinerary)
        
        current_node = self.hotel
        for i in range(len(solution)):
            if self.next_node_check(current_node,solution[i]):
                solution[i] = self.set_next_node_depart_arrive_time(current_node,solution[i])
                current_node = solution[i]
            else:
                return itinerary,False #cannot reset
        return solution,True
    
    def construct_solution(self):
        solution = copy.deepcopy(self.init_solution)
        solution_dict = self.solution_list_of_nodes_to_dict(solution)
        fitness = self.MAUT(solution_dict)
        sol_list = sum(solution,[])+self.rest_nodes
        
        idem_counter = 0
        tabu_list = []
        for i in range(self.max_iter):
            first_pos,second_pos = random.sample(range(len(sol_list)),2)
            first_node = sol_list[first_pos]
            second_node = sol_list[second_pos]
            
            if (first_node._id,second_node._id) in tabu_list or (second_node._id,first_node._id) in tabu_list:
                idem_counter += 1
            else:
                #swap
                sol_list[first_pos],sol_list[second_pos] = sol_list[second_pos],sol_list[first_pos]
                new_sol = self.split_itinerary(sol_list)
                new_solution_dict = self.solution_list_of_nodes_to_dict(new_sol)
                new_fitness = self.MAUT(new_solution_dict)
                if new_fitness > fitness:
                    solution = copy.deepcopy(new_sol)
                    fitness = new_fitness
                    solution_dict = copy.deepcopy(new_solution_dict)
                    self.rest_nodes = self.get_rest_nodes_from_solution(solution)
                    sol_list = sum(solution,[])+self.rest_nodes
                    tabu_list.append((first_node._id,second_node._id))
                    idem_counter = 0
                else:
                    idem_counter += 1
            if idem_counter%5 == 0 and idem_counter>0:
                tabu_list = []
            if len(tabu_list) > self.max_tabu_size:
                tabu_list = tabu_list[1:]
            if idem_counter > self.max_idem:
                return solution_dict,fitness
        
        return solution_dict,fitness
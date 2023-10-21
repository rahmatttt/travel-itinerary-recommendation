from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class ACS_VRP(object):
    def __init__(self,alpha_t = 1,beta = 1,q0 = 0.1,init_pheromone = 0.1,rho = 0.1,alpha = 0.1,num_ant = 30,max_iter = 200,max_idem=30,random_state=None):
        self.db = ConDB()
        
        #parameter setting
        self.alpha_t = alpha_t #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in ACS transition rule
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate local pheromone update
        self.alpha = alpha #evaporation rate global pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration ACS
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
        self.timematrix = self.add_pheromone_to_timematrix(copy.deepcopy(timematrix))
        
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
    
    def add_pheromone_to_timematrix(self, timematrix):
        for i in timematrix:
            for j in timematrix[i]:
                timematrix[i][j]['pheromone'] = self.init_pheromone
        return timematrix
    
    def min_max_scaler(self,min_value,max_value,value):
        return (value-min_value)/(max_value-min_value)
    
    def MAUT(self,solutions,use_penalty = True):
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
        score_tarif = 1-self.min_max_scaler(self.min_tarif,self.max_tarif,sum_tarif) * self.degree_tarif
        
        #waktu
        waktu_per_day = [self.diff_second_between_time(i[0],i[-1]) for i in waktu_ls]
        sum_waktu = sum(waktu_per_day)
        score_waktu = 1-self.min_max_scaler(self.min_waktu,self.max_waktu,sum_waktu)*self.degree_waktu
        
        #poi
        count_poi = len(index_ls)
        score_poi = self.min_max_scaler(self.min_poi,self.max_poi,count_poi)
        
        if use_penalty==True:
            #poi penalty
            penalty_index = [node._id for node in self.tour if node._id not in index_ls]
            count_penalty = len(penalty_index)
            score_poipenalty = 1-self.min_max_scaler(self.min_poi_penalty,self.max_poi_penalty,count_penalty) * self.degree_poi_penalty
            
            #time penalty
            penalty_per_day = [max(self.diff_second_between_time(i[-1],self.max_travel_time),0) for i in waktu_ls]
            sum_time_penalty = sum(penalty_per_day)
            score_timepenalty = 1-self.min_max_scaler(self.min_time_penalty,self.max_time_penalty,sum_time_penalty) * self.degree_time_penalty
        else:
            score_poipenalty = 0
            score_timepenalty = 0
            
        #MAUT
        pembilang = score_rating+score_tarif+score_waktu+score_poi+score_poipenalty+score_timepenalty
        penyebut = self.degree_rating+self.degree_tarif+self.degree_waktu+self.degree_poi+self.degree_poi_penalty+self.degree_time_penalty
        maut = pembilang/penyebut
        return maut
    
    def MAUT_between_two_nodes(self,current_node,next_node):
        score_rating = self.degree_rating * self.min_max_scaler(self.min_rating,self.max_rating,next_node.rating)
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.rating))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.timematrix[current_node._id][next_node._id]['waktu']))
        maut = (score_rating+score_tarif+score_waktu)/(self.degree_rating+self.degree_tarif+self.degree_waktu)
        return maut
    
    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix):
        max_pos = np.argmax([local_pheromone_matrix[current_node._id][next_node._id]['pheromone']*(self.MAUT_between_two_nodes(current_node,next_node)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node
    
    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val
        
        #probability
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            next_node_prob.append(node_prob)
        
        next_node = random.choices(next_node_candidates,next_node_prob,k=1)
        return next_node[0]
    
    def transition_rule(self,current_node,next_node_candidates,local_pheromone_matrix):
        q = random.uniform(0,1)
        if q <= self.q0: #exploitation
            next_node = self.exploitation(current_node,next_node_candidates,local_pheromone_matrix)
        else: #exploration
            next_node = self.exploration(current_node,next_node_candidates,local_pheromone_matrix)
        return next_node
    
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
    
    def local_pheromone_update(self,solutions,fitness,local_pheromone_matrix):
        for day in solutions:
            nodes = [self.hotel._id] + day['index'] + [self.hotel._id]
            i = 1
            while i < len(nodes):
                pheromone = local_pheromone_matrix[nodes[i-1]][nodes[i]]['pheromone']
                pheromone = ((1-self.rho)*pheromone)+(self.rho*fitness)
                local_pheromone_matrix[nodes[i-1]][nodes[i]]['pheromone'] = pheromone
                i += 1
        return local_pheromone_matrix
    
    def global_pheromone_update(self,best_solutions,best_fitness):
        solutions = copy.deepcopy(best_solutions)
        solution_index = []
        for day in solutions:
            day_solution = [self.hotel._id] + day['index'] + [self.hotel._id] 
            day_solution = [(day_solution[i-1],day_solution[i]) for i in range(1,len(day['index']))]
            solution_index = solution_index + day_solution
        
        for node in self.timematrix:
            for next_node in self.timematrix[node]:
                pheromone = self.timematrix[node][next_node]['pheromone']
                if (node,next_node) in solution_index:
                    pheromone = ((1-self.alpha)*pheromone)+(self.alpha*best_fitness)
                else:
                    pheromone = ((1-self.alpha)*pheromone)+0
                self.timematrix[node][next_node]['pheromone'] = pheromone
    
    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            best_found_solution = None
            best_found_fitness = 0
            local_pheromone_matrix = copy.deepcopy(self.timematrix)
            for ant in range(self.num_ant): #step
                ant_solution = []
                day = 1
                tabu_nodes = []
                while day<=self.travel_days:
                    current_node = self.hotel
                    ant_day_solution = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}
                    
                    for pos in range(len(self.tour)+1):
                        #recheck next node candidates (perlu dicek jam sampainya apakah melebihi max time)
                        next_node_candidates = [node for node in self.tour if self.next_node_check(current_node,node)==True and node._id not in tabu_nodes]
                        
                        if len(next_node_candidates) > 0:
                            #transition rules
                            next_node = self.transition_rule(current_node,next_node_candidates,local_pheromone_matrix)
                            next_node = self.set_next_node_depart_arrive_time(current_node,next_node)

                            #change current node and delete it from available nodes
                            current_node = next_node
                            ant_day_solution['index'].append(current_node._id)
                            ant_day_solution['rating'].append(current_node.rating)
                            ant_day_solution['tarif'].append(current_node.tarif)
                            ant_day_solution['waktu'].append(current_node.arrive_time)
                            tabu_nodes.append(current_node._id)
                        elif len(next_node_candidates) == 0 and current_node._id != self.hotel._id:
                            last_node = copy.deepcopy(self.hotel)
                            last_node = self.set_next_node_depart_arrive_time(current_node,last_node)
                            ant_day_solution['waktu'].append(last_node.arrive_time)
                            break
                        else:
                            break
                    
                    if len(ant_day_solution['index'])>0:
                        ant_solution.append(ant_day_solution)

                    if len(tabu_nodes) == len(self.tour):
                        break

                    day += 1
                
                fitness = self.MAUT(ant_solution)
                if fitness > best_found_fitness:
                    best_found_fitness = fitness
                    best_found_solution = copy.deepcopy(ant_solution)
                
                # local pheromone update
                local_pheromone_matrix = self.local_pheromone_update(ant_solution,fitness,local_pheromone_matrix)
            
            #global pheromone update
            self.global_pheromone_update(best_found_solution,best_found_fitness)
            
            #checking best vs best found
            if best_found_fitness > best_fitness:
                best_fitness = best_found_fitness
                best_solution = copy.deepcopy(best_found_solution)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness


class ACS_TSP(object):
    def __init__(self,alpha_t = 1,beta = 1,q0 = 0.1,init_pheromone = 0.1,rho = 0.1,alpha = 0.1,num_ant = 30,max_iter = 200,max_idem=30,random_state=None):
        self.db = ConDB()
        
        #parameter setting
        self.alpha_t = alpha_t #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in ACS transition rule
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate local pheromone update
        self.alpha = alpha #evaporation rate global pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration ACS
        self.max_idem = max_idem
        
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
        self.timematrix = self.add_pheromone_to_timematrix(copy.deepcopy(timematrix))
        
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
    
    def add_pheromone_to_timematrix(self, timematrix):
        for i in timematrix:
            for j in timematrix[i]:
                timematrix[i][j]['pheromone'] = self.init_pheromone
        return timematrix
    
    def min_max_scaler(self,min_value,max_value,value):
        return (value-min_value)/(max_value-min_value)
    
    def MAUT_TSP(self,solutions):
        #concat all attribute lists (except for waktu)
        index_ls = solutions['index']
        rating_ls = solutions['rating']
        tarif_ls = solutions['tarif']
                
        #rating
        avg_rating = sum(rating_ls)/len(rating_ls)
        score_rating = self.min_max_scaler(self.min_rating,self.max_rating,avg_rating)*self.degree_rating
        
        #tarif
        sum_tarif = sum(tarif_ls)
        score_tarif = 1-self.min_max_scaler(self.min_tarif,self.max_tarif,sum_tarif) * self.degree_tarif
        
        #waktu
        sum_waktu = solutions['waktu']
        score_waktu = 1-self.min_max_scaler(self.min_waktu,self.max_waktu,sum_waktu)*self.degree_waktu
        
        #MAUT
        pembilang = score_rating+score_tarif+score_waktu
        penyebut = self.degree_rating+self.degree_tarif+self.degree_waktu
        maut = pembilang/penyebut
        return maut
    
    def MAUT(self,solutions,use_penalty = True):
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
        score_tarif = 1-self.min_max_scaler(self.min_tarif,self.max_tarif,sum_tarif) * self.degree_tarif
        
        #waktu
        waktu_per_day = [self.diff_second_between_time(i[0],i[-1]) for i in waktu_ls]
        sum_waktu = sum(waktu_per_day)
        score_waktu = 1-self.min_max_scaler(self.min_waktu,self.max_waktu,sum_waktu)*self.degree_waktu
        
        #poi
        count_poi = len(index_ls)
        score_poi = self.min_max_scaler(self.min_poi,self.max_poi,count_poi)
        
        if use_penalty==True:
            #poi penalty
            penalty_index = [node._id for node in self.tour if node._id not in index_ls]
            count_penalty = len(penalty_index)
            score_poipenalty = 1-self.min_max_scaler(self.min_poi_penalty,self.max_poi_penalty,count_penalty) * self.degree_poi_penalty
            
            #time penalty
            penalty_per_day = [max(self.diff_second_between_time(i[-1],self.max_travel_time),0) for i in waktu_ls]
            sum_time_penalty = sum(penalty_per_day)
            score_timepenalty = 1-self.min_max_scaler(self.min_time_penalty,self.max_time_penalty,sum_time_penalty) * self.degree_time_penalty
        else:
            score_poipenalty = 0
            score_timepenalty = 0
            
        #MAUT
        pembilang = score_rating+score_tarif+score_waktu+score_poi+score_poipenalty+score_timepenalty
        penyebut = self.degree_rating+self.degree_tarif+self.degree_waktu+self.degree_poi+self.degree_poi_penalty+self.degree_time_penalty
        maut = pembilang/penyebut
        return maut
    
    def MAUT_between_two_nodes(self,current_node,next_node):
        score_rating = self.degree_rating * self.min_max_scaler(self.min_rating,self.max_rating,next_node.rating)
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.rating))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.timematrix[current_node._id][next_node._id]['waktu']))
        maut = (score_rating+score_tarif+score_waktu)/(self.degree_rating+self.degree_tarif+self.degree_waktu)
        return maut
    
    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix):
        max_pos = np.argmax([local_pheromone_matrix[current_node._id][next_node._id]['pheromone']*(self.MAUT_between_two_nodes(current_node,next_node)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node
    
    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val
        
        #probability
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            next_node_prob.append(node_prob)
        
        next_node = random.choices(next_node_candidates,next_node_prob,k=1)
        return next_node[0]
    
    def transition_rule(self,current_node,next_node_candidates,local_pheromone_matrix):
        q = random.uniform(0,1)
        if q <= self.q0: #exploitation
            next_node = self.exploitation(current_node,next_node_candidates,local_pheromone_matrix)
        else: #exploration
            next_node = self.exploration(current_node,next_node_candidates,local_pheromone_matrix)
        return next_node
    
    def next_node_check(self,current_node,next_node):
        time_needed = self.time_to_second(current_node.depart_time)+self.timematrix[current_node._id][next_node._id]["waktu"]+next_node.waktu_kunjungan
        time_limit = self.time_to_second(self.max_travel_time)
        if (time_needed <= time_limit):
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
    
    def local_pheromone_update(self,solutions,fitness,local_pheromone_matrix):
        nodes = [self.hotel._id] + solutions['index'] + [self.hotel._id]
        i = 1
        while i < len(nodes):
            pheromone = local_pheromone_matrix[nodes[i-1]][nodes[i]]['pheromone']
            pheromone = ((1-self.rho)*pheromone)+(self.rho*fitness)
            local_pheromone_matrix[nodes[i-1]][nodes[i]]['pheromone'] = pheromone
            i += 1
        return local_pheromone_matrix
    
    def global_pheromone_update(self,best_solutions,best_fitness):
        solutions = [self.hotel._id] + best_solutions['index'] + [self.hotel._id]
        solution_index = [(solutions[i-1],solutions[i]) for i in range(1,len(solutions))]
        
        for node in self.timematrix:
            for next_node in self.timematrix[node]:
                pheromone = self.timematrix[node][next_node]['pheromone']
                if (node,next_node) in solution_index:
                    pheromone = ((1-self.alpha)*pheromone)+(self.alpha*best_fitness)
                else:
                    pheromone = ((1-self.alpha)*pheromone)+0
                self.timematrix[node][next_node]['pheromone'] = pheromone
     
    def TSP(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            best_found_solution = None
            best_found_solution_dict = None
            best_found_fitness = 0
            local_pheromone_matrix = copy.deepcopy(self.timematrix)
            for ant in range(self.num_ant): #step
                ant_solution = []
                ant_solution_dict = {"index":[],"waktu":0,"rating":[],"tarif":[]}
                current_node = self.hotel
                
                for pos in range(len(self.tour)+1):
                    #generate next node candidates
                    next_node_candidates = [node for node in self.tour if node._id not in ant_solution_dict['index']]
                    
                    if len(next_node_candidates)>0:
                        #transition rules
                        next_node = self.transition_rule(current_node,next_node_candidates,local_pheromone_matrix)
                        
                        #add to solution list
                        ant_solution.append(next_node)
                        ant_solution_dict['index'].append(next_node._id)
                        ant_solution_dict['rating'].append(next_node.rating)
                        ant_solution_dict['tarif'].append(next_node.tarif)
                        ant_solution_dict['waktu'] += self.timematrix[current_node._id][next_node._id]['waktu']+next_node.waktu_kunjungan
                        
                        #change current node
                        current_node = next_node
                    elif len(next_node_candidates) == 0 and current_node._id != self.hotel._id:
                        last_node = copy.deepcopy(self.hotel)
                        ant_solution_dict['waktu'] += self.timematrix[current_node._id][last_node._id]['waktu']
                        break
                    else:
                        break
                
                fitness = self.MAUT_TSP(ant_solution_dict)
                if fitness > best_found_fitness:
                    best_found_fitness = fitness
                    best_found_solution = copy.deepcopy(ant_solution)
                    best_found_solution_dict = copy.deepcopy(ant_solution_dict)
                
                # local pheromone update
                local_pheromone_matrix = self.local_pheromone_update(ant_solution_dict,fitness,local_pheromone_matrix)
            
            #global pheromone update
            self.global_pheromone_update(best_found_solution_dict,best_found_fitness)
            
            #checking best vs best found
            if best_found_fitness > best_fitness:
                best_fitness = best_found_fitness
                best_solution = copy.deepcopy(best_found_solution)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness
    
    def construct_solution(self):
        solution,fitness = self.TSP()
        day = 1
        final_solution = []
        tabu_nodes = []
        while day <= self.travel_days:
            current_node = self.hotel
            day_solution = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}
            next_node_candidates = [node for node in solution if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                time_needed = self.time_to_second(current_node.depart_time)+self.timematrix[current_node._id][next_node_candidates[i]._id]["waktu"]+next_node_candidates[i].waktu_kunjungan
                if time_needed >= self.time_to_second(next_node_candidates[i].jam_tutup):
                    continue
                elif self.next_node_check(current_node,next_node_candidates[i]):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i])
                    day_solution['index'].append(next_node_candidates[i]._id)
                    day_solution['waktu'].append(next_node_candidates[i].arrive_time)
                    day_solution['rating'].append(next_node_candidates[i].rating)
                    day_solution['tarif'].append(next_node_candidates[i].tarif)
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
                else:
                    break
            if current_node._id != self.hotel._id:
                self.hotel = self.set_next_node_depart_arrive_time(current_node,self.hotel)
                day_solution['waktu'].append(self.hotel.arrive_time)
            
            if len(day_solution['index']) > 0:
                final_solution.append(day_solution)
            
            if len(tabu_nodes) == len(self.tour):
                break

            day += 1
        
        final_fitness = self.MAUT(final_solution)
        return final_solution,final_fitness
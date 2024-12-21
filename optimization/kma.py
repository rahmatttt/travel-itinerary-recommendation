# KOMODO MLIPIR ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class KMA_VRP(object):
    def __init__(self,n1 = 5,n2 = 200,max_n2 = 300,min_n2 = 20,p1=0.5,p2=0.5,d2=0.5,max_iter = 1000,max_idem = 20,random_state=None):
        self.db = ConDB()
        
        # parameter setting
        self.n1 = n1 #number of komodo(population) in first phase
        self.n2 = n2 #number of komodo in second phase
        self.max_n2 = max_n2 #number of maximum komodo for second phase
        self.min_n2 = min_n2 #number of minimum komodo for second phase
        self.m = None #number of dimension
        self.p1 = p1 #portion of big male komodo in first phase
        self.p2 = p2 #portion of big male komodo in second phase
        self.d1 = None #mlipir rate in first phase
        self.d2 = d2 #mlipir rate in second phase
        self.a = 5 #number of decreased or increased generation in second phase
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
        
        # initiate number of each komodo for the first phase
        self.n_big_male = int(np.ceil((self.p1*self.n1)-1))
        self.n_female = 1
        self.n_small_male = int(self.n1 - (self.n_big_male+self.n_female))
        
        self.m = len(self.tour) #number of dimension
        self.d1 = (self.m - 1)/self.m #mlipir rate in first phase
        
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
    
    def initiate_cluster_komodo(self,komodo_ls):
        #for first clustering
        komodo_dict = [{'solution':i,
                       'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(i))))} for i in komodo_ls]
        komodo_dict = sorted(komodo_dict, key=lambda x: x["fitness"], reverse = True)
        big_males = copy.deepcopy(komodo_dict[:self.n_big_male])
        female = copy.deepcopy(komodo_dict[self.n_big_male])
        small_males = copy.deepcopy(komodo_dict[self.n_big_male+self.n_female:])
        return big_males,female,small_males
    
    def re_cluster_komodo(self,komodo_dict):
        #for reclustering
        komodo_dict = sorted(komodo_dict, key=lambda x: x["fitness"], reverse = True)
        big_males = copy.deepcopy(komodo_dict[:self.n_big_male])
        female = copy.deepcopy(komodo_dict[self.n_big_male])
        small_males = copy.deepcopy(komodo_dict[self.n_big_male+self.n_female:])
        return big_males,female,small_males
    
    def generate_population_first_phase(self):
        #the individu (komodo) is a list consists of a number [0,1]
        komodo_ls = [np.array([random.random() for i in range(self.m)]) for j in range(self.n1)]
        return komodo_ls
    
    def generate_population_second_phase(self, best_so_far_komodo, initiate_population = True):
        if initiate_population == True:
            komodo_ls = [np.array([random.random() for i in range(self.m)]) for j in range(self.n2)]
        else:
            komodo_ls = [np.array([random.random() for i in range(self.m)]) for i in range(self.a)]
        return komodo_ls
    
    def big_male_movement(self,big_males):
        new_big_males = []
        for i in range(len(big_males)):
            sum_wij = 0
            for j in range(len(big_males)):
                if i!=j:
                    r1 = random.uniform(0,1)
                    r2 = random.uniform(0,1)
                    if big_males[i]['fitness']<big_males[j]['fitness'] or r2<0.5:
                        wij = r1*(big_males[j]['solution'] - big_males[i]['solution'])
                    else:
                        wij = r1*(big_males[i]['solution'] - big_males[j]['solution'])
                    sum_wij = sum_wij + wij
            new_big_male = big_males[i]['solution'] + sum_wij
            fitness_new_big_male = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(new_big_male))))
            new_big_males.append({'solution':new_big_male,
                                 'fitness':fitness_new_big_male})
        
        new_big_males = new_big_males + big_males
        new_big_males = sorted(new_big_males, key=lambda x: x["fitness"], reverse = True)
        return new_big_males[:self.n_big_male]
    
    def female_movement(self,female,biggest_male):
        if random.uniform(0,1)<0.5:
            random_number = np.random.uniform(0,1,size=self.m)
            first_child = (random_number * female['solution']) + ((1-random_number)*female['solution'])
            second_child = (random_number * female['solution']) + ((1-random_number)*female['solution'])
            females = [
                {'solution':first_child,
                'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(first_child))))},
                {'solution':second_child,
                'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(second_child))))},
                copy.deepcopy(female)
            ]
        else:
            ub = max(female['solution'])
            lb = min(female['solution'])
            alpha = 0.1 #radius of parthenogenesis (fixed value)
            r = random.uniform(0,1)
            first_child = female['solution'] + ((2*r)-1)*alpha*np.abs(ub-lb)
            females = [
                {'solution':first_child,
                'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(first_child))))},
                copy.deepcopy(female)
            ]
        new_female = sorted(females, key=lambda x: x["fitness"], reverse = True)
        return new_female[0]
    
    def small_male_movement(self,small_males,big_males,mlipir_rate):
        new_small_males = []
        for i in range(len(small_males)):
            sum_wij = 0
            for j in range(len(big_males)):
                r1 = np.random.uniform(0,1,size=self.m)
                if random.uniform(0,1) < mlipir_rate:
                    wij = r1*(big_males[j]['solution'] - small_males[i]['solution'])
                else:
                    wij = 0
                sum_wij = sum_wij + wij
            new_small_male = small_males[i]['solution'] + sum_wij
            fitness_new_small_male = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(new_small_male))))
            new_small_males.append({'solution':new_small_male,
                                 'fitness':fitness_new_small_male})
        
        new_small_males = new_small_males + small_males
        new_small_males = sorted(new_small_males, key=lambda x: x["fitness"], reverse = True)
        return new_small_males[:self.n_small_male]
    
    def first_phase(self):
        best_komodo = {'solution':None,
                      'fitness':0}
        
        idem_counter = 0
        
        komodo_ls = self.generate_population_first_phase()
        big_males,female,small_males = self.initiate_cluster_komodo(komodo_ls)
                
        for i in range(self.max_iter):
            #big males movement
            big_males = self.big_male_movement(big_males)
            
            #female movement
            female = self.female_movement(female,big_males)
            
            #small males movement
            small_males = self.small_male_movement(small_males,big_males,mlipir_rate = self.d1)
            
            #recluster
            komodo_ls = big_males + [female] + small_males
            big_males,female,small_males = self.re_cluster_komodo(komodo_ls)
            
            if big_males[0]['fitness'] > best_komodo['fitness']:
                best_komodo = big_males[0]
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_komodo
            
        return best_komodo
    
    def second_phase(self,best_so_far_komodo):
        best_komodo = best_so_far_komodo
        
        f_hist = [best_komodo['fitness'],best_komodo['fitness'],best_komodo['fitness']]
        
        idem_counter = 0
        komodo_ls = self.generate_population_second_phase(best_komodo, initiate_population = True)
        
        self.n_big_male = int(np.ceil((self.p2*self.n2)-1))
        self.n_female = 1
        self.n_small_male = int(self.n2 - (self.n_big_male+self.n_female))
        big_males,female,small_males = self.initiate_cluster_komodo(komodo_ls)
        
        for i in range(self.max_iter):
            #big males movement
            big_males = self.big_male_movement(big_males)
            
            #female movement
            female = self.female_movement(female,big_males)
            
            #small males movement
            small_males = self.small_male_movement(small_males,big_males,mlipir_rate = self.d2)
            
            #recluster
            komodo_ls = big_males + [female] + small_males
            
            f_hist = [best_komodo['fitness']] + f_hist
            if len(f_hist) > 3:
                f_hist = f_hist[:3]
            
            d_f1 = np.abs(f_hist[0]-f_hist[1])/f_hist[0]
            d_f2 = np.abs(f_hist[1]-f_hist[2])/f_hist[1]
            if d_f1 > 0 and d_f2 > 0:
                if self.n2 > self.min_n2:
                    self.n2 = self.n2 - self.a
                    komodo_ls = komodo_ls[:self.n2]
            else:
                if self.n2 < self.max_n2:
                    self.n2 = self.n2 + self.a
                    new_komodo_ls = self.generate_population_second_phase(best_komodo, initiate_population = False)
                    new_komodo_dict = [{'solution':sol,
                                        'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(sol))))} for sol in new_komodo_ls]
                    komodo_ls = komodo_ls + new_komodo_dict
                    
            self.n_big_male = int(np.ceil((self.p2*self.n2)-1))
            self.n_female = 1
            self.n_small_male = int(self.n2 - (self.n_big_male+self.n_female))
            big_males,female,small_males = self.re_cluster_komodo(komodo_ls)
            
            if big_males[0]['fitness'] > best_komodo['fitness']:
                best_komodo = big_males[0]
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_komodo
            
        return best_komodo
    
    def construct_solution(self):
        best_komodo = self.second_phase(self.first_phase())
        
        best_solution = self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_komodo['solution'])))
        best_fitness = best_komodo['fitness']
        
        return best_solution,best_fitness
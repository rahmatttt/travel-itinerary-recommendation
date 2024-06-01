# DISCRETE KOMODO ALGORITHM 
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class DKA_VRP(object):
    def __init__(self,n = 5,p=0.5,smep=5,max_iter = 1000,max_idem = 20,random_state=None):
        self.db = ConDB()
        
        # parameter setting
        self.n = n #number of komodo
        self.p = p #portion of big male komodo
        self.smep = smep #small male exploration probability (0,10)
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
    
    def set_model(self,tour,hotel,timematrix,init_solution=[],travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        #initiate model
        self.tour = copy.deepcopy(tour)
        self.hotel = copy.deepcopy(hotel)
        self.travel_days = travel_days
        self.hotel.depart_time = depart_time
        self.max_travel_time = max_travel_time
        self.timematrix = copy.deepcopy(timematrix)
        
        self.n_big_male = int(np.floor((self.p*self.n)-1))
        self.n_female = 1
        self.n_small_male = int(self.n - (self.n_big_male+self.n_female))
        
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
            penalty_per_day = [max(self.diff_second_between_time(i[-1],self.max_travel_time),0) for i in waktu_ls]
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
    
    def swap_operator(self,komodo):
        k = copy.deepcopy(komodo)
        pos1,pos2 = random.sample(range(len(k)),2)
        k[pos1],k[pos2] = k[pos2],k[pos1]
        return k
    
    def cluster_komodo(self,komodo_ls):
        komodo_dict = [{'solution':i,
                       'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(i)))} for i in komodo_ls]
        komodo_dict = sorted(komodo_dict, key=lambda x: x["fitness"], reverse = True)
        big_males = copy.deepcopy(komodo_dict[:self.n_big_male])
        female = copy.deepcopy(komodo_dict[self.n_big_male])
        small_males = copy.deepcopy(komodo_dict[self.n_big_male+self.n_female:])
        return big_males,female,small_males
    
    def distance_between_komodo(self,komodo,komodo_target):
        #distance between komodo is the edges in komodo_target that not exist in komodo
        k = [i._id for i in komodo]
        k_edge = [[k[i-1],k[i]] for i in range(1,len(k))]
        
        k_target = [i._id for i in komodo_target]
        k_target_edge = [[k_target[i-1],k_target[i]] for i in range(1,len(k_target))]
        
        distance = [i for i in k_target_edge if i not in k_edge]
        return distance
    
    def similar_edge_between_komodo(self,komodo,komodo_target):
        #distance between komodo is the edges in komodo_target that not exist in komodo
        k = [i._id for i in komodo]
        k_edge = [[k[i-1],k[i]] for i in range(1,len(k))]
        
        k_target = [i._id for i in komodo_target]
        k_target_edge = [[k_target[i-1],k_target[i]] for i in range(1,len(k_target))]
        
        similar = [i for i in k_target_edge if i in k_edge]
        return similar
    
    def find_segment(self,komodo,komodo_target,edge_target):
        k = [i._id for i in komodo]
        
        k_target = [i._id for i in komodo_target]
        k_target_edge = [[k_target[i-1],k_target[i]] for i in range(1,len(k_target))]
        if k.index(edge_target[0]) < k.index(edge_target[1]):
            x = edge_target[0]
            pos_x = k.index(edge_target[0])
            
            y = edge_target[1]
            pos_y = k.index(edge_target[1])
        else:
            x = edge_target[1]
            pos_x = k.index(edge_target[1])
            
            y = edge_target[0]
            pos_y = k.index(edge_target[0])
            
        # segment x and a
        if pos_x == 0:
            segment_x = [pos_x,pos_x+1] #start,end in [start:end] => in index 0 only
            segment_a = -1
        else:
            stop_point = pos_x
            for i in range(pos_x-1,-1,-1):
                if [k[i],k[i+1]] in k_target_edge:
                    stop_point = stop_point - 1
                else:
                    break
            segment_x = [stop_point,pos_x+1] #stop point until pos_x
            segment_a = [0,stop_point]
        
        # segment y and c
        if pos_y == len(k)-1:
            segment_y = [pos_y,pos_y+1] #start,end in [start:end] => in index -1 only
            segment_c = -1
        else:
            stop_point = pos_y
            for i in range(pos_y,len(k)-1):
                if [k[i],k[i+1]] in k_target_edge:
                    stop_point = stop_point + 1
                else:
                    break
            segment_y = [pos_y,stop_point+1] #pos_y until stop_point
            segment_c = [stop_point+1,len(k)]
        
        #segment b
        segment_b = -1 if segment_y[0]-(segment_x[1]-1) == 1 else [segment_x[1],segment_y[0]]

        return segment_a,segment_x,segment_b,segment_y,segment_c
    
    def edge_construction(self,komodo,komodo_target):
        # find distance
        distances = self.distance_between_komodo(komodo,komodo_target)
        
        if len(distances) > 0:
            selected_distance = distances[random.randint(0,len(distances)-1)]

            # create segment
            segment_a,segment_x,segment_b,segment_y,segment_c = self.find_segment(komodo,komodo_target,selected_distance)
            nodes_a = [] if segment_a == -1 else copy.deepcopy(komodo[segment_a[0]:segment_a[1]])
            nodes_x = copy.deepcopy(komodo[segment_x[0]:segment_x[1]])
            nodes_b = [] if segment_b == -1 else copy.deepcopy(komodo[segment_b[0]:segment_b[1]])
            nodes_y = copy.deepcopy(komodo[segment_y[0]:segment_y[1]])
            nodes_c = [] if segment_c == -1 else copy.deepcopy(komodo[segment_c[0]:segment_c[1]])

            if nodes_b != []:
                #operator 1
                result1 = nodes_a + nodes_b + nodes_x + nodes_y + nodes_c
                fitness1 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result1)))

                #operator 2
                result2 = nodes_a + nodes_b + nodes_y[::-1] + nodes_x[::-1] + nodes_c
                fitness2 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result2)))

                #operator 3
                result3 = nodes_a + nodes_y[::-1] + nodes_x[::-1] + nodes_b + nodes_c
                fitness3 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result3)))

                #operator 4
                result4 = nodes_a + nodes_x + nodes_y + nodes_b + nodes_c
                fitness4 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result4)))
                
                best_operator = np.argmax([fitness1,fitness2,fitness3,fitness4]) + 1
            else:
                #operator 1
                result1 = nodes_a + nodes_y[::-1] + nodes_x[::-1] + nodes_c
                fitness1 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result1)))
                
                #operator 2
                result2 = nodes_a + nodes_x + nodes_y + nodes_c
                result2[segment_x[1]],result2[segment_y[0]] = result2[segment_y[0]],result2[segment_x[1]]
                fitness2 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result2)))
                best_operator = np.argmax([fitness1,fitness2]) + 1
            
            if best_operator == 1:
                return result1,fitness1
            elif best_operator == 2:
                return result2,fitness2
            elif best_operator == 3:
                return result3,fitness3
            else:
                return result4,fitness4
        else:
            result = komodo
            fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(komodo)))
            return result,fitness
    
    def edge_destruction(self,komodo,komodo_target):
        # find distance
        similars = self.similar_edge_between_komodo(komodo,komodo_target)
        selected_edge = similars[random.randint(0,len(distances)-1)]
        
        # create segment
        segment_a,segment_x,segment_b,segment_y,segment_c = self.find_segment(komodo,komodo_target,selected_edge)
        nodes_a = [] if segment_a == -1 else copy.deepcopy(komodo[segment_a[0]:segment_a[1]])
        nodes_x = copy.deepcopy(komodo[segment_x[0]:segment_x[1]])
        nodes_y = copy.deepcopy(komodo[segment_y[0]:segment_y[1]])
        nodes_c = [] if segment_c == -1 else copy.deepcopy(komodo[segment_c[0]:segment_c[1]])
        
        #operator 1 (nodes_a != [])
        if nodes_a != []:
            result1 = nodes_x + nodes_a + nodes_y + nodes_c
            fitness1 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result1)))
        else:
            fitness1 = -999 #eliminate this operator
            
        #operator 2
        result2 = nodes_a + nodes_y + nodes_x + nodes_c
        fitness2 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result2)))
        
        #operator 3
        result3 = nodes_a + nodes_x[::-1] + nodes_y[::-1] + nodes_c
        fitness3 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result3)))
        
        #operator 4 (len(nodes_x)>1)
        if len(nodes_X)>1:
            result4 = nodes_a + nodes_x[::-1] + nodes_y + nodes_c
            fitness4 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result4)))
        else:
            fitness4 = -999 #eliminate this operator
            
        #operator 5 (len(nodes_y)>1)
        if len(nodes_y)>1:
            result5 = nodes_a + nodes_x + nodes_y[::-1] + nodes_c
            fitness5 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result5)))
        else:
            fitness5 = -999 #eliminate this operator
        
        #operator 6 (nodes_c != [])
        if nodes_c != []:
            result6 = nodes_a + nodes_x + nodes_c + nodes_y
            fitness6 = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(result6)))
        else:
            fitness6 = -999 #eliminate this operator
            
        best_operator = np.argmax([fitness1,fitness2,fitness3,fitness4,fitness5,fitness6]) + 1
        if best_operator == 1:
            return result1,fitness1
        elif best_operator == 2:
            return result2,fitness2
        elif best_operator == 3:
            return result3,fitness3
        elif best_operator == 4:
            return result4,fitness4
        elif best_operator == 5:
            return result5,fitness5
        else:
            return result6,fitness6
    
    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        
        idem_counter = 0
        komodo_ls = [random.sample(self.tour,len(self.tour)) for i in range(self.n)]
        
        big_males,female,small_males = self.cluster_komodo(komodo_ls)
        
        for i in range(self.max_iter):
                        
            #big_males movement
            for j in range(len(big_males)):
                new_big_males = []
                for k in range(len(big_males)):
                    if j != k:
                        if big_males[j]['fitness']<big_males[k]['fitness'] or random.uniform(0,1) < 0.5:
                            new_k,new_fitness = self.edge_construction(big_males[j]['solution'],big_males[k]['solution'])
                            new_big_males.append({'solution':new_k,'fitness':new_fitness})
                        else:
                            new_k,new_fitness = self.edge_destruction(big_males[j]['solution'],big_males[k]['solution'])
                            new_big_males.append({'solution':new_k,'fitness':new_fitness})
                        
                        new_big_males = sorted(new_big_males, key=lambda x: x["fitness"], reverse = True)
                        if i == 0:
                            if big_males[0]['fitness'] < new_big_males[0]['fitness']:
                                big_males[j] = new_big_males[0]
                        else:
                            big_males[j] = new_big_males[0]
            
            #female movement
            if random.uniform(0,1) < 0.5:
                result1,fitness1 = self.edge_construction(female['solution'],big_males[0]['solution'])
                result2,fitness2 = self.edge_construction(big_males[0]['solution'],female['solution'])
                if fitness1 > fitness2:
                    female['solution'] = result1
                    female['fitness'] = fitness1
                else:
                    female['solution'] = result2
                    female['fitness'] = fitness2
            else:
                female['solution'] = self.swap_operator(female['solution'])
                female['fitness'] = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(female['solution'])))
            
            #small males movement
            for j in range(len(small_males)):
                new_small_males = []
                for k in range(len(big_males)):
                    if random.randint(0,10) <= self.smep:
                        new_k = self.swap_operator(small_males[j]['solution'])
                        new_fitness = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(small_males[j]['solution'])))
                        new_small_males.append({'solution':new_k,'fitness':new_fitness})
                    else:
                        new_k,new_fitness = self.edge_construction(small_males[j]['solution'],big_males[k]['solution'])
                        new_small_males.append({'solution':new_k,'fitness':new_fitness})

                    new_small_males = sorted(new_small_males, key=lambda x: x["fitness"], reverse = True)
                    small_males[j] = new_small_males[0]
            
            # update new komodo_ls
            komodo_ls = big_males + [female] + small_males
            komodo_ls = [i['solution'] for i in komodo_ls]
            
            # update cluster komodo
            big_males,female,small_males = self.cluster_komodo(komodo_ls)
            
            # check best solution
            if best_fitness < big_males[0]['fitness']:
                best_solution = self.solution_list_of_nodes_to_dict(self.split_itinerary(big_males[0]['solution']))
                best_fitness = big_males[0]['fitness']
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness
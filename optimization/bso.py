from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class BSO_VRP(object):
    def __init__(self,p1 = 0.7,p2 = 0.7,p3 = 0.9,p4 = 0.3,max_iter = 100,max_idem = 15,two_opt_method="best",random_state=None):
        self.db = ConDB()
        
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
        
        if len(self.rest_nodes) == 0:
            self.p3 = 0 #no interchange with rest nodes
            
        if len(self.init_solution)<=1 and len(self.rest_nodes)>0:
            self.p3 = 1 #no interchange between clusters
        elif len(self.init_solution)<=1 and len(self.rest_nodes)==0:
            self.p1 = 1 #no two interchange (only two opt)
    
    def set_max_iter(self,max_iter):
        self.max_iter = max_iter

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
    
    def generate_init_solution(self):
        solution = list(copy.deepcopy(self.tour))
        random.shuffle(solution)
        
        final_solution = [] #2d list of nodes
        day = 1
        tabu_nodes = [] #list of tabu node's id
        while day <= self.travel_days:
            current_node = self.hotel
            day_solution = [] #list of nodes
            next_node_candidates = [node for node in solution if node._id not in tabu_nodes]
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
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node._id for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.tour if node._id not in solution_id]
        return rest_nodes
    
    def random_clustering(self,solution_nodes):
        random.shuffle(solution_nodes)
        cluster_a = solution_nodes[:len(solution_nodes)//2]
        cluster_b = solution_nodes[len(solution_nodes)//2:]
        return cluster_a,cluster_b
    
    def find_center_index(self,cluster_nodes):
        return np.argmax([self.MAUT(self.solution_list_of_nodes_to_dict([i])) for i in cluster_nodes])
    
    def two_opt(self,itinerary):
        solution = [self.hotel] + itinerary + [self.hotel]
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict([solution[1:-1]]),use_penalty=False)
        n = len(solution)
        
        if self.two_opt_method == "first": #first improvement
        	improved = False
	        for i in range(1,n-2):
	            for j in range(i+2,n-1):
	                if j-i == 1:
	                    continue
	                temp_solution = copy.deepcopy(solution)
	                temp_solution[i+1:j+1] = reversed(temp_solution[i+1:j+1])
	                temp_solution[1:-1],status = self.reset_depart_arrive_time(temp_solution[1:-1])
	                if status:
	                    temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp_solution[1:-1]]),use_penalty=False)
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
	                    temp_solution[1:-1],status = self.reset_depart_arrive_time(temp_solution[1:-1])
	                    if status:
	                        temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp_solution[1:-1]]),use_penalty=False)
	                        if temp_fitness > fitness:
	                            solution = copy.deepcopy(temp_solution)
	                            fitness = temp_fitness
	                            improved = True

        return solution[1:-1] #return best or return unchanged
    
    def two_interchange(self,itinerary1,itinerary2, rest_nodes = False):
        operator = [(0,1),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        solution1 = copy.deepcopy(itinerary1)
        solution2 = copy.deepcopy(itinerary2)
        if rest_nodes == False:
            fitness = self.MAUT(self.solution_list_of_nodes_to_dict([solution1,solution2]),use_penalty=False)
        else:
            fitness = self.MAUT(self.solution_list_of_nodes_to_dict([solution1]),use_penalty=False)
        
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
                    
                    if op[0]>1 and len(solution1)>1:
                        if i == 1:
                            temp1[i+1] = [temp1[i+1][1]]
                        elif i == len(sol1)-1:
                            temp1[i-1] = [temp1[i-1][0]]
                        else:
                            temp1[i-1] = [temp1[i-1][0]]
                            temp1[i+1] = [temp1[i+1][1]]
                    
                    if op[1]>1 and len(solution2)>1:
                        if j == 1:
                            temp2[j+1] = [temp2[j+1][1]]
                        elif j == len(sol2)-1:
                            temp2[j-1] = [temp2[j-1][0]]
                        else:
                            temp2[j-1] = [temp2[j-1][0]]
                            temp2[j+1] = [temp2[j+1][1]]
                    
                    #flatten
                    temp1 = sum(temp1,[])
                    temp2 = sum(temp2,[])
                    
                    #reset depart and arrive time
                    temp1,status1 = self.reset_depart_arrive_time(temp1)
                    if rest_nodes == False:
                        temp2,status2 = self.reset_depart_arrive_time(temp2)
                    else:
                        status2 = True
                    
                    if status1 and status2:
                        #count fitness
                        if rest_nodes == False and len(temp2) > 0 and len(temp1) > 0:
                            temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp1,temp2]),use_penalty=False)
                        elif rest_nodes == False and len(temp2) > 0:
                        	temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp2]),use_penalty=False)
                        elif rest_nodes == False and len(temp1) > 0:
                        	temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp1]),use_penalty=False)
                        elif rest_nodes == True and len(temp1) > 0:
                            temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp1]),use_penalty=False)
                        else:
                        	temp_fitness = 0
                    
                        if temp_fitness > fitness:
                            return temp1,temp2 #first improvement
            
            return itinerary1,itinerary2 #if no improvement
    
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
            if random.uniform(0,1) < self.p1:
                #2-opt
                cluster_id = random.randint(1,2) if len(solution)>1 else 1
                
                # randomization p2
                if random.uniform(0,1) < self.p2:
                    #center
                    itinerary_id = clusters[cluster_id]['center']
                else:
                    #random itienrary
                    itinerary_id = random.randint(0,len(clusters[cluster_id]['list'])-1)
                clusters[cluster_id]['list'][itinerary_id] = self.two_opt(clusters[cluster_id]['list'][itinerary_id])
            else:
                #2-interchange
                if random.uniform(0,1) < self.p3:
                    #interchange a cluster with rest nodes
                    cluster_id = random.randint(1,2) if len(clusters)>1 else 1
                    
                    #randomization p4
                    if random.uniform(0,1) < self.p4:
                        itinerary_id = clusters[cluster_id]['center']
                    else:
                        itinerary_id = random.randint(0,len(clusters[cluster_id]['list'])-1)
                    clusters[cluster_id]['list'][itinerary_id],self.rest_nodes = self.two_interchange(clusters[cluster_id]['list'][itinerary_id],self.rest_nodes,rest_nodes=True)
                else:
                    #interchange between two clusters
                    #randomization p4
                    if random.uniform(0,1) < self.p4:
                        itinerary_id_1 = clusters[1]['center']
                        itinerary_id_2 = clusters[2]['center']
                    else:
                        itinerary_id_1 = random.randint(0,len(clusters[1]['list'])-1)
                        itinerary_id_2 = random.randint(0,len(clusters[2]['list'])-1)
                    clusters[1]['list'][itinerary_id_1],clusters[2]['list'][itinerary_id_2] = self.two_interchange(clusters[1]['list'][itinerary_id_1],clusters[2]['list'][itinerary_id_2],rest_nodes=False)
            
            #merge cluster
            solution = []
            for cluster_id in clusters:
                solution.extend(clusters[cluster_id]['list'])
            
            solution = [sol for sol in solution if len(sol)>0]
            solution_dict = self.solution_list_of_nodes_to_dict(solution)
            new_fitness = self.MAUT(solution_dict)
            if new_fitness > fitness:
                fitness = new_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return solution,solution_dict,fitness
        
        return solution,solution_dict,fitness

class BSO_TSP(object):
    def __init__(self,p1 = 0.5,max_iter = 100,max_idem = 20, two_opt_method = "best", random_state = None):
        self.db = ConDB()
        
        # parameter setting
        self.p1 = p1 #less than: 2-opt, more than: 2-interchange
        self.max_iter = max_iter
        self.max_idem = max_idem

        self.two_opt_method = two_opt_method
        
        # set initial solution
        self.init_solution = [] #1D list of nodes, [node1,node2,....]

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
        self.max_waktu_tsp = None
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
        
        self.max_waktu_tsp = 0  # Initialize the total waktu to 0

        for source, destinations in self.timematrix.items():
            for destination, values in destinations.items():
                self.max_waktu_tsp += values['waktu']

        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        else:
            self.init_solution = self.generate_init_solution()
    
    def set_max_iter(self,max_iter):
        self.max_iter = max_iter

    def set_init_solution(self,init_solution):
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution

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
        score_tarif = (1-self.min_max_scaler(self.min_tarif,self.max_tarif,sum_tarif)) * self.degree_tarif
        
        #waktu
        sum_waktu = solutions['waktu']
        score_waktu = (1-self.min_max_scaler(self.min_waktu,self.max_waktu_tsp,sum_waktu))*self.degree_waktu
        
        #MAUT
        pembilang = score_rating+score_tarif+score_waktu
        penyebut = self.degree_rating+self.degree_tarif+self.degree_waktu
        maut = pembilang/penyebut
        return maut
    
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
    
    def create_solution_dict_TSP(self,solution_nodes,start_node=None,end_node=None):
        solution = copy.deepcopy(solution_nodes)
        current_node = self.hotel if start_node == None else start_node
        final_solution = {"index":[],"waktu":0,"rating":[],"tarif":[]}
        for i in range(len(solution)):
            final_solution['index'].append(solution[i]._id)
            final_solution['waktu'] += self.timematrix[current_node._id][solution[i]._id]['waktu']+solution[i].waktu_kunjungan
            final_solution['rating'].append(solution[i].rating)
            final_solution['tarif'].append(solution[i].tarif)
            current_node = solution[i]
        if current_node._id != self.hotel._id:
            last_node = copy.deepcopy(self.hotel) if end_node == None else end_node
            final_solution['waktu'] += self.timematrix[current_node._id][last_node._id]['waktu']

        return final_solution
    
    def generate_init_solution(self):
        solution = list(copy.deepcopy(self.tour))    
        random.shuffle(solution)
        return solution
        
    def two_opt(self,itinerary,start_node=None,end_node=None):
        start_node = self.hotel if start_node == None else start_node
        end_node = self.hotel if end_node == None else end_node
        solution = [start_node] + itinerary + [end_node]
        fitness = self.MAUT_TSP(self.create_solution_dict_TSP(solution[1:-1]))
        n = len(solution)
        if self.two_opt_method == "first":
            improved = False
            for i in range(1,n-2):
                for j in range(i+2,n-1):
                    if j-i == 1:
                        continue
                    temp_solution = copy.deepcopy(solution)
                    temp_solution[i+1:j+1] = reversed(temp_solution[i+1:j+1])
                    temp_fitness = self.MAUT_TSP(self.create_solution_dict_TSP(temp_solution[1:-1]))
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
                    for j in range(1+2,n-1):
                        if j-1 == 1:
                            continue
                        temp_solution = copy.deepcopy(solution)
                        temp_solution[i+1:j+1] = reversed(temp_solution[i+1:j+1])
                        temp_fitness = self.MAUT_TSP(self.create_solution_dict_TSP(temp_solution[1:-1]))
                        if temp_fitness > fitness:
                            solution = copy.deepcopy(temp_solution)
                            fitness = temp_fitness
                            improved = True

        return solution[1:-1] #best improvement
    
    def two_interchange(self,itinerary1,itinerary2):
        operator = [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        solution1 = copy.deepcopy(itinerary1)
        solution2 = copy.deepcopy(itinerary2)
        fitness = self.MAUT_TSP(self.create_solution_dict_TSP(solution1+solution2))
        
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
                    
                    if op[0]>1 and len(solution1)>1:
                        if i == 1:
                            temp1[i+1] = [temp1[i+1][1]]
                        elif i == len(sol1)-1:
                            temp1[i-1] = [temp1[i-1][0]]
                        else:
                            temp1[i-1] = [temp1[i-1][0]]
                            temp1[i+1] = [temp1[i+1][1]]
                    
                    if op[1]>1 and len(solution2)>1:
                        if j == 1:
                            temp2[j+1] = [temp2[j+1][1]]
                        elif j == len(sol2)-1:
                            temp2[j-1] = [temp2[j-1][0]]
                        else:
                            temp2[j-1] = [temp2[j-1][0]]
                            temp2[j+1] = [temp2[j+1][1]]
                    
                    #flatten
                    temp1 = sum(temp1,[])
                    temp2 = sum(temp2,[])
                    
                    #count fitness
                    temp_fitness = self.MAUT_TSP(self.create_solution_dict_TSP(temp1+temp2))
                    
                    if temp_fitness > fitness:
                        return temp1,temp2 #first improvement
            
            return itinerary1,itinerary2 #if no improvement
    
    def TSP(self):
        fitness = 0
        solution = copy.deepcopy(self.init_solution)
        idem_counter = 0
        for i in range(self.max_iter):
            #clustering
            clusters = {}
            clusters[1] = copy.deepcopy(solution[:len(solution)//2])
            clusters[2] = copy.deepcopy(solution[len(solution)//2:])

            #randomization p1
            if random.uniform(0,1) < self.p1:
                #2-opt
                cluster_id = random.randint(1,2)
                if cluster_id == 1:
                    start_node = self.hotel
                    end_node = clusters[cluster_id+1][0]
                else:
                    start_node = clusters[cluster_id-1][-1]
                    end_node = self.hotel
                clusters[cluster_id] = self.two_opt(clusters[cluster_id],start_node,end_node)
            else:
                #2-interchange between two clusters
                clusters[1],clusters[2] = self.two_interchange(clusters[1],clusters[2])
            
            #merge cluster
            solution = []
            for cluster_id in clusters:
                solution.extend(clusters[cluster_id])
            
            new_fitness = self.MAUT_TSP(self.create_solution_dict_TSP(solution))
            if new_fitness > fitness:
                fitness = new_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    solution,fitness
        
        return solution,fitness
    
    def construct_solution(self):
        solution,fitness = self.TSP()
        day = 1
        final_solution = []
        final_solution_dict = []
        tabu_nodes = []
        while day <= self.travel_days:
            current_node = self.hotel
            day_solution = []
            day_solution_dict = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}
            next_node_candidates = [node for node in solution if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                time_needed = self.time_to_second(current_node.depart_time)+self.timematrix[current_node._id][next_node_candidates[i]._id]["waktu"]+next_node_candidates[i].waktu_kunjungan
                if time_needed >= self.time_to_second(next_node_candidates[i].jam_tutup):
                    continue
                elif self.next_node_check(current_node,next_node_candidates[i]):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i])
                    day_solution.append(next_node_candidates[i])
                    day_solution_dict['index'].append(next_node_candidates[i]._id)
                    day_solution_dict['waktu'].append(next_node_candidates[i].arrive_time)
                    day_solution_dict['rating'].append(next_node_candidates[i].rating)
                    day_solution_dict['tarif'].append(next_node_candidates[i].tarif)
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
                else:
                    break
            if current_node._id != self.hotel._id:
                self.hotel = self.set_next_node_depart_arrive_time(current_node,self.hotel)
                day_solution_dict['waktu'].append(self.hotel.arrive_time)
            
            if len(day_solution_dict['index']) > 0:
                final_solution.append(day_solution)
                final_solution_dict.append(day_solution_dict)
            
            if len(tabu_nodes) == len(self.tour):
                break
            
            day += 1
        
        final_fitness = self.MAUT(final_solution_dict)
        return final_solution,final_solution_dict,final_fitness
    
    
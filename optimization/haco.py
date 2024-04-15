from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class HACO_VRP(object):
    def __init__(self,alpha = 1,beta = 4,q0 = 0.4,q1=0.6,q2=0.3,q3=0.6,init_pheromone = 0.5,rho = 0.7,num_ant = 30,max_iter = 100,max_idem=50,random_state=None):
        self.db = ConDB()
        
        #parameter setting
        self.alpha = alpha #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in HACO transition rule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration HACO
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
    
    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix):
        max_pos = np.argmax([(local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha)*(self.MAUT_between_two_nodes(current_node,next_node)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node
    
    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val
        
        #probability
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            node_prob = 0.0001 if node_prob == 0 else node_prob
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
    
    def pheromone_update(self,pheromone_matrix):
        for node in self.timematrix:
            for next_node in self.timematrix[node]:
                pheromone = self.timematrix[node][next_node]['pheromone']
                sum_delta = sum(pheromone_matrix[node][next_node]['delta'])
                pheromone = ((1-self.rho)*pheromone)+sum_delta
                self.timematrix[node][next_node]['pheromone'] = pheromone
    
    def init_delta_to_pheromone_matrix(self,pheromone_matrix):
        for i in pheromone_matrix:
            for j in pheromone_matrix[i]:
                pheromone_matrix[i][j]['delta'] = []
        return pheromone_matrix
    
    def add_delta_to_pheromone_matrix(self,day_solution,pheromone_matrix,fitness):
        nodes = [self.hotel._id] + day_solution['index'] + [self.hotel._id]
        node_edges = [(nodes[idx-1],nodes[idx]) for idx in range(1,len(nodes))]
        for i,j in node_edges:
            pheromone_matrix[i][j]['delta'].append(fitness)
        
        return pheromone_matrix
    
    def mutation(self,solution_flatten_nodes):
        solution = copy.deepcopy(solution_flatten_nodes)
        
        if random.uniform(0,1) <= self.q2:
            #interchange
            for i in range(random.randint(1,len(solution)-1)):
                node1 = random.randint(0,len(solution)-1)
                node2 = random.randint(node1+1,len(solution)-1) if node1 < len(solution)-1 else random.randint(0,node1-1)
                
                solution[node1],solution[node2] = solution[node2],solution[node1]
                
        elif random.uniform(0,1) > self.q2 and random.uniform(0,1) <= self.q3:
            #shift
            node1 = random.randint(0,len(solution)-1)
            node2 = random.randint(node1+1,len(solution)-1) if node1 < len(solution)-1 else random.randint(0,node1-1)
            
            if node1 < node2:
                solution.insert(node1,solution.pop(node2))
            else:
                solution.insert(node2,solution.pop(node1))
            
        else:
            #inverse
            node1 = random.randint(0,len(solution)-1)
            node2 = random.randint(node1+1,len(solution)-1) if node1 < len(solution)-1 else random.randint(0,node1-1)
            
            if node1 < node2:
                solution[node1:node2] = reversed(solution[node1:node2])
            else:
                solution[node2:node1] = reversed(solution[node2:node1])
        return solution
    
    def create_solution_dict(self,solution_flatten_nodes):
        solution = copy.deepcopy(solution_flatten_nodes)
        
        day = 1
        final_solution = []
        tabu_nodes = []
        while day <= self.travel_days:
            current_node = self.hotel
            day_solution = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}
            next_node_candidates = [node for node in solution if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                if self.next_node_check(current_node,next_node_candidates[i]):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i])
                    day_solution['index'].append(next_node_candidates[i]._id)
                    day_solution['waktu'].append(next_node_candidates[i].arrive_time)
                    day_solution['rating'].append(next_node_candidates[i].rating)
                    day_solution['tarif'].append(next_node_candidates[i].tarif)
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
            if current_node._id != self.hotel._id:
                self.hotel = self.set_next_node_depart_arrive_time(current_node,self.hotel)
                day_solution['waktu'].append(self.hotel.arrive_time)
            
            if len(day_solution['index']) > 0:
                final_solution.append(day_solution)
            
            day += 1
        
        return final_solution
    
    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node._id for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.tour if node._id not in solution_id]
        return rest_nodes
    
    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            local_pheromone_matrix = copy.deepcopy(self.timematrix)
            local_pheromone_matrix = self.init_delta_to_pheromone_matrix(local_pheromone_matrix)
            all_ant_solution = [] #[{"solution":[],"fitness":}]
            for ant in range(self.num_ant): #step
                ant_solution = []
                ant_solution_dict = []
                day = 1
                tabu_nodes = []
                while day<=self.travel_days:
                    current_node = self.hotel
                    ant_day_solution = []
                    ant_day_solution_dict = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}
                    
                    for pos in range(len(self.tour)+1):
                        #recheck next node candidates (perlu dicek jam sampainya apakah melebihi max time)
                        next_node_candidates = [node for node in self.tour if self.next_node_check(current_node,node)==True and node._id not in tabu_nodes]
                        
                        if len(next_node_candidates) > 0:
                            #transition rules
                            next_node = self.transition_rule(current_node,next_node_candidates,local_pheromone_matrix)
                            next_node = self.set_next_node_depart_arrive_time(current_node,next_node)

                            #change current node and delete it from available nodes
                            current_node = next_node
                            ant_day_solution.append(current_node)
                            ant_day_solution_dict['index'].append(current_node._id)
                            ant_day_solution_dict['rating'].append(current_node.rating)
                            ant_day_solution_dict['tarif'].append(current_node.tarif)
                            ant_day_solution_dict['waktu'].append(current_node.arrive_time)
                            tabu_nodes.append(current_node._id)
                        elif len(next_node_candidates) == 0 and current_node._id != self.hotel._id:
                            last_node = copy.deepcopy(self.hotel)
                            last_node = self.set_next_node_depart_arrive_time(current_node,last_node)
                            ant_day_solution_dict['waktu'].append(last_node.arrive_time)
                            break
                        else:
                            break
                    
                    if len(ant_day_solution_dict['index'])>0:
                        ant_solution.append(ant_day_solution)
                        ant_solution_dict.append(ant_day_solution_dict)
                    
                    if len(tabu_nodes) == len(self.tour):
                        break

                    day += 1
                
                fitness = self.MAUT(ant_solution_dict)
                all_ant_solution.append({"solution":copy.deepcopy(ant_solution_dict),"fitness":fitness})
                
                #mutation
                if random.uniform(0,1)<=self.q1:
                    rest_nodes = self.get_rest_nodes_from_solution(ant_solution)
                    ant_mutation = self.mutation(sum(ant_solution,[])+rest_nodes)
                    ant_mutation_dict = self.create_solution_dict(ant_mutation)
                    fitness_mutation = self.MAUT(ant_mutation_dict)
                    all_ant_solution.append({"solution":copy.deepcopy(ant_mutation_dict),"fitness":fitness_mutation})
            
            #get top num_ant from all_ant_solution
            all_ant_solution = sorted(all_ant_solution, key=lambda x: x["fitness"], reverse = True)
            all_ant_solution = all_ant_solution[:self.num_ant]
            
            best_found_solution = copy.deepcopy(all_ant_solution[0]["solution"])
            best_found_fitness = all_ant_solution[0]["fitness"]
            
            #add delta
            for solution in all_ant_solution:
                for day in solution["solution"]:
                    local_pheromone_matrix = self.add_delta_to_pheromone_matrix(day,local_pheromone_matrix,solution["fitness"])
            
            #pheromone update
            self.pheromone_update(local_pheromone_matrix)
            
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

class HACO_TSP(object):
    def __init__(self,alpha = 1,beta = 3,q0 = 0.8,q1=0.4,q2=0.3,q3=0.6,init_pheromone = 0.5,rho = 0.7,num_ant = 35,max_iter = 100,max_idem=50,random_state=None):
        self.db = ConDB()
        
        #parameter setting
        self.alpha = alpha #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in HACO transition rule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration HACO
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
        self.max_waktu_tsp = None
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
    
    def MAUT_between_two_nodes(self,current_node,next_node):
        score_rating = self.degree_rating * self.min_max_scaler(self.min_rating,self.max_rating,next_node.rating)
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.rating))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.timematrix[current_node._id][next_node._id]['waktu']))
        maut = (score_rating+score_tarif+score_waktu)/(self.degree_rating+self.degree_tarif+self.degree_waktu)
        return maut
    
    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix):
        max_pos = np.argmax([(local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha)*(self.MAUT_between_two_nodes(current_node,next_node)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node
    
    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val
        
        #probability
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            node_prob = 0.0001 if node_prob == 0 else node_prob
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
    
    def pheromone_update(self,pheromone_matrix):
        for node in self.timematrix:
            for next_node in self.timematrix[node]:
                pheromone = self.timematrix[node][next_node]['pheromone']
                sum_delta = sum(pheromone_matrix[node][next_node]['delta'])
                pheromone = ((1-self.rho)*pheromone)+sum_delta
                self.timematrix[node][next_node]['pheromone'] = pheromone
    
    def init_delta_to_pheromone_matrix(self,pheromone_matrix):
        for i in pheromone_matrix:
            for j in pheromone_matrix[i]:
                pheromone_matrix[i][j]['delta'] = []
        return pheromone_matrix
    
    def add_delta_to_pheromone_matrix(self,solution,pheromone_matrix,fitness):
        nodes = [self.hotel._id] + solution['index'] + [self.hotel._id]
        node_edges = [(nodes[idx-1],nodes[idx]) for idx in range(1,len(nodes))]
        for i,j in node_edges:
            pheromone_matrix[i][j]['delta'].append(fitness)

        return pheromone_matrix
    
    def mutation(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        
        if random.uniform(0,1) <= self.q2:
            #interchange
            for i in range(random.randint(1,len(solution)-1)):
                node1 = random.randint(0,len(solution)-1)
                node2 = random.randint(node1+1,len(solution)-1) if node1 < len(solution)-1 else random.randint(0,node1-1)
                
                solution[node1],solution[node2] = solution[node2],solution[node1]
                
        elif random.uniform(0,1) > self.q2 and random.uniform(0,1) <= self.q3:
            #shift
            node1 = random.randint(0,len(solution)-1)
            node2 = random.randint(node1+1,len(solution)-1) if node1 < len(solution)-1 else random.randint(0,node1-1)
            
            if node1 < node2:
                solution.insert(node1,solution.pop(node2))
            else:
                solution.insert(node2,solution.pop(node1))
            
        else:
            #inverse
            node1 = random.randint(0,len(solution)-1)
            node2 = random.randint(node1+1,len(solution)-1) if node1 < len(solution)-1 else random.randint(0,node1-1)
            
            if node1 < node2:
                solution[node1:node2] = reversed(solution[node1:node2])
            else:
                solution[node2:node1] = reversed(solution[node2:node1])
        return solution
    
    def create_solution_dict_TSP(self,solution_nodes):
        solution = copy.deepcopy(solution_nodes)
        
        day = 1
        current_node = self.hotel
        final_solution = {"index":[],"waktu":0,"rating":[],"tarif":[]}
        for i in range(len(solution)):
            final_solution['index'].append(solution[i]._id)
            final_solution['waktu'] += self.timematrix[current_node._id][solution[i]._id]['waktu']+solution[i].waktu_kunjungan
            final_solution['rating'].append(solution[i].rating)
            final_solution['tarif'].append(solution[i].tarif)
            current_node = solution[i]
        if current_node._id != self.hotel._id:
            last_node = copy.deepcopy(self.hotel)
            final_solution['waktu'] += self.timematrix[current_node._id][last_node._id]['waktu']

        return final_solution
    
    def TSP(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            local_pheromone_matrix = copy.deepcopy(self.timematrix)
            local_pheromone_matrix = self.init_delta_to_pheromone_matrix(local_pheromone_matrix)
            all_ant_solution = [] #[{"solution":[],"solution_dict":{},"fitness":}]
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
                        ant_solution_dict['waktu'] += self.timematrix[current_node._id][next_node._id]['waktu']
                        break
                    else:
                        break
                
                fitness = self.MAUT_TSP(ant_solution_dict)
                all_ant_solution.append({"solution":copy.deepcopy(ant_solution),
                                         "solution_dict":copy.deepcopy(ant_solution_dict),
                                         "fitness":fitness})
                
                #mutation
                if random.uniform(0,1)<=self.q1:
                    ant_mutation = self.mutation(ant_solution)
                    ant_mutation_dict = self.create_solution_dict_TSP(ant_mutation)
                    fitness_mutation = self.MAUT_TSP(ant_mutation_dict)
                    all_ant_solution.append({"solution":copy.deepcopy(ant_mutation),
                                             "solution_dict":copy.deepcopy(ant_mutation_dict),
                                             "fitness":fitness_mutation})
                
            #get top num_ant from all_ant_solution
            all_ant_solution = sorted(all_ant_solution, key=lambda x: x["fitness"], reverse = True)
            all_ant_solution = all_ant_solution[:self.num_ant]
            
            best_found_solution = copy.deepcopy(all_ant_solution[0]["solution"])
            best_found_fitness = all_ant_solution[0]["fitness"]
            
            # add delta
            for solution in all_ant_solution:
                local_pheromone_matrix = self.add_delta_to_pheromone_matrix(solution["solution_dict"],local_pheromone_matrix,solution["fitness"])
            
            #pheromone update
            self.pheromone_update(local_pheromone_matrix)
            
            #checking best vs best found
            if best_found_fitness >= best_fitness:
                best_fitness = best_found_fitness
                best_solution = copy.deepcopy(best_found_solution)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness
        
        return best_solution,best_fitness
    
    def construct_solution(self):
        day = 1
        final_solution = []
        tabu_nodes = []
        temp_tour = copy.deepcopy(self.tour)
        while day <= self.travel_days:
            solution,fitness = self.TSP()
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
                    self.tour = [node for node in self.tour if node._id != next_node_candidates[i]._id] 
                    current_node = next_node_candidates[i]
                else:
                    break
            if current_node._id != self.hotel._id:
                self.hotel = self.set_next_node_depart_arrive_time(current_node,self.hotel)
                day_solution['waktu'].append(self.hotel.arrive_time)
            
            if len(day_solution['index']) > 0:
                final_solution.append(day_solution)
            
            if len(self.tour) == 0:
                break

            day += 1
        
        final_fitness = self.MAUT(final_solution)
        self.tour = copy.deepcopy(temp_tour)
        return final_solution,final_fitness
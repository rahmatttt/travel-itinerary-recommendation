from optimization.koneksi import ConDB
import random
import copy
import time
import datetime
import numpy as np

class BaseOptimizer(object):
    """Attributes and methods shared by every optimizer in this folder."""

    def __init__(self,random_state = None,seed_numpy = False):
        self.db = ConDB()

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
            if seed_numpy:
                np.random.seed(random_state)

    def set_model(self,tour,hotel,timematrix,travel_days = 3,depart_time = datetime.time(8, 0, 0),max_travel_time = datetime.time(20, 0, 0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        #initiate model
        self.tour = copy.deepcopy(tour)
        self.hotel = copy.deepcopy(hotel)
        self.travel_days = travel_days
        self.hotel.depart_time = depart_time
        self.max_travel_time = max_travel_time
        self.timematrix = self.prepare_timematrix(timematrix)
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

    def prepare_timematrix(self,timematrix):
        return copy.deepcopy(timematrix)

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
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.tarif))
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
        """Random order of all POIs, as a 1D list of nodes."""
        solution = list(copy.deepcopy(self.tour))    
        random.shuffle(solution)
        return solution

    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node._id for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.tour if node._id not in solution_id]
        return rest_nodes

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

    def solution_list_of_float_to_nodes(self,solutions):
        index_order = np.flip(np.argsort(solutions))
        return list(np.array(self.tour)[index_order])


class BaseAntColony(BaseOptimizer):
    """Shared parts of the ant colony family (ACO, ACS, HACO, ACS-BSO)."""

    def prepare_timematrix(self,timematrix):
        return self.add_pheromone_to_timematrix(copy.deepcopy(timematrix))

    def add_pheromone_to_timematrix(self, timematrix):
        for i in timematrix:
            for j in timematrix[i]:
                timematrix[i][j]['pheromone'] = self.init_pheromone
        return timematrix

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

    def construct_solution(self):
        """Itinerary builder for the *_TSP classes: calls self.TSP() for every day, so the subclass must define TSP(). The *_VRP classes override this method."""
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


class BaseACS(BaseAntColony):
    """Ant Colony System rules shared by ACS and the ACS-BSO hybrid."""

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
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            node_prob = 0.0001 if node_prob == 0 else node_prob
            next_node_prob.append(node_prob)
        
        next_node = random.choices(next_node_candidates,next_node_prob,k=1)
        return next_node[0]

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

    def set_max_iter_acs(self,max_iter_acs):
        self.max_iter_acs = max_iter_acs


class BaseILS(BaseOptimizer):
    """Shared parts of the Iterated Local Search family (ILS, ILS-SA, ILS-TS, ILS-SA-TS)."""

    def construct_solution(self):
        idem_counter = 0
        
        best_solution,best_fitness = self.local_search(self.generate_init_solution())
        
        for i in range(self.max_iter):
            #perturbation
            candidate = self.perturbation(best_solution)
            
            #local search
            candidate,candidate_fitness = self.local_search(candidate)
            
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

    def generate_init_solution(self):
        """Random continuous vector with one value per POI, decoded by solution_list_of_float_to_nodes."""
        return np.random.uniform(-5.12,5.12,size=len(self.tour))

    def perturbation(self,solution_nodes):
        return solution_nodes + np.random.uniform(-self.strength,self.strength,size=len(solution_nodes))

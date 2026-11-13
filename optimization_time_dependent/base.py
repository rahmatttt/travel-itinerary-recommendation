from optimization.base import BaseOptimizer as _ParentBase
from optimization.base import BaseILS as _ParentBaseILS
from optimization.koneksi import ConDB
import random
import copy
import time
import datetime
import numpy as np

class BaseOptimizer(_ParentBase):
    """Attributes and methods shared by every optimizer in this folder."""

    def set_model(self,tour,hotel,timematrix,start_day = 5,travel_days = 3,depart_time = datetime.time(8, 0, 0),max_travel_time = datetime.time(20, 0, 0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        self.start_day = start_day # day index : monday - sunday => 0 - 6

    def MAUT_between_two_nodes(self,current_node,next_node,jam,day_type):
        score_rating = self.degree_rating * self.min_max_scaler(self.min_rating,self.max_rating,next_node.rating)
        score_tarif = self.degree_tarif * (1-self.min_max_scaler(self.min_tarif,self.max_tarif,next_node.tarif))
        score_waktu = self.degree_waktu * (1-self.min_max_scaler(self.min_waktu,self.max_waktu,self.timematrix[day_type][current_node._id][next_node._id][jam]))
        maut = (score_rating+score_tarif+score_waktu)/(self.degree_rating+self.degree_tarif+self.degree_waktu)
        return maut

    def next_node_check(self,current_node,next_node,jam,day_type):
        time_needed = self.time_to_second(current_node.depart_time)+self.timematrix[day_type][current_node._id][next_node._id][jam]+next_node.waktu_kunjungan
        time_limit = self.time_to_second(self.max_travel_time)
        if (time_needed <= time_limit) and (time_needed <= self.time_to_second(next_node.jam_tutup)):
            return True
        else:
            return False

    def set_next_node_depart_arrive_time(self,current_node,next_node,jam,day_type):
        arrive_time = self.time_to_second(current_node.depart_time)+self.timematrix[day_type][current_node._id][next_node._id][jam]
        arrive_time = max([arrive_time,self.time_to_second(next_node.jam_buka)])
        next_node.arrive_time = self.second_to_time(arrive_time)
        if next_node.tipe.lower() != "hotel":
            next_node.depart_time = self.second_to_time(arrive_time+next_node.waktu_kunjungan)
        return next_node

    def get_day_type(self,day):
        #input : day offset from start_day (0 = first travel day)
        #output: "weekend" or "weekday"
        day_index = (self.start_day + day) % 7
        return "weekend" if day_index >= 5 else "weekday"

    def reset_depart_arrive_time(self,itinerary,day = 0):
        #input : list of nodes itinerary (one day), day offset of that itinerary from start_day
        #output: updated depart and arrive time, if any unvalid time then return False
        solution = copy.deepcopy(itinerary)

        current_node = self.hotel
        day_type = self.get_day_type(day)
        for i in range(len(solution)):
            if self.next_node_check(current_node,solution[i],current_node.depart_time.hour,day_type):
                solution[i] = self.set_next_node_depart_arrive_time(current_node,solution[i],current_node.depart_time.hour,day_type)
                current_node = solution[i]
            else:
                return itinerary,False #cannot reset
        return solution,True

    def solution_list_of_nodes_to_dict(self, solutions, days = None):
        #days: day offset (from start_day) of each itinerary in solutions; None = consecutive days from start_day
        days = list(range(len(solutions))) if days is None else days
        solution_dict = []
        for day_pos,day in enumerate(solutions):
            day_type = self.get_day_type(days[day_pos])
            day_solution = {"index":[],"waktu":[self.hotel.depart_time],"rating":[],"tarif":[]}
            day_solution['index'] = [i._id for i in day]
            day_solution['rating'] = [i.rating for i in day]
            day_solution['tarif'] = [i.tarif for i in day]
            
            last_waktu = self.time_to_second(day[-1].depart_time) + self.timematrix[day_type][day[-1]._id][self.hotel._id][day[-1].depart_time.hour]
            day_solution['waktu'].extend([i.arrive_time for i in day] + [self.second_to_time(last_waktu)])
            
            solution_dict.append(day_solution)
        return solution_dict

    def split_itinerary(self,init_itinerary):
        final_solution = [] #2d list of nodes
        day = 1
        day_index = self.start_day
        tabu_nodes = [] #list of tabu node's id
        while day <= self.travel_days:
            current_node = self.hotel
            day_type = "weekend" if day_index >= 5 else "weekday"
            day_solution = [] #list of nodes
            next_node_candidates = [node for node in init_itinerary if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                if self.next_node_check(current_node,next_node_candidates[i],current_node.depart_time.hour,day_type):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i],current_node.depart_time.hour,day_type)
                    day_solution.append(next_node_candidates[i])
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
            
            if len(day_solution) > 0:
                final_solution.append(day_solution)
            
            if len(tabu_nodes) == len(self.tour):
                break
            
            day += 1
            day_index = day_index + 1 if day_index + 1 <= 6 else 0
        return final_solution


class BaseAntColony(BaseOptimizer):
    """Shared parts of the ant colony family (HACO, ACS-BSO) for time-dependent data.

    The timematrix here is timematrix[day_type][a][b][jam], so pheromone cannot be stored inside it
    like in optimization.base. It is kept in a separate matrix instead: self.pheromone_matrix[a][b]['pheromone']
    (one value per edge, independent of day_type and jam)."""

    def set_model(self,tour,hotel,timematrix,start_day = 5,travel_days = 3,depart_time = datetime.time(8, 0, 0),max_travel_time = datetime.time(20, 0, 0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,start_day = start_day,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        self.pheromone_matrix = self.create_pheromone_matrix()

    def create_pheromone_matrix(self):
        node_ids = [self.hotel._id] + [node._id for node in self.tour]
        return {i:{j:{'pheromone':self.init_pheromone} for j in node_ids if j != i} for i in node_ids}

    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix,jam,day_type):
        max_pos = np.argmax([(local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha)*(self.MAUT_between_two_nodes(current_node,next_node,jam,day_type)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node

    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix,jam,day_type):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node,jam,day_type)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val

        #probability
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node,jam,day_type)**self.beta
            node_prob = (pheromone_in_edge*heuristic_val)/sum_sample
            node_prob = 0.0001 if node_prob == 0 else node_prob
            next_node_prob.append(node_prob)

        next_node = random.choices(next_node_candidates,next_node_prob,k=1)
        return next_node[0]

    def transition_rule(self,current_node,next_node_candidates,local_pheromone_matrix,jam,day_type):
        q = random.uniform(0,1)
        if q <= self.q0: #exploitation
            next_node = self.exploitation(current_node,next_node_candidates,local_pheromone_matrix,jam,day_type)
        else: #exploration
            next_node = self.exploration(current_node,next_node_candidates,local_pheromone_matrix,jam,day_type)
        return next_node

    def pheromone_update(self,pheromone_matrix):
        for node in self.pheromone_matrix:
            for next_node in self.pheromone_matrix[node]:
                pheromone = self.pheromone_matrix[node][next_node]['pheromone']
                sum_delta = sum(pheromone_matrix[node][next_node]['delta'])
                pheromone = ((1-self.rho)*pheromone)+sum_delta
                self.pheromone_matrix[node][next_node]['pheromone'] = pheromone

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

    def create_solution_dict(self,solution_flatten_nodes):
        solution = copy.deepcopy(solution_flatten_nodes)

        day = 1
        day_index = self.start_day
        final_solution = []
        tabu_nodes = []
        while day <= self.travel_days:
            current_node = self.hotel
            day_type = "weekend" if day_index >= 5 else "weekday"
            day_solution = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}
            next_node_candidates = [node for node in solution if node._id not in tabu_nodes]
            for i in range(len(next_node_candidates)):
                if self.next_node_check(current_node,next_node_candidates[i],current_node.depart_time.hour,day_type):
                    next_node_candidates[i] = self.set_next_node_depart_arrive_time(current_node,next_node_candidates[i],current_node.depart_time.hour,day_type)
                    day_solution['index'].append(next_node_candidates[i]._id)
                    day_solution['waktu'].append(next_node_candidates[i].arrive_time)
                    day_solution['rating'].append(next_node_candidates[i].rating)
                    day_solution['tarif'].append(next_node_candidates[i].tarif)
                    tabu_nodes.append(next_node_candidates[i]._id)
                    current_node = next_node_candidates[i]
            if current_node._id != self.hotel._id:
                self.hotel = self.set_next_node_depart_arrive_time(current_node,self.hotel,current_node.depart_time.hour,day_type)
                day_solution['waktu'].append(self.hotel.arrive_time)

            if len(day_solution['index']) > 0:
                final_solution.append(day_solution)

            day += 1
            day_index = day_index + 1 if day_index + 1 <= 6 else 0

        return final_solution


class BaseACS(BaseAntColony):
    """Ant Colony System rules shared by the ACS-BSO hybrid (time-dependent)."""

    def exploitation(self,current_node,next_node_candidates,local_pheromone_matrix,jam,day_type):
        max_pos = np.argmax([local_pheromone_matrix[current_node._id][next_node._id]['pheromone']*(self.MAUT_between_two_nodes(current_node,next_node,jam,day_type)**self.beta) for next_node in next_node_candidates])
        next_node = next_node_candidates[max_pos]
        return next_node

    def exploration(self,current_node,next_node_candidates,local_pheromone_matrix,jam,day_type):
        #penyebut
        sum_sample = 0
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node,jam,day_type)**self.beta
            sum_sample += pheromone_in_edge*heuristic_val

        #probability
        sum_sample = 0.0001 if sum_sample == 0 else sum_sample
        next_node_prob = []
        for next_node in next_node_candidates:
            pheromone_in_edge = local_pheromone_matrix[current_node._id][next_node._id]['pheromone']**self.alpha_t
            heuristic_val = self.MAUT_between_two_nodes(current_node,next_node,jam,day_type)**self.beta
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

        for node in self.pheromone_matrix:
            for next_node in self.pheromone_matrix[node]:
                pheromone = self.pheromone_matrix[node][next_node]['pheromone']
                if (node,next_node) in solution_index:
                    pheromone = ((1-self.alpha)*pheromone)+(self.alpha*best_fitness)
                else:
                    pheromone = ((1-self.alpha)*pheromone)+0
                self.pheromone_matrix[node][next_node]['pheromone'] = pheromone

    def set_max_iter_acs(self,max_iter_acs):
        self.max_iter_acs = max_iter_acs


class BaseILS(BaseOptimizer,_ParentBaseILS):
    """Shared parts of the Iterated Local Search family (ILS, ILS-SA, ILS-TS, ILS-SA-TS) for time-dependent data.

    construct_solution, generate_init_solution and perturbation come unchanged from optimization.base.BaseILS;
    split_itinerary and solution_list_of_nodes_to_dict come from the time-dependent BaseOptimizer above."""
    pass

import random
import copy
import datetime
import numpy as np
from optimization.base import BaseOptimizer

class TS_VRP(BaseOptimizer):
    def __init__(self,max_tabu_size = 20,max_iter = 7500,max_idem = 7500,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.max_tabu_size = max_tabu_size #max size of tabu list
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
        self.rest_nodes = [] #1D list of nodes, [node1,node2,node4,node5,....]
    
    def set_model(self,tour,hotel,timematrix,init_solution=[],travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        else:
            self.init_solution = self.generate_init_solution()
        self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)
    
    def generate_init_solution(self):
        """Greedy itinerary: each step adds the feasible POI with the highest MAUT_between_two_nodes. 2D list of nodes (one list per day)."""
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
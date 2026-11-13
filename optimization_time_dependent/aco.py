# ANT COLONY OPTIMIZATION
import copy
import datetime
from optimization_time_dependent.base import BaseAntColony

class ACO_VRP_TD(BaseAntColony):
    def __init__(self,alpha = 1,beta = 4,q0 = 0.6,init_pheromone = 0.3,rho = 0.1,num_ant = 35,max_iter = 100,max_idem=50,random_state=None):
        super().__init__(random_state = random_state)

        #parameter setting
        self.alpha = alpha #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in ACO transition rule
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration ACO
        self.max_idem = max_idem

    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            best_found_solution = None
            best_found_fitness = 0
            local_pheromone_matrix = copy.deepcopy(self.pheromone_matrix)
            local_pheromone_matrix = self.init_delta_to_pheromone_matrix(local_pheromone_matrix)
            for ant in range(self.num_ant): #step
                ant_solution = []
                ant_solution_dict = []
                day = 1
                day_index = self.start_day
                tabu_nodes=[]
                while day<=self.travel_days:
                    current_node = self.hotel
                    day_type = "weekend" if day_index >= 5 else "weekday"
                    ant_day_solution = []
                    ant_day_solution_dict = {"index":[],"waktu":[current_node.depart_time],"rating":[],"tarif":[]}

                    for pos in range(len(self.tour)+1):
                        #recheck next node candidates (perlu dicek jam sampainya apakah melebihi max time)
                        next_node_candidates = [node for node in self.tour if node._id not in tabu_nodes and self.next_node_check(current_node,node,current_node.depart_time.hour,day_type)==True]

                        if len(next_node_candidates) > 0:
                            #transition rules
                            next_node = self.transition_rule(current_node,next_node_candidates,local_pheromone_matrix,current_node.depart_time.hour,day_type)
                            next_node = self.set_next_node_depart_arrive_time(current_node,next_node,current_node.depart_time.hour,day_type)

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
                            last_node = self.set_next_node_depart_arrive_time(current_node,last_node,current_node.depart_time.hour,day_type)
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
                    day_index = day_index + 1 if day_index + 1 <= 6 else 0

                fitness = self.MAUT(ant_solution_dict)
                if fitness > best_found_fitness:
                    best_found_fitness = fitness
                    best_found_solution = copy.deepcopy(ant_solution_dict)

                #add delta
                for day in ant_solution_dict:
                    local_pheromone_matrix = self.add_delta_to_pheromone_matrix(day,local_pheromone_matrix,fitness)

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

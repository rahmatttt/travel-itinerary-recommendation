# HYBRID ACS - BSO
import random
import copy
import datetime
from optimization_time_dependent.bso import BSO_VRP_TD
from optimization_time_dependent.base import BaseACS

class ACSBSO_VRP_TD(BaseACS):
    def __init__(self,alpha_t = 1,beta = 1,q0 = 0.1,init_pheromone = 0.1,rho = 0.1,alpha = 0.1,num_ant = 30,max_iter_acs = 200,max_iter_bso=15,p0=0.5,p1=0.4,p2=0.4,p3=0.5,p4=0.5,max_idem_acs=30,max_idem_bso=10,random_state=None):
        super().__init__()

        #ACS parameter setting
        self.alpha_t = alpha_t #relative value for pheromone (in transition rule)
        self.beta = beta #relative value for heuristic value (in transition rule)
        self.q0 = q0 #threshold in ACS transition rule
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate local pheromone update
        self.alpha = alpha #evaporation rate global pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter_acs = max_iter_acs #max iteration ACS
        self.max_idem = max_idem_acs #stop if the best fitness doesn't increase for max_idem iteration

        #hybrid setting
        self.p0 = p0 #less than: no BSO, more than: do BSO

        # BSO parameter setting
        self.p1 = p1 #less than: 2-opt, more than: 2-interchange
        self.p2 = p2 #less than: center, more than: random cluster (2-opt)
        self.p3 = p3 #less than: interchange a cluster and rest nodes, more than: interchange between clusters
        self.p4 = p4 #less than: center, more than: random cluster

        self.max_iter_bso = max_iter_bso #max iteration of BSO
        self.max_idem_bso = max_idem_bso #stop if the best fitness doesn't increase for max_idem iteration
        self.bso_model = BSO_VRP_TD(p1=self.p1,p2=self.p2,p3=self.p3,p4=self.p4,
                                max_iter=self.max_iter_bso,max_idem=self.max_idem_bso,
                                two_opt_method = "first",random_state=random_state)

        #set random seed
        self.random_state = random_state
        random.seed(self.random_state)

    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter_acs): #iteration
            best_found_solution = None
            best_found_solution_dict = None
            best_found_fitness = 0
            local_pheromone_matrix = copy.deepcopy(self.pheromone_matrix)
            for ant in range(self.num_ant): #step
                ant_solution = []
                ant_solution_dict = []
                day = 1
                day_index = self.start_day
                tabu_nodes = []
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
                    best_found_solution = copy.deepcopy(ant_solution)
                    best_found_solution_dict = copy.deepcopy(ant_solution_dict)

                # local pheromone update
                local_pheromone_matrix = self.local_pheromone_update(ant_solution_dict,fitness,local_pheromone_matrix)

            #BSO
            if random.uniform(0,1) > self.p0:
                self.bso_model.set_model(self.tour,self.hotel,self.timematrix,init_solution=best_found_solution,
                                        start_day=self.start_day,travel_days=self.travel_days,
                                        depart_time=self.hotel.depart_time,max_travel_time=self.max_travel_time,
                                        degree_waktu = self.degree_waktu,
                                        degree_tarif = self.degree_tarif,degree_rating = self.degree_rating)

                _,new_solution,new_fitness = self.bso_model.construct_solution()


                if new_fitness > best_found_fitness:
                    best_found_solution_dict = new_solution
                    best_found_fitness = new_fitness

            #global pheromone update
            self.global_pheromone_update(best_found_solution_dict,best_found_fitness)

            #checking best vs best found
            if best_found_fitness > best_fitness:
                best_fitness = best_found_fitness
                best_solution = copy.deepcopy(best_found_solution_dict)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_solution,best_fitness

        return best_solution,best_fitness

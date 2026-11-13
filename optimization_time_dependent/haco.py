# HYBRID ANT COLONY OPTIMIZATION
import random
import copy
import datetime
from optimization_time_dependent.base import BaseAntColony

class HACO_VRP_TD(BaseAntColony):
    #CLASSIC HACO (WITH MUTATION)
    def __init__(self,alpha = 1,beta = 4,q0 = 0.4,q1=0.6,q2=0.3,q3=0.6,init_pheromone = 0.5,rho = 0.7,num_ant = 30,max_iter = 100,max_idem=50,random_state=None):
        super().__init__(random_state = random_state)

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

    def mutation(self,solution_flatten_nodes):
        solution = copy.deepcopy(solution_flatten_nodes)

        if random.uniform(0,1) <= self.q2:
            #interchange
            for i in range(random.randint(1,len(solution)-1)):
                node1,node2 = random.sample(range(len(solution)),2)
                solution[node1],solution[node2] = solution[node2],solution[node1]
                return solution

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

    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            local_pheromone_matrix = copy.deepcopy(self.pheromone_matrix)
            local_pheromone_matrix = self.init_delta_to_pheromone_matrix(local_pheromone_matrix)
            all_ant_solution = [] #[{"solution":[],"fitness":}]
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
                all_ant_solution.append({"solution":copy.deepcopy(ant_solution_dict),"fitness":fitness})

                #mutation
                if random.uniform(0,1)<=self.q1:
                    rest_nodes = self.get_rest_nodes_from_solution(ant_solution)
                    if len(sum(ant_solution,[])+rest_nodes) > 1:
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

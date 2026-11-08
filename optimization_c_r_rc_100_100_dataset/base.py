import random
import copy
import numpy as np

class BaseOptimizer(object):
    """Attributes and methods shared by every optimizer in this folder."""

    def __init__(self,random_state = None):
        # data model setting
        self.nodes = None #node yang dipilih oleh user untuk dikunjungi
        self.depot = None #depot: starting and end point
        self.max_travel_time = None
        self.num_vehicle = None

        #set random seed
        if random_state != None:
            random.seed(random_state)

    def set_model(self,nodes,depot,num_vehicle = 3):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]

    def fitness(self,solution):
        return sum([sol["S"] for sol in sum(solution,[])])

    def fitness_between_two_nodes(self,current_node,next_node):
        return current_node["S"]+next_node["S"]

    def euclidean(self,node1,node2):
        return np.sqrt(((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2))

    def next_node_check(self,current_node,next_node,current_time):
        travel_time = self.euclidean(current_node,next_node)
        arrival_time = current_time + travel_time
        finish_time = max(arrival_time,next_node["O"])+next_node["d"]
        return_to_depot_time = self.euclidean(next_node,self.depot)
        if (arrival_time <= next_node["C"]) and (finish_time + return_to_depot_time <= self.max_travel_time):
            return True
        else:
            return False

    def split_itinerary(self,solution):
        routes = []
        
        vehicle = 1
        tabu_nodes = []
        
        while vehicle <= self.num_vehicle:
            current_loc = self.depot
            current_time = self.depot["O"]
            route = []
            next_node_candidates = [node for node in solution if node["id"] not in tabu_nodes]
            for next_node in next_node_candidates:
                travel_time = self.euclidean(current_loc,next_node)
                arrival_time = current_time + travel_time
                finish_time = max(arrival_time,next_node["O"])+next_node["d"]
                return_to_depot_time = self.euclidean(next_node,self.depot)
                if (arrival_time <= next_node["C"]) and (finish_time + return_to_depot_time <= self.max_travel_time):
                    route.append(next_node)
                    tabu_nodes.append(next_node["id"])
                    current_loc = next_node
                    current_time = finish_time
                else:
                    continue
            
            if len(route) > 0:
                routes.append(route)
            
            if len(tabu_nodes) == len(self.nodes):
                break
            
            vehicle += 1
        
        return routes

    def get_rest_nodes_from_solution(self,solution_nodes):
        solution_id = [node["id"] for node in sum(solution_nodes,[])]
        rest_nodes = [node for node in self.nodes if node["id"] not in solution_id]
        return rest_nodes

    def solution_list_of_float_to_nodes(self,solutions):
        index_order = np.flip(np.argsort(solutions))
        return list(np.array(self.nodes)[index_order])


class BaseAntColony(BaseOptimizer):
    """Shared parts of the ant colony family (ACO, ACS, HACO, ACS-BSO)."""

    def transition_rule(self,current_node,next_node_candidates,local_pheromone_matrix):
        q = random.uniform(0,1)
        if q <= self.q0: #exploitation
            next_node = self.exploitation(current_node,next_node_candidates,local_pheromone_matrix)
        else: #exploration
            next_node = self.exploration(current_node,next_node_candidates,local_pheromone_matrix)
        return next_node


class BaseACS(BaseAntColony):
    """Ant Colony System rules shared by ACS and the ACS-BSO hybrid."""



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
                    best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))
                    return best_solution,best_fitness
        
        best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_solution))
        return best_solution,best_fitness

    def generate_init_solution(self):
        return np.random.uniform(-5.12,5.12,size=len(self.nodes))

    def perturbation(self,solution_nodes):
        return solution_nodes + np.random.uniform(-self.strength,self.strength,size=len(solution_nodes)) 

# HYBRID ACS - BSO
import random
import copy
import datetime
from optimization.bso import BSO_VRP,BSO_TSP
from optimization.base import BaseACS

class ACSBSO_VRP(BaseACS):
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
        self.bso_model = BSO_VRP(p1=self.p1,p2=self.p2,p3=self.p3,p4=self.p4,
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
            local_pheromone_matrix = copy.deepcopy(self.timematrix)
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
                if fitness > best_found_fitness:
                    best_found_fitness = fitness
                    best_found_solution = copy.deepcopy(ant_solution)
                    best_found_solution_dict = copy.deepcopy(ant_solution_dict)
                
                # local pheromone update
                local_pheromone_matrix = self.local_pheromone_update(ant_solution_dict,fitness,local_pheromone_matrix)
            
            #BSO
            if random.uniform(0,1) > self.p0:
                self.bso_model.set_model(self.tour,self.hotel,self.timematrix,init_solution=best_found_solution,
                                        travel_days=self.travel_days,degree_waktu = self.degree_waktu,
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

class ACSBSO_TSP(BaseACS):
    def __init__(self,alpha_t = 1,beta = 3,q0 = 0.1,init_pheromone = 0.1,rho = 0.9,alpha = 0.1,num_ant = 30,max_iter_acs = 200,max_iter_bso=15,p0=0.5,p1=0.9,max_idem_acs=30,max_idem_bso=10,random_state=None):
        super().__init__(random_state = random_state)

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

        #BSO model
        self.bso_model = BSO_TSP(p1=p1,max_iter=max_iter_bso,max_idem=max_idem_bso,
                                two_opt_method="first",random_state=random_state)
        
        self.max_waktu_tsp = None
    
    def set_model(self,tour,hotel,timematrix,travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        self.max_waktu_tsp = 0  # Initialize the total waktu to 0
        for source, destinations in self.timematrix.items():
            for destination, values in destinations.items():
                self.max_waktu_tsp += values['waktu']
        self.bso_model.set_model(tour,hotel,timematrix,init_solution=[],travel_days=travel_days,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
    
    def local_pheromone_update(self,solutions,fitness,local_pheromone_matrix):
        nodes = [self.hotel._id] + solutions['index'] + [self.hotel._id]
        i = 1
        while i < len(nodes):
            pheromone = local_pheromone_matrix[nodes[i-1]][nodes[i]]['pheromone']
            pheromone = ((1-self.rho)*pheromone)+(self.rho*fitness)
            local_pheromone_matrix[nodes[i-1]][nodes[i]]['pheromone'] = pheromone
            i += 1
        return local_pheromone_matrix
    
    def global_pheromone_update(self,best_solutions,best_fitness):
        solutions = [self.hotel._id] + best_solutions['index'] + [self.hotel._id]
        solution_index = [(solutions[i-1],solutions[i]) for i in range(1,len(solutions))]
        
        for node in self.timematrix:
            for next_node in self.timematrix[node]:
                pheromone = self.timematrix[node][next_node]['pheromone']
                if (node,next_node) in solution_index:
                    pheromone = ((1-self.alpha)*pheromone)+(self.alpha*best_fitness)
                else:
                    pheromone = ((1-self.alpha)*pheromone)+0
                self.timematrix[node][next_node]['pheromone'] = pheromone
     
    def TSP(self):
        best_solution = None
        best_fitness = 0
        idem_counter = 0
        for i in range(self.max_iter_acs): #iteration
            best_found_solution = None
            best_found_solution_dict = None
            best_found_fitness = 0
            local_pheromone_matrix = copy.deepcopy(self.timematrix)
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
                        last_node = copy.deepcopy(self.hotel)
                        ant_solution_dict['waktu'] += self.timematrix[current_node._id][last_node._id]['waktu']
                        break
                    else:
                        break
                
                fitness = self.MAUT_TSP(ant_solution_dict)
                if fitness >= best_found_fitness:
                    best_found_fitness = fitness
                    best_found_solution = copy.deepcopy(ant_solution)
                    best_found_solution_dict = copy.deepcopy(ant_solution_dict)
                
                # local pheromone update
                local_pheromone_matrix = self.local_pheromone_update(ant_solution_dict,fitness,local_pheromone_matrix)
            
            #BSO
            if random.uniform(0,1) > self.p0:
                self.bso_model.set_init_solution(best_found_solution)
                new_solution,new_fitness = self.bso_model.TSP()
            
            
                if new_fitness >= best_found_fitness:
                    best_found_solution = new_solution
                    best_found_solution_dict = self.bso_model.create_solution_dict_TSP(new_solution)
                    best_found_fitness = new_fitness
            
            #global pheromone update
            self.global_pheromone_update(best_found_solution_dict,best_found_fitness)

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
    
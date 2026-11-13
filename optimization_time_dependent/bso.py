# BRAINSTORM OPTIMIZATION
import random
import copy
import datetime
import numpy as np
from optimization_time_dependent.base import BaseOptimizer

class BSO_VRP_TD(BaseOptimizer):
    def __init__(self,p1 = 0.7,p2 = 0.7,p3 = 0.9,p4 = 0.3,max_iter = 100,max_idem = 15,two_opt_method="best",random_state=None):
        super().__init__(random_state = random_state)

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
        self.init_days = [] #day offset (from start_day) of each itinerary in init_solution, [0,1,....]
        self.rest_nodes = [] #1D list of nodes, [node1,node2,node4,node5,....]

    def set_model(self,tour,hotel,timematrix,init_solution=[],start_day = 5,travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,start_day = start_day,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
        else:
            self.init_solution = self.generate_init_solution()
        self.init_days = list(range(len(self.init_solution))) #init solution is ordered by day
        self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)
        if len(self.rest_nodes) == 0:
            self.p3 = 0 #no interchange with rest nodes
        if len(self.init_solution)<=1 and len(self.rest_nodes)>0:
            self.p3 = 1 #no interchange between clusters
        elif len(self.init_solution)<=1 and len(self.rest_nodes)==0:
            self.p1 = 1 #no two interchange (only two opt)
    
    def set_init_solution(self,init_solution):
        # inital solution
        if len(init_solution) > 0:
            self.init_solution = init_solution
            self.init_days = list(range(len(self.init_solution))) #init solution is ordered by day

            self.rest_nodes = self.get_rest_nodes_from_solution(self.init_solution)

            if len(self.rest_nodes) == 0:
                self.p3 = 0 #no interchange with rest nodes

            if len(self.init_solution)<=1 and len(self.rest_nodes)>0:
                self.p3 = 1 #no interchange between clusters
            elif len(self.init_solution)<=1 and len(self.rest_nodes)==0:
                self.p1 = 1 #no two interchange (only two opt)
    
    def generate_init_solution(self):
        """Random order of all POIs split into days, as a 2D list of nodes (one list per day)."""
        return self.split_itinerary(super().generate_init_solution())
    
    def random_clustering(self,solution_nodes,days):
        #shuffle positions instead of the itineraries so every itinerary keeps its day (same random draws as shuffling the list)
        order = list(range(len(solution_nodes)))
        random.shuffle(order)
        cluster_a = [solution_nodes[i] for i in order[:len(order)//2]]
        cluster_b = [solution_nodes[i] for i in order[len(order)//2:]]
        days_a = [days[i] for i in order[:len(order)//2]]
        days_b = [days[i] for i in order[len(order)//2:]]
        return cluster_a,days_a,cluster_b,days_b

    def find_center_index(self,cluster_nodes,cluster_days):
        return np.argmax([self.MAUT(self.solution_list_of_nodes_to_dict([i],[d])) for i,d in zip(cluster_nodes,cluster_days)])

    def sort_by_day(self,solution,days):
        #output: itineraries and days ordered by day (for the returned solution)
        pairs = sorted(zip(days,solution), key=lambda x: x[0])
        return [p[1] for p in pairs],[p[0] for p in pairs]

    def two_opt(self,itinerary,day):
        solution = [self.hotel] + itinerary + [self.hotel]
        fitness = self.MAUT(self.solution_list_of_nodes_to_dict([solution[1:-1]],[day]),use_penalty=False)
        n = len(solution)
        
        if self.two_opt_method == "first": #first improvement
            improved = False
            for i in range(1,n-2):
                for j in range(i+2,n-1):
                    if j-i == 1:
                        continue
                    temp_solution = copy.deepcopy(solution)
                    temp_solution[i+1:j+1] = reversed(temp_solution[i+1:j+1])
                    temp_solution[1:-1],status = self.reset_depart_arrive_time(temp_solution[1:-1],day)
                    if status:
                        temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp_solution[1:-1]],[day]),use_penalty=False)
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
                        temp_solution[1:-1],status = self.reset_depart_arrive_time(temp_solution[1:-1],day)
                        if status:
                            temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp_solution[1:-1]],[day]),use_penalty=False)
                            if temp_fitness > fitness:
                                solution = copy.deepcopy(temp_solution)
                                fitness = temp_fitness
                                improved = True

        return solution[1:-1] #return best or return unchanged
    
    def two_interchange(self,itinerary1,itinerary2, day1, day2 = None, rest_nodes = False):
        #day1, day2: day offset of itinerary1 and itinerary2 (day2 is not used if rest_nodes == True)
        operator = [(0,1),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        solution1 = copy.deepcopy(itinerary1)
        solution2 = copy.deepcopy(itinerary2) #rest nodes if rest_nodes == True
        if rest_nodes == False:
            fitness = self.MAUT(self.solution_list_of_nodes_to_dict([solution1,solution2],[day1,day2]),use_penalty=False)
        else:
            fitness = self.MAUT(self.solution_list_of_nodes_to_dict([solution1],[day1]),use_penalty=False)
        
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
                    
                    #flatten
                    temp1 = sum(temp1,[])
                    temp2 = sum(temp2,[])

                    # unify the node list if op > 1
                    if op[0]>1:
                        tabu_nodes = []
                        temp_nodes = []
                        for node in temp1:
                            if node._id not in tabu_nodes:
                                temp_nodes.append(node)
                                tabu_nodes.append(node._id)
                        temp1 = copy.deepcopy(temp_nodes)
                        
                    if op[1]>1:
                        tabu_nodes = []
                        temp_nodes = []
                        for node in temp2:
                            if node._id not in tabu_nodes:
                                temp_nodes.append(node)
                                tabu_nodes.append(node._id)
                        temp2 = copy.deepcopy(temp_nodes)
                    
                    #reset depart and arrive time
                    temp1,status1 = self.reset_depart_arrive_time(temp1,day1)
                    if rest_nodes == False:
                        temp2,status2 = self.reset_depart_arrive_time(temp2,day2)
                    else:
                        status2 = True
                    
                    if status1 and status2:
                        #count fitness
                        if rest_nodes == False and len(temp2) > 0 and len(temp1) > 0:
                            temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp1,temp2],[day1,day2]),use_penalty=False)
                        elif rest_nodes == False and len(temp2) > 0:
                        	temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp2],[day2]),use_penalty=False)
                        elif rest_nodes == False and len(temp1) > 0:
                        	temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp1],[day1]),use_penalty=False)
                        elif rest_nodes == True and len(temp1) > 0:
                            temp_fitness = self.MAUT(self.solution_list_of_nodes_to_dict([temp1],[day1]),use_penalty=False)
                        else:
                        	temp_fitness = 0
                    
                        if temp_fitness > fitness:
                            return temp1,temp2 #first improvement
            
            return itinerary1,itinerary2 #if no improvement
    
    def construct_solution(self):
        fitness = 0
        solution = copy.deepcopy(self.init_solution)
        days = copy.deepcopy(self.init_days) #day offset of each itinerary in solution
        idem_counter = 0
        for i in range(self.max_iter):
            if len(solution) > 1:
                #clustering
                clusters = {1:{},2:{}}
                clusters[1]['list'],clusters[1]['days'],clusters[2]['list'],clusters[2]['days'] = self.random_clustering(solution,days)

                #find center index
                clusters[1]['center'] = self.find_center_index(clusters[1]['list'],clusters[1]['days'])
                clusters[2]['center'] = self.find_center_index(clusters[2]['list'],clusters[2]['days'])
            else:
                clusters = {1:{}}
                clusters[1]['list'] = copy.deepcopy(solution)
                clusters[1]['days'] = copy.deepcopy(days)
                clusters[1]['center'] = self.find_center_index(clusters[1]['list'],clusters[1]['days'])
            
            #randomization p1
            if random.uniform(0,1) < self.p1 or (len(clusters)==1 and len(self.rest_nodes)==0):
                #2-opt
                cluster_id = random.randint(1,2) if len(solution)>1 else 1
                
                # randomization p2
                if random.uniform(0,1) < self.p2:
                    #center
                    itinerary_id = clusters[cluster_id]['center']
                else:
                    #random itienrary
                    itinerary_id = random.randint(0,len(clusters[cluster_id]['list'])-1)
                clusters[cluster_id]['list'][itinerary_id] = self.two_opt(clusters[cluster_id]['list'][itinerary_id],clusters[cluster_id]['days'][itinerary_id])
            else:
                #2-interchange
                if random.uniform(0,1) < self.p3 or (len(clusters)==1 and len(self.node)>0):
                    #interchange a cluster with rest nodes
                    cluster_id = random.randint(1,2) if len(clusters)>1 else 1
                    
                    #randomization p4
                    if random.uniform(0,1) < self.p4:
                        itinerary_id = clusters[cluster_id]['center']
                    else:
                        itinerary_id = random.randint(0,len(clusters[cluster_id]['list'])-1)
                    clusters[cluster_id]['list'][itinerary_id],self.rest_nodes = self.two_interchange(clusters[cluster_id]['list'][itinerary_id],self.rest_nodes,clusters[cluster_id]['days'][itinerary_id],rest_nodes=True)
                else:
                    #interchange between two clusters
                    #randomization p4
                    if random.uniform(0,1) < self.p4:
                        itinerary_id_1 = clusters[1]['center']
                        itinerary_id_2 = clusters[2]['center']
                    else:
                        itinerary_id_1 = random.randint(0,len(clusters[1]['list'])-1)
                        itinerary_id_2 = random.randint(0,len(clusters[2]['list'])-1)
                    clusters[1]['list'][itinerary_id_1],clusters[2]['list'][itinerary_id_2] = self.two_interchange(clusters[1]['list'][itinerary_id_1],clusters[2]['list'][itinerary_id_2],clusters[1]['days'][itinerary_id_1],clusters[2]['days'][itinerary_id_2],rest_nodes=False)

            #merge cluster
            solution = []
            days = []
            for cluster_id in clusters:
                solution.extend(clusters[cluster_id]['list'])
                days.extend(clusters[cluster_id]['days'])

            days = [d for d,sol in zip(days,solution) if len(sol)>0]
            solution = [sol for sol in solution if len(sol)>0]
            solution_dict = self.solution_list_of_nodes_to_dict(solution,days)
            new_fitness = self.MAUT(solution_dict)
            if new_fitness > fitness:
                fitness = new_fitness
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    solution,days = self.sort_by_day(solution,days)
                    return solution,self.solution_list_of_nodes_to_dict(solution,days),fitness

        solution,days = self.sort_by_day(solution,days)
        return solution,self.solution_list_of_nodes_to_dict(solution,days),fitness

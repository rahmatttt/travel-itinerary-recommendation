# GENETIC ALGORITHM
from optimization.koneksi import ConDB
import random
import math
import copy
import time
import json
import datetime
import numpy as np

class DKA(object):
    def __init__(self,n = 5,p=0.5,smep=5,max_iter = 1000,max_idem = 20,random_state=None, return_fitness_history=False):
        # parameter setting
        self.n = n #number of komodo
        self.p = p #portion of big male komodo
        self.smep = smep #small male exploration probability (0,10)
        self.max_iter = max_iter #max iteration of tabu search
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration

        # set initial solution
        self.init_solution = [] #2D list of nodes, [[node1,node2,....],[node4,node5,....]]
        self.rest_nodes = [] #1D list of nodes, [node1,node2,node4,node5,....]
        
        # data model setting
        self.nodes = None #node yang dipilih oleh user untuk dikunjungi
        self.depot = None #depot: starting and end point
        self.max_travel_time = None
        self.num_vehicle = None
    
        #set random seed
        if random_state != None:
            random.seed(random_state)
            
        self.return_fitness_history = return_fitness_history
    
    def set_model(self,nodes,depot,num_vehicle=3):
        #initiate model
        self.nodes = copy.deepcopy(nodes)
        self.depot = copy.deepcopy(depot)
        self.num_vehicle = num_vehicle
        self.max_travel_time = depot["C"]
        
        self.n_big_male = int(np.floor((self.p*self.n)-1))
        self.n_big_male = 1 if self.n_big_male == 0 else self.n_big_male
        self.n_female = 1
        self.n_small_male = int(self.n - (self.n_big_male+self.n_female))
    
    def fitness(self,solution):
        return sum([sol["S"] for sol in sum(solution,[])])
    
    def fitness_between_two_nodes(self,current_node,next_node):
        return current_node["S"]+next_node["S"]
    
    def euclidean(self,solution1,solution2):
        return np.sqrt(((solution1["x"] - solution2["x"])**2 + (solution1["y"] - solution2["y"])**2))
    
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
    
    def swap_operator(self,komodo):
        k = copy.deepcopy(komodo)
        pos1,pos2 = random.sample(range(len(k)),2)
        k[pos1],k[pos2] = k[pos2],k[pos1]
        return k
    
    def cluster_komodo(self,komodo_ls):
        komodo_dict = [{'solution':i,
                       'fitness':self.fitness(self.split_itinerary(i))} for i in komodo_ls]
        komodo_dict = sorted(komodo_dict, key=lambda x: x["fitness"], reverse = True)
        big_males = copy.deepcopy(komodo_dict[:self.n_big_male])
        female = copy.deepcopy(komodo_dict[self.n_big_male])
        small_males = copy.deepcopy(komodo_dict[self.n_big_male+self.n_female:])
        return big_males,female,small_males
    
    def distance_between_komodo(self,komodo,komodo_target):
        #distance between komodo is the edges in komodo_target that not exist in komodo
        k = [i["id"] for i in komodo]
        k_edge = [[k[i-1],k[i]] for i in range(1,len(k))]
        
        k_target = [i["id"] for i in komodo_target]
        k_target_edge = [[k_target[i-1],k_target[i]] for i in range(1,len(k_target))]
        
        distance = [i for i in k_target_edge if i not in k_edge]
        return distance
    
    def similar_edge_between_komodo(self,komodo,komodo_target):
        #distance between komodo is the edges in komodo_target that not exist in komodo
        k = [i["id"] for i in komodo]
        k_edge = [[k[i-1],k[i]] for i in range(1,len(k))]
        
        k_target = [i["id"] for i in komodo_target]
        k_target_edge = [[k_target[i-1],k_target[i]] for i in range(1,len(k_target))]
        
        similar = [i for i in k_target_edge if i in k_edge]
        return similar
    
    def find_segment(self,komodo,komodo_target,edge_target):
        k = [i["id"] for i in komodo]
        
        k_target = [i["id"] for i in komodo_target]
        k_target_edge = [[k_target[i-1],k_target[i]] for i in range(1,len(k_target))]
        if k.index(edge_target[0]) < k.index(edge_target[1]):
            x = edge_target[0]
            pos_x = k.index(edge_target[0])
            
            y = edge_target[1]
            pos_y = k.index(edge_target[1])
        else:
            x = edge_target[1]
            pos_x = k.index(edge_target[1])
            
            y = edge_target[0]
            pos_y = k.index(edge_target[0])
            
        # segment x and a
        if pos_x == 0:
            segment_x = [pos_x,pos_x+1] #start,end in [start:end] => in index 0 only
            segment_a = -1
        else:
            stop_point = pos_x
            for i in range(pos_x-1,-1,-1):
                if [k[i],k[i+1]] in k_target_edge:
                    stop_point = stop_point - 1
                else:
                    break
            segment_x = [stop_point,pos_x+1] #stop point until pos_x
            segment_a = [0,stop_point]
        
        # segment y and c
        if pos_y == len(k)-1:
            segment_y = [pos_y,pos_y+1] #start,end in [start:end] => in index -1 only
            segment_c = -1
        else:
            stop_point = pos_y
            for i in range(pos_y,len(k)-1):
                if [k[i],k[i+1]] in k_target_edge:
                    stop_point = stop_point + 1
                else:
                    break
            segment_y = [pos_y,stop_point+1] #pos_y until stop_point
            segment_c = [stop_point+1,len(k)]
        
        #segment b
        segment_b = -1 if segment_y[0]-(segment_x[1]-1) == 1 else [segment_x[1],segment_y[0]]

        return segment_a,segment_x,segment_b,segment_y,segment_c
    
    def edge_construction(self,komodo,komodo_target):
        # find distance
        distances = self.distance_between_komodo(komodo,komodo_target)
        
        if len(distances) > 0:
            selected_distance = distances[random.randint(0,len(distances)-1)]

            # create segment
            segment_a,segment_x,segment_b,segment_y,segment_c = self.find_segment(komodo,komodo_target,selected_distance)
            nodes_a = [] if segment_a == -1 else copy.deepcopy(komodo[segment_a[0]:segment_a[1]])
            nodes_x = copy.deepcopy(komodo[segment_x[0]:segment_x[1]])
            nodes_b = [] if segment_b == -1 else copy.deepcopy(komodo[segment_b[0]:segment_b[1]])
            nodes_y = copy.deepcopy(komodo[segment_y[0]:segment_y[1]])
            nodes_c = [] if segment_c == -1 else copy.deepcopy(komodo[segment_c[0]:segment_c[1]])

            if nodes_b != []:
                #operator 1
                result1 = nodes_a + nodes_b + nodes_x + nodes_y + nodes_c
                fitness1 = self.fitness(self.split_itinerary(result1))

                #operator 2
                result2 = nodes_a + nodes_b + nodes_y[::-1] + nodes_x[::-1] + nodes_c
                fitness2 = self.fitness(self.split_itinerary(result2))

                #operator 3
                result3 = nodes_a + nodes_y[::-1] + nodes_x[::-1] + nodes_b + nodes_c
                fitness3 = self.fitness(self.split_itinerary(result3))

                #operator 4
                result4 = nodes_a + nodes_x + nodes_y + nodes_b + nodes_c
                fitness4 = self.fitness(self.split_itinerary(result4))
                
                best_operator = np.argmax([fitness1,fitness2,fitness3,fitness4]) + 1
            else:
                #operator 1
                result1 = nodes_a + nodes_y[::-1] + nodes_x[::-1] + nodes_c
                fitness1 = self.fitness(self.split_itinerary(result1))
                
                #operator 2
                result2 = nodes_a + nodes_x + nodes_y + nodes_c
                result2[segment_x[1]],result2[segment_y[0]] = result2[segment_y[0]],result2[segment_x[1]]
                fitness2 = self.fitness(self.split_itinerary(result2))
                best_operator = np.argmax([fitness1,fitness2]) + 1
            
            if best_operator == 1:
                return result1,fitness1
            elif best_operator == 2:
                return result2,fitness2
            elif best_operator == 3:
                return result3,fitness3
            else:
                return result4,fitness4
        else:
            result = komodo
            fitness = self.fitness(self.split_itinerary(komodo))
            return result,fitness
        
    def edge_destruction(self,komodo,komodo_target):
        # find similars
        similars = self.similar_edge_between_komodo(komodo,komodo_target)

        if len(similars) > 0:
            selected_edge = similars[random.randint(0,len(similars)-1)] if len(similars) > 1 else similars[0]
            
            # create segment
            segment_a,segment_x,segment_b,segment_y,segment_c = self.find_segment(komodo,komodo_target,selected_edge)
            nodes_a = [] if segment_a == -1 else copy.deepcopy(komodo[segment_a[0]:segment_a[1]])
            nodes_x = copy.deepcopy(komodo[segment_x[0]:segment_x[1]])
            nodes_y = copy.deepcopy(komodo[segment_y[0]:segment_y[1]])
            nodes_c = [] if segment_c == -1 else copy.deepcopy(komodo[segment_c[0]:segment_c[1]])
            
            #operator 1 (nodes_a != [])
            if nodes_a != []:
                result1 = nodes_x + nodes_a + nodes_y + nodes_c
                fitness1 = self.fitness(self.split_itinerary(result1))
            else:
                fitness1 = -999 #eliminate this operator
                
            #operator 2
            result2 = nodes_a + nodes_y + nodes_x + nodes_c
            fitness2 = self.fitness(self.split_itinerary(result2))
            
            #operator 3
            result3 = nodes_a + nodes_x[::-1] + nodes_y[::-1] + nodes_c
            fitness3 = self.fitness(self.split_itinerary(result3))
            
            #operator 4 (len(nodes_x)>1)
            if len(nodes_x)>1:
                result4 = nodes_a + nodes_x[::-1] + nodes_y + nodes_c
                fitness4 = self.fitness(self.split_itinerary(result4))
            else:
                fitness4 = -999 #eliminate this operator
                
            #operator 5 (len(nodes_y)>1)
            if len(nodes_y)>1:
                result5 = nodes_a + nodes_x + nodes_y[::-1] + nodes_c
                fitness5 = self.fitness(self.split_itinerary(result5))
            else:
                fitness5 = -999 #eliminate this operator
            
            #operator 6 (nodes_c != [])
            if nodes_c != []:
                result6 = nodes_a + nodes_x + nodes_c + nodes_y
                fitness6 = self.fitness(self.split_itinerary(result6))
            else:
                fitness6 = -999 #eliminate this operator
                
            best_operator = np.argmax([fitness1,fitness2,fitness3,fitness4,fitness5,fitness6]) + 1
            if best_operator == 1:
                return result1,fitness1
            elif best_operator == 2:
                return result2,fitness2
            elif best_operator == 3:
                return result3,fitness3
            elif best_operator == 4:
                return result4,fitness4
            elif best_operator == 5:
                return result5,fitness5
            else:
                return result6,fitness6
        else:
            result = komodo
            fitness = self.fitness(self.split_itinerary(komodo))
            return result,fitness
    
    def construct_solution(self):
        best_solution = None
        best_fitness = 0
        
        idem_counter = 0
        komodo_ls = [random.sample(self.nodes,len(self.nodes)) for i in range(self.n)]
        
        big_males,female,small_males = self.cluster_komodo(komodo_ls)

        fitness_history = []
        
        for i in range(self.max_iter):
                        
            #big_males movement
            for j in range(len(big_males)):
                new_big_males = []
                for k in range(len(big_males)):
                    if j != k:
                        if big_males[j]['fitness']<big_males[k]['fitness'] or random.uniform(0,1) < 0.5:
                            new_k,new_fitness = self.edge_construction(big_males[j]['solution'],big_males[k]['solution'])
                            new_big_males.append({'solution':new_k,'fitness':new_fitness})
                        else:
                            new_k,new_fitness = self.edge_destruction(big_males[j]['solution'],big_males[k]['solution'])
                            new_big_males.append({'solution':new_k,'fitness':new_fitness})
                        
                        new_big_males = sorted(new_big_males, key=lambda x: x["fitness"], reverse = True)
                        if i == 0:
                            if big_males[0]['fitness'] < new_big_males[0]['fitness']:
                                big_males[j] = new_big_males[0]
                        else:
                            big_males[j] = new_big_males[0]
            
            #female movement
            if random.uniform(0,1) < 0.5:
                result1,fitness1 = self.edge_construction(female['solution'],big_males[0]['solution'])
                result2,fitness2 = self.edge_construction(big_males[0]['solution'],female['solution'])
                if fitness1 > fitness2:
                    female['solution'] = result1
                    female['fitness'] = fitness1
                else:
                    female['solution'] = result2
                    female['fitness'] = fitness2
            else:
                female['solution'] = self.swap_operator(female['solution'])
                female['fitness'] = self.fitness(self.split_itinerary(female['solution']))
            
            #small males movement
            for j in range(len(small_males)):
                new_small_males = []
                for k in range(len(big_males)):
                    if random.randint(0,10) <= self.smep:
                        new_k = self.swap_operator(small_males[j]['solution'])
                        new_fitness = self.fitness(self.split_itinerary(small_males[j]['solution']))
                        new_small_males.append({'solution':new_k,'fitness':new_fitness})
                    else:
                        new_k,new_fitness = self.edge_construction(small_males[j]['solution'],big_males[k]['solution'])
                        new_small_males.append({'solution':new_k,'fitness':new_fitness})

                    new_small_males = sorted(new_small_males, key=lambda x: x["fitness"], reverse = True)
                    small_males[j] = new_small_males[0]
            
            # update new komodo_ls
            komodo_ls = big_males + [female] + small_males
            komodo_ls = [i['solution'] for i in komodo_ls]
            
            # update cluster komodo
            big_males,female,small_males = self.cluster_komodo(komodo_ls)

            if self.return_fitness_history == True:
                fitness_history.append(big_males[0]['fitness'])
            
            # check best solution
            if best_fitness < big_males[0]['fitness']:
                best_solution = self.split_itinerary(big_males[0]['solution'])
                best_fitness = big_males[0]['fitness']
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    if self.return_fitness_history == True:
                        return best_solution,best_fitness,fitness_history
                    else:
                        return best_solution,best_fitness
        
        if self.return_fitness_history == True:
            return best_solution,best_fitness,fitness_history
        else:
            return best_solution,best_fitness
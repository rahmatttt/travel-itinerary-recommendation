# KOMODO MLIPIR ALGORITHM
import random
import copy
import datetime
import numpy as np
from optimization.base import BaseOptimizer

class KMA_VRP(BaseOptimizer):
    def __init__(self,n1 = 5,n2 = 200,max_n2 = 300,min_n2 = 20,p1=0.5,p2=0.5,d2=0.5,max_iter = 1000,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state,seed_numpy = True)

        # parameter setting
        self.n1 = n1 #number of komodo(population) in first phase
        self.n2 = n2 #number of komodo in second phase
        self.max_n2 = max_n2 #number of maximum komodo for second phase
        self.min_n2 = min_n2 #number of minimum komodo for second phase
        self.m = None #number of dimension
        self.p1 = p1 #portion of big male komodo in first phase
        self.p2 = p2 #portion of big male komodo in second phase
        self.d1 = None #mlipir rate in first phase
        self.d2 = d2 #mlipir rate in second phase
        self.a = 5 #number of decreased or increased generation in second phase
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
    
    def set_model(self,tour,hotel,timematrix,travel_days = 3, depart_time = datetime.time(8,0,0),max_travel_time = datetime.time(20,0,0),degree_waktu = 1,degree_tarif = 1,degree_rating = 1):
        super().set_model(tour,hotel,timematrix,travel_days = travel_days,depart_time = depart_time,max_travel_time = max_travel_time,degree_waktu = degree_waktu,degree_tarif = degree_tarif,degree_rating = degree_rating)
        # initiate number of each komodo for the first phase
        self.n_big_male = int(np.ceil((self.p1*self.n1)-1))
        self.n_female = 1
        self.n_small_male = int(self.n1 - (self.n_big_male+self.n_female))
        self.m = len(self.tour) #number of dimension
        self.d1 = (self.m - 1)/self.m #mlipir rate in first phase
    
    def initiate_cluster_komodo(self,komodo_ls):
        #for first clustering
        komodo_dict = [{'solution':i,
                       'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(i))))} for i in komodo_ls]
        komodo_dict = sorted(komodo_dict, key=lambda x: x["fitness"], reverse = True)
        big_males = copy.deepcopy(komodo_dict[:self.n_big_male])
        female = copy.deepcopy(komodo_dict[self.n_big_male])
        small_males = copy.deepcopy(komodo_dict[self.n_big_male+self.n_female:])
        return big_males,female,small_males
    
    def re_cluster_komodo(self,komodo_dict):
        #for reclustering
        komodo_dict = sorted(komodo_dict, key=lambda x: x["fitness"], reverse = True)
        big_males = copy.deepcopy(komodo_dict[:self.n_big_male])
        female = copy.deepcopy(komodo_dict[self.n_big_male])
        small_males = copy.deepcopy(komodo_dict[self.n_big_male+self.n_female:])
        return big_males,female,small_males
    
    def generate_population_first_phase(self):
        #the individu (komodo) is a list consists of a number [0,1]
        komodo_ls = [np.array([random.random() for i in range(self.m)]) for j in range(self.n1)]
        return komodo_ls
    
    def generate_population_second_phase(self, best_so_far_komodo, initiate_population = True):
        if initiate_population == True:
            komodo_ls = [np.array([random.random() for i in range(self.m)]) for j in range(self.n2)]
        else:
            komodo_ls = [np.array([random.random() for i in range(self.m)]) for i in range(self.a)]
        return komodo_ls
    
    def big_male_movement(self,big_males):
        new_big_males = []
        for i in range(len(big_males)):
            sum_wij = 0
            for j in range(len(big_males)):
                if i!=j:
                    r1 = random.uniform(0,1)
                    r2 = random.uniform(0,1)
                    if big_males[i]['fitness']<big_males[j]['fitness'] or r2<0.5:
                        wij = r1*(big_males[j]['solution'] - big_males[i]['solution'])
                    else:
                        wij = r1*(big_males[i]['solution'] - big_males[j]['solution'])
                    sum_wij = sum_wij + wij
            new_big_male = big_males[i]['solution'] + sum_wij
            fitness_new_big_male = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(new_big_male))))
            new_big_males.append({'solution':new_big_male,
                                 'fitness':fitness_new_big_male})
        
        new_big_males = new_big_males + big_males
        new_big_males = sorted(new_big_males, key=lambda x: x["fitness"], reverse = True)
        return new_big_males[:self.n_big_male]
    
    def female_movement(self,female,biggest_male):
        if random.uniform(0,1)<0.5:
            random_number = np.random.uniform(0,1,size=self.m)
            first_child = (random_number * female['solution']) + ((1-random_number)*female['solution'])
            second_child = (random_number * female['solution']) + ((1-random_number)*female['solution'])
            females = [
                {'solution':first_child,
                'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(first_child))))},
                {'solution':second_child,
                'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(second_child))))},
                copy.deepcopy(female)
            ]
        else:
            ub = max(female['solution'])
            lb = min(female['solution'])
            alpha = 0.1 #radius of parthenogenesis (fixed value)
            r = random.uniform(0,1)
            first_child = female['solution'] + ((2*r)-1)*alpha*np.abs(ub-lb)
            females = [
                {'solution':first_child,
                'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(first_child))))},
                copy.deepcopy(female)
            ]
        new_female = sorted(females, key=lambda x: x["fitness"], reverse = True)
        return new_female[0]
    
    def small_male_movement(self,small_males,big_males,mlipir_rate):
        new_small_males = []
        for i in range(len(small_males)):
            sum_wij = 0
            for j in range(len(big_males)):
                r1 = np.random.uniform(0,1,size=self.m)
                if random.uniform(0,1) < mlipir_rate:
                    wij = r1*(big_males[j]['solution'] - small_males[i]['solution'])
                else:
                    wij = 0
                sum_wij = sum_wij + wij
            new_small_male = small_males[i]['solution'] + sum_wij
            fitness_new_small_male = self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(new_small_male))))
            new_small_males.append({'solution':new_small_male,
                                 'fitness':fitness_new_small_male})
        
        new_small_males = new_small_males + small_males
        new_small_males = sorted(new_small_males, key=lambda x: x["fitness"], reverse = True)
        return new_small_males[:self.n_small_male]
    
    def first_phase(self):
        best_komodo = {'solution':None,
                      'fitness':0}
        
        idem_counter = 0
        
        komodo_ls = self.generate_population_first_phase()
        big_males,female,small_males = self.initiate_cluster_komodo(komodo_ls)
                
        for i in range(self.max_iter):
            #big males movement
            big_males = self.big_male_movement(big_males)
            
            #female movement
            female = self.female_movement(female,big_males)
            
            #small males movement
            small_males = self.small_male_movement(small_males,big_males,mlipir_rate = self.d1)
            
            #recluster
            komodo_ls = big_males + [female] + small_males
            big_males,female,small_males = self.re_cluster_komodo(komodo_ls)
            
            if big_males[0]['fitness'] > best_komodo['fitness']:
                best_komodo = big_males[0]
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_komodo
            
        return best_komodo
    
    def second_phase(self,best_so_far_komodo):
        best_komodo = best_so_far_komodo
        
        f_hist = [best_komodo['fitness'],best_komodo['fitness'],best_komodo['fitness']]
        
        idem_counter = 0
        komodo_ls = self.generate_population_second_phase(best_komodo, initiate_population = True)
        
        self.n_big_male = int(np.ceil((self.p2*self.n2)-1))
        self.n_female = 1
        self.n_small_male = int(self.n2 - (self.n_big_male+self.n_female))
        big_males,female,small_males = self.initiate_cluster_komodo(komodo_ls)
        
        for i in range(self.max_iter):
            #big males movement
            big_males = self.big_male_movement(big_males)
            
            #female movement
            female = self.female_movement(female,big_males)
            
            #small males movement
            small_males = self.small_male_movement(small_males,big_males,mlipir_rate = self.d2)
            
            #recluster
            komodo_ls = big_males + [female] + small_males
            
            f_hist = [best_komodo['fitness']] + f_hist
            if len(f_hist) > 3:
                f_hist = f_hist[:3]
            
            d_f1 = np.abs(f_hist[0]-f_hist[1])/f_hist[0]
            d_f2 = np.abs(f_hist[1]-f_hist[2])/f_hist[1]
            if d_f1 > 0 and d_f2 > 0:
                if self.n2 > self.min_n2:
                    self.n2 = self.n2 - self.a
                    komodo_ls = komodo_ls[:self.n2]
            else:
                if self.n2 < self.max_n2:
                    self.n2 = self.n2 + self.a
                    new_komodo_ls = self.generate_population_second_phase(best_komodo, initiate_population = False)
                    new_komodo_dict = [{'solution':sol,
                                        'fitness':self.MAUT(self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(sol))))} for sol in new_komodo_ls]
                    komodo_ls = komodo_ls + new_komodo_dict
                    
            self.n_big_male = int(np.ceil((self.p2*self.n2)-1))
            self.n_female = 1
            self.n_small_male = int(self.n2 - (self.n_big_male+self.n_female))
            big_males,female,small_males = self.re_cluster_komodo(komodo_ls)
            
            if big_males[0]['fitness'] > best_komodo['fitness']:
                best_komodo = big_males[0]
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_komodo
            
        return best_komodo
    
    def construct_solution(self):
        best_komodo = self.second_phase(self.first_phase())
        
        best_solution = self.solution_list_of_nodes_to_dict(self.split_itinerary(self.solution_list_of_float_to_nodes(best_komodo['solution'])))
        best_fitness = best_komodo['fitness']
        
        return best_solution,best_fitness
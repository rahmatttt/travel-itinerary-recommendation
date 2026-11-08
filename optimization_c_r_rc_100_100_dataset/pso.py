# PARTICLE SWARM OPTIMIZATION
import random
import copy
import numpy as np
from optimization_c_r_rc_100_100_dataset.base import BaseOptimizer

class PSO(BaseOptimizer):
    def __init__(self,n = 30, w = 0.7,c1 = 1.5,c2 = 1.5,max_iter = 200,max_idem = 20,random_state=None):
        super().__init__(random_state = random_state)

        # parameter setting
        self.n = n #number of particle(population)
        self.w = w #inertia weight
        self.c1 = c1 #cognitive coefficient: Strength of pull toward personal best (individual learning)
        self.c2 = c2 #social coefficient: Strength of pull toward global best (social learning)
        self.max_iter = max_iter #max iteration
        self.max_idem = max_idem #stop if the best fitness doesn't increase for max_idem iteration
    
    def generate_population(self):
        #the individu (particle) is a list consists of a number [0,1]
        particle_ls = [np.array([random.random() for i in range(len(self.nodes))]) for j in range(self.n)]
        particle_dict = []
        for particle in particle_ls:
            solution = particle
            fitness = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(particle)))
            velocity = np.random.uniform(-1,1,size=len(particle))
            particle_dict.append({
                "solution":solution,
                "velocity":velocity,
                "fitness":fitness,
                "best_fitness":fitness,
                "best_solution":solution
            })
            
        return particle_dict
    
    def update_velocity(self,particle_dict,best_solution):
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        cognitive = self.c1 * r1 * (particle_dict["best_solution"] - particle_dict["solution"])
        social = self.c2 * r2 * (best_solution - particle_dict["solution"])
        particle_dict['velocity'] = self.w * particle_dict['velocity'] + cognitive + social
        return particle_dict
    
    def update_solution(self,particle_dict):
        particle_dict['solution'] = particle_dict['solution']+particle_dict['velocity']
        particle_dict['fitness'] = self.fitness(self.split_itinerary(self.solution_list_of_float_to_nodes(particle_dict['solution'])))
        if particle_dict['fitness'] > particle_dict['best_fitness']:
            particle_dict['best_fitness'] = particle_dict['fitness']
            particle_dict['best_solution'] = particle_dict['solution']
        return particle_dict
    
    def construct_solution(self):
        idem_counter = 0
        
        particle_dict = self.generate_population()
        
        best_particle = sorted(particle_dict,key=lambda x: x['fitness'], reverse=True)[0]
        best_found_particle = copy.deepcopy(best_particle)
        
        for i in range(self.max_iter):
            for j in range(len(particle_dict)):
                #update velocity
                particle_dict[j] = self.update_velocity(particle_dict[j],best_particle['solution'])
                
                #update solution/position
                particle_dict[j] = self.update_solution(particle_dict[j])
                
                if particle_dict[j]['fitness'] > best_found_particle['fitness']:
                    best_found_particle = copy.deepcopy(particle_dict[j])
            
            if best_found_particle['fitness'] > best_particle['fitness']:
                best_particle = copy.deepcopy(best_found_particle)
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_particle['best_solution']))
                    best_fitness = best_particle['best_fitness']
                    return best_solution,best_fitness
        
        best_solution = self.split_itinerary(self.solution_list_of_float_to_nodes(best_particle['best_solution']))
        best_fitness = best_particle['best_fitness']
        return best_solution,best_fitness
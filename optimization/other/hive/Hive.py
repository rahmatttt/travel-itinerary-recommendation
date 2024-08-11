from scipy.stats import levy
import math
import random
import copy
import numpy as np


class Agent:
    def __init__(self, agent_length, lower_bound, upper_bound, f, is_maximizing):
        self.agent_length = agent_length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.f = f
        self.vector = []
        self.new_vector = []
        self.fitness = 0
        self.new_fitness = 0
        self.food_concentration = 0
        self.new_food_concentration = 0
        self.abandonment_count = 0
        self.is_maximizing = is_maximizing

    def generate_random_vector(self):
        self.vector = [(0 + random.random()) * 10 for i in range(self.agent_length)]
        self.abandonment_count = 0

    def generate_random_new_vector(self):
        self.new_vector = [(0 + random.random()) * 10 for i in range(self.agent_length)]

    def set_vector(self, vector):
        self.vector = vector
        self.abandonment_count = 0

    def set_new_vector(self, vector):
        self.new_vector = vector

    def set_food_concentration(self, food_concentration):
        self.food_concentration = food_concentration

    def set_new_food_concentration(self, new_food_concentration):
        self.new_food_concentration = new_food_concentration

    def set_fitness(self):
        if self.is_maximizing:
            self.fitness = 1 + self.f(self.vector)
        else:
            self.fitness = 1 / (1 + self.f(self.vector))

    def set_new_fitness(self):
        if self.is_maximizing:
            self.new_fitness = 1 + self.f(self.new_vector)
        else:
            self.fitness = 1 / (1 + self.f(self.new_vector))

    def add_abandonment_count(self):
        self.abandonment_count += 1

    @staticmethod
    def fix_outside_limit(lower_bound, upper_bound, vector):
        for i in range(len(vector)):
            if vector[i] < lower_bound:
                vector[i] = lower_bound
            elif vector[i] > upper_bound:
                vector[i] = upper_bound
        return vector

    def check(self):
        self.set_vector(self.fix_outside_limit(self.lower_bound, self.upper_bound, self.vector))
        self.set_new_vector(self.fix_outside_limit(self.lower_bound, self.upper_bound, self.new_vector))


class ABC:
    population = []
    MAX_ABANDONMENT_COUNT = 5
    XBest = None
    FBest = None

    def __init__(self, max_iter, f, agent_count, agent_length, lower_bound, upper_bound, is_maximizing, n):
        self.T = max_iter  # iterasi maksimal
        self.t = 0  # iterasi saat ini
        self.f = f
        self.agent_count = agent_count
        self.agent_length = agent_length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population = []
        self.is_maximizing = is_maximizing
        self.n = n

    def get_top_n_agents(self, n):
        best_bee = copy.deepcopy(self.population)
        best_bee.sort(key=lambda agent: agent.fitness, reverse=self.is_maximizing)
        best_bee = best_bee[:n]
        XbestN = []
        for i in range(len(best_bee)):
            XbestN.append(best_bee[i].vector)
        return XbestN[:n]

    def run(self):
        print('Starting ABC')
        self.initialization_phase()  # initialise
        self.find_best()
        for i in range(self.T):
            self.t = i + 1
            self.employed_phase()
            self.onlook_phase()
            self.scout_phase()
            self.find_best()
            print("Iteration " + str(self.t) + " | Fbest = " + str(self.FBest))
        Xbests = self.get_top_n_agents(self.n)
        return Xbests, self.FBest

    def initialization_phase(self):
        self.population = []
        for i in range(self.agent_count):
            self.population.append(Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing))
            self.population[i].generate_random_vector()
            self.population[i].set_fitness()

    def is_a_better_than_b(self, fitness_a, fitness_b):
        return (self.is_maximizing and fitness_a > fitness_b) or ((not self.is_maximizing) and fitness_a < fitness_b)

    def select_food_source_index(self, except_i):
        cumulative_probability = 0
        r = random.random()

        for i in range(len(self.population)):
            cumulative_probability += self.population[i].food_concentration
            if r < cumulative_probability and i != except_i:
                return i
        if except_i == -1:
            return len(self.population)-1
        elif except_i == self.agent_count-1:
            return 0
        return self.agent_count-1

    def employed_phase(self):
        d_rand = math.floor(random.uniform(0, self.agent_length-1))
        q_rand = math.floor(random.uniform(0, self.agent_count-1))
        for i in range(self.agent_count):
            while q_rand == i:
                q_rand = math.floor(random.uniform(0, self.agent_count-1))
            new_v = copy.deepcopy(self.population[i].vector)
            new_v[d_rand] = new_v[d_rand] + random.uniform(-1, 1) * (self.population[i].vector[d_rand] - self.population[q_rand].vector[d_rand])
            self.population[i].set_new_vector(new_v)
            self.population[i].check()
            self.population[i].set_new_fitness()
            if self.population[i].new_fitness > self.population[i].fitness:  # jika nilai fitness baru lebih baik dari nilai fitness sebelumnya
                self.population[i].set_vector(self.population[i].new_vector)  # ganti FS saat ini dengan FS baru
                self.population[i].check()
            else:
                self.population[i].add_abandonment_count()

    def onlook_phase(self):
        # menghitung total fitness seluruh FS
        total_fitness = 0
        arr_fitness = np.array([])
        for i in range(self.agent_count):
            total_fitness += self.population[i].fitness
            arr_fitness = np.append(arr_fitness, [self.population[i].fitness])

        # Menghitung food concentration semua FS
        for i in range(self.agent_count):
            self.population[i].set_food_concentration(self.population[i].fitness / total_fitness)

        onlooker_population = []

        for i in range(self.agent_count):
            random_fs_index = self.select_food_source_index(-1)
            k = self.select_food_source_index(random_fs_index)
            t = random.random()
            fs_vector = self.population[random_fs_index].vector
            k_vector = self.population[k].vector
            v = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)
            v.set_vector(np.array(fs_vector) + t * (np.array(fs_vector) - np.array(k_vector)))
            v.set_fitness()
            v.set_food_concentration(total_fitness)

            if self.is_a_better_than_b(v.fitness, self.population[random_fs_index].fitness):
                onlooker_population.append(Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing))
                onlooker_population[i].set_vector(v.vector)
                onlooker_population[i].set_fitness()
                onlooker_population[i].set_food_concentration(total_fitness)
                # self.population[i].set_vector(v.vector)
                # self.population[i].set_fitness()
                # self.population[i].set_food_concentration(total_fitness)
            else:
                # self.population[i].add_abandonment_count()
                agent = copy.deepcopy(self.population[random_fs_index])
                onlooker_population.append(agent)

    def scout_phase(self):
        for i in range(self.agent_count):
            if self.population[i].abandonment_count > self.MAX_ABANDONMENT_COUNT:
                self.population[i] = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)
                self.population[i].generate_random_vector()
                self.population[i].set_fitness()

    def find_best(self):
        for i in range(self.agent_count):
            fitness = self.population[i].fitness
            if self.FBest is None or self.is_a_better_than_b(fitness, self.FBest):
                self.FBest = fitness
                self.XBest = self.population[i].vector


class HABC(ABC):
    def get_E(self):
        E0 = random.uniform(-1, 1)
        return 2 * E0 * (1 - self.t/self.T)

    def get_Y(self, agent, is_soft_besiege):
        E = self.get_E()
        r = random.uniform(-2, 0)
        J = 2*(1 - r)

        X = np.array(agent)
        Xbest = np.array(self.XBest)

        Y = None
        if is_soft_besiege:
            Y = Xbest - E*np.absolute(J*Xbest - X)
        else:
            mat = np.zeros((self.agent_count, self.agent_length))
            for i in range(self.agent_count):
                mat[i] = self.population[i].vector
            Xm = np.mean(mat, axis=0)
            Y = Xbest - E*np.absolute(J*Xbest - Xm)
        return Agent.fix_outside_limit(self.lower_bound, self.upper_bound, np.ndarray.tolist(Y))

    def LF(self, D):
        a, b = 10, 0
        return levy.pdf(a, b, D)

    def get_Z(self, Y):
        S = self.upper_bound * np.random.rand(self.agent_length)
        D = self.upper_bound * np.random.rand(self.agent_length)
        LFD = self.LF(D)
        Z = np.add(Y, np.multiply(S, LFD))
        return Agent.fix_outside_limit(self.lower_bound, self.upper_bound, np.ndarray.tolist(Z))

    def onlook_phase(self):
        E = self.get_E()

        for i in range(self.agent_count):
            vector_Y = self.get_Y(self.population[i].vector, E >= 0.5)
            vector_Z = self.get_Z(vector_Y)

            Y = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)
            Z = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)

            Y.set_vector(vector_Y)
            Y.set_fitness()

            Z.set_vector(vector_Z)
            Z.set_fitness()

            if Y.fitness > self.population[i].fitness:
                self.population[i].set_vector(vector_Y)
                self.population[i].set_fitness()
            elif Z.fitness > self.population[i].fitness:
                self.population[i].set_vector(vector_Z)
                self.population[i].set_fitness()

    def scout_phase(self):
        for i in range(self.agent_count):
            middle_value = (self.upper_bound + self.lower_bound) / 2
            if self.population[i].abandonment_count > self.MAX_ABANDONMENT_COUNT:
                # generate solusi baru
                new_agent = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)
                new_agent.generate_random_vector()
                new_agent.set_fitness()

                # generate reverse solution
                reverse_solution = np.subtract(self.upper_bound, new_agent.vector)
                reverse_agent = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)
                reverse_agent.set_vector(reverse_solution)
                reverse_agent.check()
                reverse_agent.set_fitness()

                # generate cauchy reverse learning
                cauchy_reverse_solution = []
                for j in range(len(reverse_solution)):
                    cauchy_reverse_solution.append(random.uniform(middle_value, reverse_solution[j]))
                cauchy_reverse_agent = Agent(self.agent_length, self.lower_bound, self.upper_bound, self.f, self.is_maximizing)
                cauchy_reverse_agent.set_vector(cauchy_reverse_solution)
                cauchy_reverse_agent.check()
                cauchy_reverse_agent.set_fitness()

                if new_agent.fitness > reverse_agent.fitness > cauchy_reverse_agent.fitness:
                    self.population[i] = new_agent
                elif reverse_agent.fitness > cauchy_reverse_agent.fitness:
                    self.population[i] = reverse_agent
                else:
                    self.population[i] = cauchy_reverse_agent

#%%


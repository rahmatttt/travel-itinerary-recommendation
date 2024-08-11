from optimization.other.vns import VNS
from optimization.other.algorithm import Algorithm

import copy
import datetime
import random
import math

class WOA:
    def __init__(self, fitness, agents, lower_bound, upper_bound, max_iter, agent_count, N, is_maximizing=False):
        self.fitness_function = fitness
        self.agents = agents
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.agent_count = agent_count
        self.is_maximizing = is_maximizing
        self.N = N

    def get_best_agent(self):
        Fbest = self.fitness_function(self.agents[0])
        Xbest = self.agents[0]
        for agent in self.agents:
            fitness = self.fitness_function(agent)
            if (self.is_maximizing and fitness > Fbest) or ((not self.is_maximizing) and fitness < Fbest):
                Fbest = fitness
                Xbest = copy.deepcopy(agent)
        return Fbest, Xbest

    def get_top_n_agents(self, n):
        XbestN = copy.deepcopy(self.agents)
        XbestN.sort(key=lambda agent: self.fitness_function(agent), reverse=self.is_maximizing)
        return XbestN[:n]

    def execute(self, is_use_vns):
        random.seed(5454)
        start_time = datetime.datetime.now()
        # setup VNS
        vns = VNS(800, self.fitness_function, True)

        t = 0
        l = random.random()-1
        b = 1

        # Fbest : nilai fitness terbaik
        # Xbest : agen terbaik
        Fbest, Xbest = self.get_best_agent()
        agent_dimension = len(Xbest)
        fitness_values = []

        # Menyimpan nilai fitness untuk setiap agen
        for i in range(len(self.agents)):
            fitness_values.append(self.fitness_function(self.agents[i]))

        # print("Initial best fitness = %.5f" % Fbest)
        while t < self.max_iter:

            a = 2 * (1 - t / self.max_iter)

            i = 0
            for agent in self.agents:
                p = random.random()
                r = random.random()
                A = 2 * a * r - a
                C = 2 * r

                D = [0.0 for k in range(agent_dimension)]
                D1 = [0.0 for k in range(agent_dimension)]
                Xnew = [0.0 for k in range(agent_dimension)]
                Xrand = [0.0 for k in range(agent_dimension)]

                if p < 0.5:
                    if abs(a) >= 1: # search for prey
                        p = random.randint(0, self.agent_count-1)
                        while p == i:
                            p = random.randint(0, self.agent_count-1)

                        Xrand = self.agents[p]

                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                    else:  # encircling prey
                        if r >= 0.5 and is_use_vns:
                            Xnew = vns.vns(agent)
                        else:
                            for j in range(agent_dimension):
                                D[j] = abs(C * Xrand[j] - agent[j])
                                Xnew[j] = Xrand[j] - A * D[j]
                else:  # bubble net attacking
                    if r >= 0.5 and is_use_vns:
                        Xnew = vns.vns(agent)
                    else:
                        for j in range(agent_dimension):
                            D[j] = abs(Xbest[j] - agent[j])
                            Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

                for j in range(agent_dimension):
                    agent[j] = Xnew[j]
                i += 1

            for i in range(len(self.agents)):
                # jika Xnew < minx atau Xnew > maxx
                for j in range(agent_dimension):
                    self.agents[i][j] = max(self.agents[i][j], self.lower_bound)
                    self.agents[i][j] = min(self.agents[i][j], self.upper_bound)

                fitness_values[i] = self.fitness_function(self.agents[i])

                if (self.is_maximizing and fitness_values[i] > Fbest) or ((not self.is_maximizing) and fitness_values[i] < Fbest):
                    Xbest = copy.copy(self.agents[i])
                    Fbest = fitness_values[i]

            print("Iteration = " + str(t) + " | best fitness = %.5f" % Fbest)
            t += 1
        end_time = datetime.datetime.now()
        print('Duration, ', end_time - start_time, 'seconds')
        return self.get_top_n_agents(self.N), Fbest


class WOA_VRP(Algorithm):
    def run_woa(self):
        woa = WOA(
            self.fitness_function,
            self.agents,
            0,
            10,
            self.MAX_ITERATIONS,
            self.AGENT_COUNT,
            self.n,
            True)
        return woa.execute(False)

    def construct_solution(self):
        Xbests, Fbest = self.run_woa()
        output = self.get_output(Xbests)

        # normalized_fitness = (Fbest - self.FITNESS_VALUE_RANGE[0]) / (self.FITNESS_VALUE_RANGE[1] - self.FITNESS_VALUE_RANGE[0])
        return output, Fbest


class WOA_VNS_VRP(WOA_VRP):
    def run_woa(self):
        woa = WOA(
            self.fitness_function,
            self.agents,
            0,
            10,
            self.MAX_ITERATIONS,
            self.AGENT_COUNT,
            self.n,
            True)
        return woa.execute(True)


class WOA_TSP(WOA_VRP):
    def run_woa(self):
        woa = WOA(
            self.tsp_fitness_function,
            self.agents,
            0,
            10,
            self.MAX_ITERATIONS,
            self.AGENT_COUNT,
            self.n,
            False)
        return woa.execute(False)


class WOA_VNS_TSP(WOA_VRP):
    def run_woa(self):
        woa = WOA(
            self.tsp_fitness_function,
            self.agents,
            0,
            10,
            self.MAX_ITERATIONS,
            self.AGENT_COUNT,
            self.n,
            False)
        return woa.execute(True)


from optimization.other.algorithm import Algorithm

import numpy as np
from numpy.random import default_rng


class FireflyAlgorithm:
    def __init__(self, pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def run(self, function, dim, lb, ub, max_evals, is_maximizing):
        fireflies = self.rng.uniform(lb, ub, (self.pop_size, dim))
        intensity = np.apply_along_axis(function, 1, fireflies)
        print(intensity)
        best = np.min(intensity)
        x_best = fireflies[0]

        evaluations = self.pop_size
        new_alpha = self.alpha
        search_range = ub - lb

        while evaluations <= max_evals:
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if (not is_maximizing and intensity[i] >= intensity[j]) or (is_maximizing and intensity[i] <= intensity[j]):
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        intensity[i] = function(fireflies[i])
                        evaluations += 1
                        best = max(intensity[i], best) if is_maximizing else min(intensity[i], best)
                        if (is_maximizing and intensity[i] > best) or (not is_maximizing and intensity[i] < best):
                            x_best = fireflies[i]
                        print(best)
        return x_best, best


selected_poi = ['18','8','13','28','27','25','20','53','49','46','76','71','75','72','67','65','63','61','60','99','79']


class FA_VRP(Algorithm):
    def run_FA(self):
        fa = FireflyAlgorithm()
        x_best, f_best = fa.run(self.fitness_function, len(selected_poi), 0, 10, 2, True)
        return x_best, f_best

    def construct_solution(self):
        x_best, f_best = self.run_FA()
        output = self.get_output([x_best])


class FA_TSP(FA_VRP):
    def run_FA(self):
        fa = FireflyAlgorithm()
        x_best, f_best = fa.run(self.tsp_fitness_function, len(selected_poi), 0, 10, 2, False)
        return x_best, f_best



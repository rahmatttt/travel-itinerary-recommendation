from optimization.other.algorithm import Algorithm
from optimization.other.hive.Hive import ABC, HABC


class ABC_VRP(Algorithm):
    def run_ABC(self):
        model = ABC(
            self.MAX_ITERATIONS,
            self.fitness_function,
            self.AGENT_COUNT,
            len(self.PREFERENCE_ID),
            0,
            10,
            True,
            self.n)
        Xbest, Fbest = model.run()
        return Xbest, Fbest

    def construct_solution(self):
        Xbests, Fbest = self.run_ABC()
        output = self.get_output(Xbests)

        # normalized_fitness = (fitness_value - self.FITNESS_VALUE_RANGE[0]) / (self.FITNESS_VALUE_RANGE[1] - self.FITNESS_VALUE_RANGE[0])
        return output, Fbest


class HABC_VRP(ABC_VRP):
    def run_ABC(self):
        model = HABC(
            self.MAX_ITERATIONS,
            self.fitness_function,
            10,
            len(self.PREFERENCE_ID),
            0,
            10,
            True,
            self.n)
        Xbest, Fbest = model.run()
        return Xbest, Fbest


class ABC_TSP(ABC_VRP):
    def run_ABC(self):
        model = ABC(
            self.MAX_ITERATIONS,
            self.tsp_fitness_function,
            self.AGENT_COUNT,
            len(self.PREFERENCE_ID),
            0,
            10,
            False,
            self.n)
        Xbest, Fbest = model.run()
        return Xbest, Fbest


class HABC_TSP(ABC_TSP):
    def run_ABC(self):
        model = HABC(
            self.MAX_ITERATIONS,
            self.tsp_fitness_function,
            10,
            len(self.PREFERENCE_ID),
            0,
            10,
            False,
            self.n)
        Xbest, Fbest = model.run()
        return Xbest, Fbest

# w = ABC_TSP(['37', '22', '16', '18', '36', '10', '5', '17', '96', '12', '6', '94', '44', '40', '24', '2', '39', '20', '8', '26'], 129, 1, 0.5, 0.5, 3, 2)

# w.construct_solution()

#%%


from optimization.other.algorithm import Algorithm
from optimization.other.cso.cso import CSO


class CUCKOO_VRP(Algorithm):
    def run_cso(self):
        bound = [(0, 10)] * len(self.PREFERENCE_ID)
        cso = CSO(
            fitness=self.fitness_function,
            bound=bound,
            min=False,
            P=5,
            n=len(self.PREFERENCE_ID),
            verbose=True,
            Tmax=self.MAX_ITERATIONS)
        return cso.execute()

    def construct_solution(self):
        Xbest, Fbest = self.run_cso()
        output = self.get_output(Xbest)
        return output


class CUCKOO_TSP(CUCKOO_VRP):
    def run_cso(self):
        bound = [(0, 10)] * len(self.PREFERENCE_ID)
        cso = CSO(
            fitness=self.tsp_fitness_function,
            bound=bound,
            min=True,
            P=5,
            n=len(self.PREFERENCE_ID),
            verbose=True,
            Tmax=self.MAX_ITERATIONS)
        return cso.execute()

#%%

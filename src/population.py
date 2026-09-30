import random
from problem    import Problem
from individual import Individual

class Population:
    def __init__(self, pop_size):
        self.pop_size  = pop_size
        self.list_indi = []

    def gen_pop(self, problem: Problem):
        for i in range(self.pop_size):
            indi = Individual()
            indi.gen_indi(problem)
            indi.cal_fitness(problem)
            self.list_indi.append(indi)

    def selection(self, k = 4):
        tour1 = random.sample(self.list_indi, k)
        tour2 = random.sample(self.list_indi, k)
        p1    = max(tour1, key=lambda indi: indi.fitness)
        p2    = max(tour2, key=lambda indi: indi.fitness)
        return p1, p2 
    
    def best_individual(self):
        return max(self.list_indi, key=lambda indiv: indiv.fitness)
    
    def survival_selection(self):
        list_indi = sorted(self.list_indi, key=lambda indi: indi.fitness, reverse=True)
        self.list_indi = list_indi[0: self.pop_size]
    
    def __repr__(self):
        res  = f"[INFO]\n"
        res += f"pop_size: {self.pop_size} \n"
        for i in range(len(self.list_indi)):
            res += f"Individual {i}: {self.list_indi[i]}\n"
        return res



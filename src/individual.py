import numpy as np
from problem import Problem

INF = 1e24

def encode(problem: Problem):
    return np.random.rand(4)

def decode(chromosome, problem: Problem):
    R_R = problem.R_R
    R_L = problem.R_L
    X_R = problem.X_R
    X_L = problem.X_L

    a = chromosome[0] * (R_R[-1] - R_R[0]) + R_R[0]
    b = chromosome[1] * (X_R[-1] - X_R[0]) + X_R[0]
    x = chromosome[2] * (R_L[-1] - R_L[0]) + R_L[0]
    y = chromosome[3] * (X_L[-1] - X_L[0]) + X_L[0]

    Z_R = complex(a, b)
    Z_L = complex(x, y)
    return Z_R, Z_L 

def get_fitness(chromosome, problem: Problem):
    Z_R, Z_L = decode(chromosome, problem)
    return problem.SNR(Z_R, Z_L)

class Individual:
    def __init__(self):
        self.chromosome = None
        self.fitness    = None
        # Cho PSO
        self.velocity   = None
        self.pbest      = None  # Personal best position
        self.pbest_fitness = -INF  # Personal best fitness

    def gen_indi(self, problem : Problem):
        self.chromosome = encode(problem)
    
    def cal_fitness(self, problem: Problem):
        self.fitness = get_fitness(self.chromosome, problem)

    def decode(self, problem: Problem):
        return decode(self.chromosome, problem)
    
    # Simulated binary crossover - SBX
    def crossover(self, other, eta = 2.0):
        r = np.random.rand()
        if (r <= 0.5):
            beta = (2*r)**(1.0/(eta + 1))
        else:
            beta = (1.0/(2*(1 - r)))**(1.0/(eta + 1))

        p1   = self.chromosome
        p2   = other.chromosome
        chr1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        chr2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        
        off1 = Individual()
        off2 = Individual()
        off1.chromosome = np.clip(chr1, 0.0, 1.0)
        off2.chromosome = np.clip(chr2, 0.0, 1.0)
        return off1, off2

    # Polynomial mutaion - PM
    def mutation(self, eta = 20.0):
        chr = self.chromosome
        for i in range(chr.size):
            mu = np.random.rand()
            if (mu <= 0.5):
                delta = (2 * mu)**(1.0/(1 + eta)) - 1
                chr[i] = chr[i] + delta * chr[i]
            else:
                delta = 1 - (2 - 2*mu)**(1.0/(1 + eta))
                chr[i] = chr[i] + delta * (1 - chr[i])
                
        chr = np.clip(chr, 0.0, 1.0)
        self.chromosome = chr

    def de_mutation(self, x1, x2, x3, F=0.8):
        """Differential mutation: v = x1 + F * (x2 - x3)"""
        mutant = x1.chromosome + F * (x2.chromosome - x3.chromosome)
        self.chromosome = np.clip(mutant, 0.0, 1.0)
    
    def de_crossover(self, target, CR=0.9):
        """Binomial crossover cho DE
        target: individual hiện tại
        CR: crossover rate
        """
        trial = Individual()
        trial.chromosome = np.copy(target.chromosome)
        
        j_rand = np.random.randint(0, target.chromosome.size)
        for j in range(target.chromosome.size):
            if np.random.rand() < CR or j == j_rand:
                trial.chromosome[j] = self.chromosome[j]
        
        return trial

    def pso_update(self, gbest, w=0.7, c1=1.5, c2=1.5):
        """Cập nhật vị trí và vận tốc cho PSO
        w: inertia weight
        c1: cognitive parameter
        c2: social parameter
        """
        if self.velocity is None:
            self.velocity = np.random.uniform(-1, 1, size=self.chromosome.shape)
        
        if self.pbest is None:
            self.pbest = self.chromosome.copy()
            self.pbest_fitness = self.fitness
        
        # Cập nhật personal best
        if self.fitness > self.pbest_fitness:
            self.pbest = self.chromosome.copy()
            self.pbest_fitness = self.fitness
        
        # Cập nhật vận tốc
        r1 = np.random.rand(self.chromosome.size)
        r2 = np.random.rand(self.chromosome.size)
        
        self.velocity = (w * self.velocity + 
                        c1 * r1 * (self.pbest - self.chromosome) +
                        c2 * r2 * (gbest.chromosome - self.chromosome))
        
        # Cập nhật vị trí
        self.chromosome = self.chromosome + self.velocity
        self.chromosome = np.clip(self.chromosome, 0.0, 1.0)

    def __repr__(self):
        return f"chromosome={self.chromosome}, fitness={self.fitness:.2f})"
    
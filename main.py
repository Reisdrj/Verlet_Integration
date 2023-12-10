import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean

p = 258
c = 9
M = 258
N = 2 << 11

B, C = (1, -1)

a = -1
b = 1
mi = 0.1
y0 = 1 

beta = 0.1
k = 60

x = np.arange(a, b, (b-a)/M)
vx = np.array([[np.power(x[i],j) for j in range(c)] for i in range(M)])
vdx = np.array([[j*np.power(x[i],j-1) if j > 0 else 0 for j in range(c)] for i in range(M)])

pop = np.random.rand(p,c)
best_ind = None
best_fit = None




def tournament_selection(group : np.ndarray, k):
    index = np.arange(len(group))
    mask = np.ones_like(group, dtype = bool)
    next_group = []

    while available := np.count_nonzero(mask):
        candidates = np.random.choice(index[mask], min(k,available), replace = False)
        winner = np.argmin(group[candidates])
        next_group.append(winner)
        mask[candidates] = False

    return next_group

def crossover(group : np.ndarray, final_size):
    new_group = np.zeros((final_size, group[0].size))
    new_group[:len(group)] = group
    index = len(group)

    while index < final_size:
        parent_index = np.random.choice(np.arange(len(group)), 2, replace = False)
        parents = group[parent_index]
        cross_point = np.random.randint(len(parents[0]))
        new_group[index][:cross_point] = parents[0][:cross_point]
        new_group[index][cross_point:] = parents[1][cross_point:]
        index += 1

    return new_group

def mutate(gen, rate):
    for i,x in enumerate(gen):
        change = np.random.rand()
        if change < rate:
            gen[i] = np.random.rand(x.shape[0])
    return gen


class GeneticSolver:
    def __init__(self, S0, I0, R0, alpha, beta, p, c, h, maxt, k, b):
        x = np.arange(0, maxt+h, h)
        t = np.array([[np.power(x[i],j) for j in range(c)] for i in range(M)])
        dt = np.array([[j*np.power(x[i],j-1) if j > 0 else 0 for j in range(c)] for i in range(M)])
        
        self.population = np.random.rand(p,3*c)
        self.t = np.c_[t, t, t]
        self.dt = np.c_[dt, dt, dt]
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.p = p
        self.c = c
        self.k = k
        self.b = b
        self.alpha = alpha
        self.beta = beta

        self.best = { 'chromossome' : None, 'fitness' : 0, 't': x[:-1]}

    def run_epoch(self):
        fitness = np.empty(self.p)

        for index,chromossome in enumerate(self.population):
            S, I, R = (self.t).reshape(3, len(self.t), self.c)
            dS, dI, dR = (self.dt).reshape(3, len(self.dt), self.c)

            chromossome = chromossome.reshape(3, self.c)

            S = np.matmul(S,chromossome[0])
            I = np.matmul(I,chromossome[1])
            R = np.matmul(R,chromossome[2])

            dS = -self.alpha*S*I/N
            dI = self.alpha*S*I/N - self.beta*I
            dR = self.beta*I 

            fitness[index] = hmean([self.fit(S,dS,self.S0),
                                   self.fit(I,dI,self.I0),
                                   self.fit(R,dR,self.R0),
                                   np.abs(len(self.t)*p - np.sum(S + I + R))])

            if self.best['chromossome'] is None or self.best['fitness'] > fitness[index]:
                self.best['chromossome'] = chromossome
                self.best['fitness'] = fitness[index]
                self.best['S'] = S
                self.best['I'] = I
                self.best['R'] = R

        selected = self.population[tournament_selection(fitness, self.k)]
        crossed = crossover(selected, len(self.population))
        self.population = mutate(crossed, self.b)

    def solve(self, iterations):
        for i in range(iterations):
            self.run_epoch()
        return self.best
    
    @staticmethod
    def fit(y, dy, y0):
        return (np.sum(np.abs(dy - y)) + np.abs(y0 - y[0])) 
    
N = 2 << 5
days = 500
M = 258

solver = GeneticSolver(N-1, 1, 0, 0.3, 0.1, 258, 9, days / M, days, 60, 0.1)
ans = solver.solve(N)

plt.plot(ans['t'], ans['S'], label = 'S')
plt.plot(ans['t'], ans['I'], label = 'I')
plt.plot(ans['t'], ans['R'], label = 'R')
plt.legend()
plt.show()



'''
def epoch():
    global pop
    global best_ind
    global best_fit

    cand = []
    fit = []

    local_min_fit = None

    for ind in pop:
        y = np.matmul(vx,ind)
        dy = np.matmul(vdx,ind)
        fitness = (np.sum(np.abs(B*dy + C*y)) + np.abs(y0 - y[0]))
        
        cand.append(ind)
        fit.append(fitness)
        
        if not local_min_fit or local_min_fit > fitness:
            local_min_fit = fitness
        
        if not best_fit or best_fit > fitness:
            best_fit = fitness
            best_ind = ind

    cand = np.array(cand)
    fit = np.array(fit)

            S, dS = [ np.matmul(S,chromossome[0]), np.matmul(dS, chromossome[0]) ]
            I, dI = [ np.matmul(I,chromossome[1]), np.matmul(dI, chromossome[1]) ]
            R, dR = [ np.matmul(R,chromossome[2]), np.matmul(dR, chromossome[2]) ]
    selected = cand[tournament_selection(fit, k)]
    new_gen = crossover(selected, len(cand))
    pop = mutate(new_gen, beta)


    print(best_fit, local_min_fit)


for i in range(N) : epoch()

print(best_ind)
print(np.matmul(vx,best_ind))

import matplotlib.pyplot as pltN = 2 << 11
days = 500
M = 258
GeneticSolve(N-1, 1, 0, 258, 9, days / M, days, 60, 0.1)

y = np.matmul(vx,best_ind)
plt.plot(x, y)
plt.show()


tmp = np.array([[1,2],[2,1],[3,2],[1,5]])
aux = np.array([1,2,3,4])
selected = tmp[tournament_selection(aux, 2)]
print(selected)
crossover(selected, len(tmp))'''

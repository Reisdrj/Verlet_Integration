import numpy as np

p = 258
c = 9
M = 258
N = 2 << 11

B, C = (1, -1)

a = -1
b = 1
beta = 0.6
mi = 0.1
y0 = 1 

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
        winner = np.min(group[candidates])
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
    return gen
            


def epoch():
    global pop
    global best_ind
    global best_fit

    cand = []

    local_min_fit = None

    for ind in pop:
        y = np.matmul(vx,ind)
        dy = np.matmul(vdx,ind)
        fitness = (np.sum(np.abs(B*dy + C*y)) + np.abs(y0 - y[0]))
        cand.append((fitness, ind))
        
        if not local_min_fit or local_min_fit > fitness:
            local_min_fit = fitness
        
        if not best_fit or best_fit > fitness:
            best_fit = fitness
            best_ind = ind

    cand = np.array(cand)

    

    cand.sort(key = lambda p : p[0])
    cand = cand[:(len(cand)+1) // 2]
    cand = [c[1] for c in cand]

    glen = len(cand) 
    moms = np.random.choice(range(glen), len(cand) // 2)
    dads = np.array([c for c in range(glen) if not c in moms])

    qmate = min(len(moms), len(dads))
    for i in range(qmate):
        dad, mom = cand[dads[i]], cand[moms[i]]
        cand.append(beta*(dad - mom) + dad)
        cand.append(beta*(mom - dad) + mom)

    if len(moms) > qmate: cand.append(cand[moms[-1]])
    elif len(dads) > qmate: cand.append(cand[dads[-1]])

    '''mut = np.random.choice(np.arange(0,p*c), int(0.1*p*c))
    for m in mut:
        cand[m // c][m % c] = a + (b - a) * np.random.rand()'''

    pop = np.array(cand)
    print(best_fit, local_min_fit)
    #print(pop)

'''for i in range(N) : epoch()

print(best_ind)
print(np.matmul(vx,best_ind))

import matplotlib.pyplot as plt

y = np.matmul(vx,best_ind)
plt.plot(x, y)
plt.show()'''

tmp = np.array([[1,2],[2,1],[3,2],[1,5]])
aux = np.array([1,2,3,4])
selected = tmp[tournament_selection(aux, 2)]
print(selected)
crossover(selected, len(tmp))

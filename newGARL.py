
"""
Created on Tue Apr 18 17:06:38 2023

@author: sadegh malekshahi
"""
import numpy as np
#import cupy as cp
import itertools
import json
#import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ypstruct import *
import random

class RLagent():
    def __init__(self, alpha, gamma, epsilon, cRange, mRange, crossover, mutation, fitness):
        ##################################################################################################################
        #------------------------------------------------ Hyperparameters ------------------------------------------------
        # the learning rate
        self.alpha = alpha

        # the discount rate
        self.gamma = gamma

        # the exploration rate
        self.epsilon = epsilon
    
        ##################################################################################################################
        #--------------------------------------- action, state, and reward spaces ----------------------------------------
        # get all combinations of crossover and mutation probabilites and convert it to a list 
    #Examples
    # itertools.product('ABCD', repeat=2)
    # AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
        self.actionSpace = list(itertools.product(cRange, mRange))

        # get the rewards table
        self.rewardSpace = np.array([200,   150,   100,   50,    25,
                                     150,   113,   75,    38,    19,
                                     100,   75,    50,    25,    13,
                                     50,    38,    25,    113,   7,
                                     0,     0,    -10,   -20,   -30,
                                    -1000, -1500, -2000, -2500, -3000])
       
        # a dictionary of all possible states, where the state is the key, and the value is the index for 
        #the q table and rewards space
        #(HH, VHD)-->int 0
        #(HH, HD)-->int 1
        self.stateSpace = { '(HH, VHD)': 0, '(HH, HD)':1,  '(HH, MD)':2,  '(HH, LD)':3,  '(HH, VLD)':4,
                            '(H, VHD)':5,   '(H, HD)':6,   '(H, MD)':7,   '(H, LD)':8,   '(H, VLD)':9,
                            '(L, VHD)':10,  '(L, HD)':11,  '(L, MD)':12,  '(L, LD)':13,  '(L, VLD)':14,
                            '(LL, VHD)':15, '(LL, HD)':16, '(LL, MD)':17, '(LL, LD)':18, '(LL, VLD)':19,
                            '(S, VHD)':20,  '(S, HD)':21,  '(S, MD)':22,  '(S, LD)':23,  '(S, VLD)':24,
                            '(I, VHD)':25,  '(I, HD)':26,  '(I, MD)':27,  '(I, LD)':28,  '(I, VLD)':29}

        # initilize the Q-table
        self.Q = np.zeros([len(self.stateSpace), len(self.actionSpace)])

        ##################################################################################################################
        # ----------------------------------------------- initialization  ------------------------------------------------
        # a variable keeping track of how much rewards it has recieved
        self.collected  = 0

        # create an array to keep count how often each action was taken
        self.actionCount = np.zeros(len(self.actionSpace))

        # the previous fitness variable is initilized with a verh high cost
        self.prevFitness = fitness
        
        # the current fitness delta
        self.fitness = 0

        # the current diversity index
        self.diversity = 1
        
        # the current reward awarded
        self.reward = 0

        # initialize the first state (high cost, and very high diversity)
        self.currState = 0

        # the first actions are given
        self.action = self.actionSpace.index((crossover, mutation))
        
        self.store_results=[]
        # # initialie the json file
        # path = 'results/SARSA/agent/'
        # self.json = path + problem + '_agent_' + run + '.json'
        # #self.json = path + problem + '_agent_' + str(len(os.listdir(path))) + '.json'
        # with open(self.json, 'w') as f:
        #     json.dump({}, f)

    #INPUT: 0 or 1
    #OUTPUT: the value if 1 and the index if 0.
    #If there are several max values, then a single one is choosen arbitrary
    def __max(self, out, arr):
        # hold any ties found
        ties = []

        # set an initial top value
        top = float('-inf')

        # for each element in the array
        for i in range(len(arr)):

            # if the current value is the new highest value
            if arr[i] > top:

                # then reset the tie list
                ties = []

                # set the new top value
                top = arr[i]

                # add the top value to the tie list
                ties.append([i, arr[i]])

            # else if the current value is tied to the highest value
            elif arr[i] == top:

                # then add it to the tie list
                ties.append([i, arr[i]])
        
        # pick a random index
        choice = np.random.choice(np.arange(len(ties)))

        # return the desired value
        return ties[choice][out]

    # INPUT: the fitnesses of the current generation
    # OUTPUT: the change in fitness as a percentage (and set the the current fitness as the previous fitness value for the next iteration)
    def __fitness(self, fitnesses):
        # get the min fitness of the population
        bestFitness = np.amin(fitnesses)
        # obtaint the difference between the current and previous fitness values
        delta = self.prevFitness - bestFitness
        
        # the difference is divided by the previous fitness to obtain a percentage
        deltaFitness = delta / self.prevFitness
        
        # the current fitness is set as the previous fitness for the next iteration
        self.prevFitness = bestFitness

        # return the fitness imrpovement as a percenetage
        return deltaFitness

    # INPUT: the population from the enviroment's response
    # OUTPUT: percentage of unique chromosomes in the population
    # def __diversity(self, array):
    #     sortarr     = array[np.lexsort(array.T[::-1])]
    #     mask        = cp.empty(array.shape[0], dtype=cp.bool_)
    #     mask[0]     = True
    #     mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    #     return sortarr[mask]
    def __diversity2(self,array,array2,array3):
        array=np.array(array)
        array2=np.array(array2)
        array3=np.array(array3)
        sortarr = array[np.lexsort(array.T[::-1])]
        mask = np.empty(array.shape[0], dtype=bool)
        mask[0] = True
        mask[1:] = np.any(sortarr[1:] != sortarr[:-1], axis=1)
        sortarr2 = array2[np.lexsort(array2.T[::-1])]
        mask2 = np.empty(array2.shape[0], dtype=bool)
        mask2[0] = True
        mask2[1:] = np.any(sortarr2[1:] != sortarr2[:-1], axis=1)
        sortarr3 = array3[np.lexsort(array3.T[::-1])]
        mask3 = np.empty(array3.shape[0], dtype=bool)
        mask3[0] = True
        mask3[1:] = np.any(sortarr3[1:] != sortarr3[:-1], axis=1)
        x=len(sortarr[mask])*100/len(array)
        y=len(sortarr2[mask2])*100/len(array2)
        z=len(sortarr3[mask3])*100/len(array3)
        m=(x+y+z)/3
        print("DIIIII IS = {}".format(m))
        return z
    # #Example input array
    # input_array = np.array([[3, 2, 1], [3, 2, 1], [9, 8, 7]])
    # result_numpy = len(__diversity_numpy(input_array))*100/len(input_array)
    # print(result_numpy)

    # INPUT: the fitness delta and diversity index of the current population
    # OUTPUT: the numberical valeus covnerted into categorical values as a tuple used to represent the state
    def __state(self, fitness, diversity):
        # an if statment to convert numerical values into into categorical bins
        if fitness < 0:
            fState = 'I'
        elif fitness == 0:
            fState = 'S'
        elif fitness < 0.01:
            fState = 'LL'
        elif fitness < 0.05:
            fState = 'L'
        elif fitness < 0.25:
            fState = 'H'
        else:
            fState = 'HH'

        # an if statment to convert numerical values into into categorical bins
        if diversity <= 20:
            dState = 'VLD'
        elif diversity <= 40:
            dState = 'LD'
        elif diversity <= 60:
            dState = 'MD'
        elif diversity <= 80:
            dState = 'HD'
        else:
            dState = 'VHD'

        # the state is obtained and formatted for latter use
        state = '(' + fState + ', ' + dState + ')'
        
        # the state key is used to find its index in the state space and set as the new state for the agent
        self.nextState = self.stateSpace[state]

    # INPUT: the object's state variable
    # OUTPUT: the reward given the state
    def __reward(self):
        # the reward is look up in the table
        self.reward = self.rewardSpace[self.nextState]

        # the rewards is added to the collection
        self.collected += self.reward

    # used for printing output
    def __findState(self):
        for i in self.stateSpace:
            if self.stateSpace[i] == self.currState:
                return i

    # print and save results from each generation
    def storeresults(self, count):
        store_results=[]
        # a dictionary holding all the results
        action = self.actionSpace[self.action]
        results = {'action':(int(action[0]), int(action[1])) , 'state':self.__findState(), 'diversity index':float(self.diversity), 'fitness delta':int(self.fitness), 'reward':int(self.reward), 'collected':int(self.collected)}        
        #the json file is opened
        # with open(self.json, 'r+') as f:

        #     # the file is loaded up
        #     data = json.load(f)

        #     # the data is updated
        #     data.update({count:results})

        #     # the data is dumped back into the file
        #     f.seek(0)
        #     json.dump(data, f, indent=4)
        
        for i in results:
            print('   ' + i + ':', results[i])
            store_results.append(results[i])
        print()
        return store_results
    # the first action is given
    def initAction(self):
        # reset the action count to disregard the first action
        self.actionCount = np.zeros(len(self.actionSpace))

        # the action count is updated
        self.actionCount[self.action] += 1
        
        # update the results log
        self.storeresults(0)

        # give the enviroment its action (the crossover and mutation probability)
        return self.actionSpace[self.action][0], self.actionSpace[self.action][1]

    # the agent decides an action for the enviroment
    def decide(self, count):
        # randomly decide to explore (with probability epsilon)
        if np.random.random() <= self.epsilon:

            # a random action is chosen
            self.action = int(np.random.randint(low=0, high=len(self.actionSpace)))

        # or exploit (with probability 1 - epsilon)
        else:

            # the max action is chosen
            self.action = int(self.__max(0, (self.Q[self.currState])))
        
        # the action count is updated
        self.actionCount[self.action] += 1

        # print and save the results
        self.storeresults(count)

        # give the enviroment its action (the crossover and mutation probability)
        return self.actionSpace[self.action][0], self.actionSpace[self.action][1]
        
    # the agent observes the enviroment's response to the agent's action
    # def observe(self, envResponse):
    #     # obtain the population and their fitnesses after an action
    #     population = envResponse.position
    #     fitnesses = envResponse.cost
        
    #     # determine the delta of the previous fitness and the current best fitness of the population and the diversity 
    #     self.fitness = self.__fitness(fitnesses)
    #     self.diversity = self.__diversity(population).shape[0]/population.shape[0]
        
    #     # get the new state and rewards
    #     self.__state(self.fitness, self.diversity)
    #     self.__reward()
    def observe2(self, envResponse,bestsol):
        # obtain the population and their fitnesses after an action
        population=[]
        population2=[]
        population3=[]
        for i in range(len(envResponse)):
            population.append(envResponse[i]['position'])
            population2.append(envResponse[i]['pricelevel'])
            population3.append(envResponse[i]['customer'])
        fitnesses = bestsol.cost.obj
        # determine the delta of the previous fitness and the current best fitness of the population and the diversity 
        self.fitness = self.__fitness(fitnesses)
        self.diversity = (self.__diversity2(population,population2,population3))     
        # get the new state and rewards
        self.__state(self.fitness, self.diversity)
        self.__reward()
        return (self.__diversity2(population,population2,population3))
    # the Q table is updated along with other variables for the q learning algorithm
    def updateQlearning(self):
        # update the q table using the bellman equation
        self.Q[self.currState, self.action] += self.alpha * (self.reward + self.gamma * self.__max(1, self.Q[self.nextState]) - self.Q[self.currState, self.action] )

        # update the current state
        self.currState = self.nextState
    def updateQlearning2(self):
        # update the q table using the bellman equation
        self.Q[self.currState, self.action] += self.alpha * (self.reward + self.gamma * self.__max(1, self.Q[self.nextState]) - self.Q[self.currState, self.action] )
        print("q-table is = {}".format(self.Q))
        q_table=self.Q
        # update the current state
        self.currState = self.nextState
        return q_table

############################################################################################################
#---------------------------------------------- for debugging ----------------------------------------------
count=0
def Qlearning(agent,it,pop,bestsol):
        if it==1000:
            return agent.updateQlearning2()
        ########### Choose action
        elif it == 0:
            pc, pm = agent.initAction()
        else:
            count=it/params.epoch
            pc, pm = agent.decide(count)
        # ########### Imitate the enviroment with random population
        # population = np.random.randint(low=1, high=1000, size=(100,100))
        population = pop
        ########### Observe state and rewards
        DIII=agent.observe2(population, bestsol)
        count=it/params.epoch
        store_results= agent.storeresults(count)
        ########### Update the policy
        agent.updateQlearning()
        return pc,pm,store_results,DIII
# Problem Definition
if __name__ == '__main__':
    cRange = np.array(range(1, 11))/10
    mRange = np.array(range(1, 11))/10
    alpha = 0.7
    gamma = 0.1
    epsilon = 0.3
    crossover = cRange[np.random.randint(1,10)]
    mutation = mRange[np.random.randint(1,10)]
    print("crossover {}: mutation = {}".format(crossover, mutation))
    fitness=10000000000000
    agent = RLagent(alpha, gamma, epsilon, cRange, mRange, crossover, mutation,fitness)
# Problem Definition
data=pd.read_excel('demand3ga6dc.xlsx', header=0)
data=np.array(data)
demand=np.zeros((90,6,4))
for i in range(2160):
    demand[data[i,0]-1,data[i,1]-1,data[i,2]-1]=data[i,3]
#
data=pd.read_excel('order3ga6dc.xlsx', header=0)
data=np.array(data)
order=data
# flowshop scheduling Test Function
def mycost(loc,s,price):

    transport=[0,0.17,0.002,0.25,0.177,0.105]
    penaltycost=90
    pricee=[70,130,250,0]
    numberofprices=len(problem.pricee)
    alfa=[1,1,1]
    numberofmachines=len(problem.alfa)
    numberofjobs=len(demand[1,:,1])
    numberofpositions=len(demand[1,:,1])
    numberofcustomers=len(demand[:,1,1])
    x=np.zeros((numberofcustomers,numberofpositions,numberofprices,numberofpositions))
    for i in range(numberofcustomers):
        x[i,loc[i],price[i],s[loc[i]]]=1
    demand2=np.zeros(numberofjobs)
    for j in range(numberofjobs):
        for l in range(numberofcustomers):
            for i in range(numberofprices):
                for k in range(numberofpositions):
                    demand2[j]+= demand[l,j,i] * x[l,j,i,k]
    demand3=(np.repeat(demand2[None,:], numberofmachines, axis=0)).transpose()
    C=np.zeros((numberofjobs,numberofmachines))
    s2 = np.sort(s)
    #s3 = [s[index] for index in s2]
    C[s2[0],0]=demand3[s2[0],0]
    for i in range(1,numberofjobs):
        C[s2[i],0]=C[s2[i-1],0]+demand3[s2[i],0]
    #completion time of first job on machine j 
    #is equal to completion time of first job 
    #on machinej-1 plus duration(firstjob, j)
    for j in range(1,numberofmachines):
        C[s2[0],j]=C[s2[0],j-1]+demand3[s2[0],j]
    for i in range(1,numberofjobs):
        for j in range(1,numberofmachines):
            C[s2[i],j]=max(C[s2[i-1],j],C[s2[i],j-1])+demand3[s2[i],j]
    profit=0
    for l in range(0,numberofcustomers):
        for i in range(0,numberofprices):
            for j in range(0,numberofjobs):
                for k in range(0,numberofpositions):
                    profit+=pricee[i]*demand[l,j,i]*x[l,j,i,k]
    transportcost=0
    for l in range(0,numberofcustomers):
        for i in range(0,numberofprices-1):
            for j in range(0,numberofjobs):
                for k in range(0,numberofpositions):
                    transportcost+=(transport[j]+order[l,j])*demand[l,j,i]*x[l,j,i,k]
    cmax=max(C[:,numberofmachines-1])
    obj=(penaltycost*cmax)-profit   
    obj=struct()
    obj.profit=profit
    obj.transportcost=transportcost
    obj.cmax=cmax
    obj.obj=(penaltycost*cmax)-profit
    return obj
#
#priceleveloptimum=np.array(pd.read_excel('multiproduct30spt3price.xlsx', header=0,sheet_name='priceleveloptimum'))
#positionoptimum=np.array(pd.read_excel('multiproduct30spt3price.xlsx', header=0,sheet_name='positionoptimum'))
problem = structure()
problem.costfunc = mycost
problem.demand=demand
problem.order=order
problem.transport=[0,0.17,0.002,0.25,0.177,0.105]
problem.pricee=[70,130,250,0]
problem.numberofprices=len(problem.pricee)
problem.alfa=[1,1,1]
problem.numberofmachines=len(problem.alfa)
problem.numberofjobs=len(demand[1,:,1])
problem.numberofpositions=len(demand[1,:,1])
problem.numberofcustomers=len(demand[:,1,1])

# GA Parameters
params = structure()
params.maxit = 1001
params.npop = 100
params.beta = 1
params.epoch=5

def run(problem, params):
    
    # Problem Information
    costfunc = problem.costfunc
    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta

    ######### az inja ############
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None
    empty_individual.pricelevel = None
    empty_individual.customer = None
    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):  
        pop[i].customer,pop[i].position, pop[i].pricelevel = create_random_solution(problem)
        pop[i].cost = costfunc(pop[i].customer,pop[i].position, pop[i].pricelevel)
        if i==0:
            bestsol = pop[i].deepcopy()  
        elif (pop[i].cost).obj < (bestsol.cost).obj:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iterations
    # for i in range(npop):
    #      print(pop[i].position)
    bestcost = np.empty(maxit)
    
    # Main Loop
    store_results2=[]
    DIII2=[]
    for it in range(maxit):
        q_table=[]
        if it%params.epoch==0:
            if it==1000:
                q_table=Qlearning(agent,it,pop,bestsol)
                break
            else:
                (pc,pm,store_results,DIII)=Qlearning(agent,it,pop,bestsol)
        print("pc {}: pm = {}".format(pc, pm))
        store_results2.append(store_results)
        DIII2.append(DIII)
        nc = int(np.round(pc*npop/2)*2)
        nm=int(np.round(pm*npop))
        costs = np.array([x.cost.obj for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)

        popc = []
        popm= []
        for _ in range(nc//2):

            # Select Parents
            #q = np.random.permutation(npop)
            #p1 = pop[q[0]]
            #p2 = pop[q[1]]

            # Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
            c1 = p1.deepcopy()
            c2 = p1.deepcopy()
            
            # Perform Crossover
            c1.customer, c2.customer = crossover(p1.customer, p2.customer)
            c1.pricelevel, c2.pricelevel = crossover(p1.pricelevel, p2.pricelevel)
            c1.position, c2.position = permutationcrossover(p1.position, p2.position)
            # Evaluate First Offspring
            c1.cost = costfunc(c1.customer,c1.position, c1.pricelevel)
            if c1.cost.obj < bestsol.cost.obj:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.customer,c2.position, c2.pricelevel)
            if c2.cost.obj < bestsol.cost.obj:
                bestsol = c2.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)

        for k in range(nm):
            i=random.randint(0, npop-1)
            p=pop[i]
            c=p.deepcopy()
            c.customer = mutate(c.customer)
            c.pricelevel = mutate(c.pricelevel)
            c.position = mutate(c.position)
            c.cost = costfunc(c.customer,c.position, c.pricelevel)
            if c.cost.obj < bestsol.cost.obj:
                bestsol = c.deepcopy()
            # Add Offsprings to popc
            popm.append(c)
         
        # Merge, Sort and Select
        pop += popc+popm
        pop = sorted(pop, key=lambda x: x.cost.obj)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost.obj

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))
    # Output
    out = struct()
    #out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    out.pop=pop
    out.store_results=store_results2
    out.q_table=q_table
    out.DIII=DIII2
    return out

def crossover(p1, p2):

    nVar=len(p1)
    c=np.random.randint(0,nVar-2)
    x11 = p1[0:c]
    x12=p1[c:]
    x21 = p2[0:c]
    x22=p2[c:]
    c1=np.append(x11,x22)
    c2=np.append(x21,x12)
    return c1, c2
def permutationcrossover(p1, p2):
    nVar=len(p1)
    c=np.random.randint(0,nVar-2)
    x11 = p1[0:c]
    x12=p1[c:]
    x21 = p2[0:c]
    x22=p2[c:]
    r1=np.intersect1d(x11, x22)
    r2=np.intersect1d(x21, x12)
    LX = np.isin(x11, r1)
    LX2= np.isin(x21,r2)
    x11=np.array([x11])
    x12=np.array([x12])
    x21=np.array([x21])
    x22=np.array([x22])
    LX=np.array([LX])
    LX2=np.array([LX2])
    if np.size(r1)>0:
        x11[LX]=r2
        x21[LX2]=r1
    c1=np.append(x11,x22)
    c2=np.append(x21,x12)
    return c1, c2
def mutate(x):
    m=np.random.randint(1,4)
    if m==1:
        n=len(x)
        n2=list(range(0,n))
        i=random.sample(n2,2)
        i1=i[0]
        i2=i[1]
        x[i1],x[i2]=x[i2],x[i1]
        y=x
    elif m==2:
        y=x
        n=len(x)
        n2=list(range(0,n))
        i=random.sample(n2,2)
        i1=min(i[0],i[1])
        i2=max(i[0],i[1])
        y[i1-1:i2]=np.flip(x[i1-1:i2])
    else:
        y=x
        n=len(x)
        n2=list(range(0,n))
        i=random.sample(n2,2)
        i1=i[0]
        i2=i[1]
        if i1==0:
            y=np.append(np.append(x[1:i2+1],x[0]),x[i2+1:])
        elif i2==0:
            y=np.append(np.append(np.append(x[0],x[i1]),x[i2+1:i1]),x[i1+1:])
        elif i1<i2:
            y=np.append(np.append(np.append(x[0:i1-1],x[i1:i2]),x[i1-1]),x[i2:])
            #y=np.append(np.append(np.append(x[0:i1],x[i1+1:i2]),x[i2]),x[i1])
            #y=np.append(x[0:i1-1], x[i1:i2], x[i1-1], x[i2:])
        else:
            y=np.append(np.append(np.append(x[0:i2],x[i1-1]),x[i2:i1-1]),x[i1:])
            #y=np.append(np.append(x[0:i2+1],x[i1]),x[i2+1:i1])
            #y=[x[0:i2], x[i1-1], x[i2:i1-1], x[i1:]]
    return y

#def apply_bound(x, varmin, varmax):
#    x.position = np.maximum(x.position, varmin)
#    x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
def create_random_solution(problem):
    s=np.random.permutation(problem.numberofjobs)
    price=np.random.randint(problem.numberofprices, size=problem.numberofcustomers)
    loc=np.random.randint(problem.numberofjobs, size=problem.numberofcustomers)
    return(loc,s,price)
# flowshop scheduling Test Function


# Run GA
out = run(problem, params)

# Results
plt.plot(out.bestcost)
# plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
x=out.store_results
xx=out.q_table
#################################################
# problem.penaltycost=123
# params.maxit = 1000
# params.npop = 80
# params.beta = 1
# params.pc = 0.3
########


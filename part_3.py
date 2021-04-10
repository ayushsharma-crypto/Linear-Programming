#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxpy as cp
import numpy as np
import json
import os


# In[2]:


POS = [ "W","N","E","S","C" ]    # position of IJ
MAT = [ 0,1,2 ]    # material with IJ
ARROW = [ 0,1,2,3 ]    # number of arrows
STATE = [ "D","R" ]    # ready and dormant state of MM
HEALTH = [ 0,25,50,75,100 ]    # MM’s health
ACTION = [ "UP","LEFT","DOWN","RIGHT","STAY","SHOOT","HIT","CRAFT","GATHER","NONE" ]    # POSSIBLE ACTIONS OF IJ

TEAM = 25

ARR = [ 0.5, 1, 2 ]
Y = ARR[TEAM % 3]
STEPCOST = -10/Y

GAMMA = 0.999
DELTA = 0.001

BLADEHITDAMAGE = 50
ARROWHITDAMAGE = 25

NEGATEREWARD = 40
os.makedirs('outputs', exist_ok=True)


# In[3]:


all_state = []
for pos in POS:
    for mat in MAT:
        for arrow in ARROW:
            for state in STATE:
                for health in HEALTH:
                    all_state.append((pos,mat,arrow,state,health))


# In[4]:


state_actions = {} # state actions in dict.
for s in all_state:
    if s[-1]==0:
        state_actions[s] = ['NONE']
    elif s[0]=='W':
        state_actions[s] = [ 'RIGHT','STAY']
        if s[2]!=0:
            state_actions[s].append('SHOOT')
    elif s[0]=='N':
        state_actions[s] = [ 'DOWN','STAY']
        if (s[1]>0):
                state_actions[s].append('CRAFT')
    elif s[0]=='E':
        state_actions[s] = [ 'LEFT','STAY' ]
        if s[2]!=0:
            state_actions[s].append('SHOOT')
        state_actions[s].append('HIT')
    elif s[0]=='S':
        state_actions[s] = [ 'UP','STAY','GATHER' ]
    else:
        state_actions[s] = [ 'UP','LEFT','DOWN','RIGHT','STAY' ]
        if s[2]>0:
            state_actions[s].append('SHOOT')
        state_actions[s].append('HIT')


# In[5]:


def get_pfsb(current_state, action):
    pfsb = []
    (pos,mat,arrow,state,health) = current_state
    if state=='D':
        
        if action == 'UP':
            if pos=='C':
                pfsb.append(('N',mat,arrow,'D',health,0.85*0.8))
                pfsb.append(('N',mat,arrow,'R',health,0.85*0.2))
            else:
                pfsb.append(('C',mat,arrow,'D',health,0.85*0.8))
                pfsb.append(('C',mat,arrow,'R',health,0.85*0.2))
            pfsb.append(('E',mat,arrow,'D',health,0.15*0.8))
            pfsb.append(('E',mat,arrow,'R',health,0.15*0.2))
            
        elif action == 'DOWN':
            if pos=='C':
                pfsb.append(('S',mat,arrow,'D',health,0.85*0.8))
                pfsb.append(('S',mat,arrow,'R',health,0.85*0.2))
            else:
                pfsb.append(('C',mat,arrow,'D',health,0.85*0.8))
                pfsb.append(('C',mat,arrow,'R',health,0.85*0.2))
            pfsb.append(('E',mat,arrow,'D',health,0.15*0.8))
            pfsb.append(('E',mat,arrow,'R',health,0.15*0.2))
            
        elif action == 'LEFT':
            if pos=='C':
                pfsb.append(('W',mat,arrow,'D',health,0.85*0.8))
                pfsb.append(('W',mat,arrow,'R',health,0.85*0.2))
                pfsb.append(('E',mat,arrow,'D',health,0.15*0.8))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.2))
            else:
                pfsb.append(('C',mat,arrow,'D',health,1*0.8))
                pfsb.append(('C',mat,arrow,'R',health,1*0.2))
        
        elif action == 'RIGHT':
            if pos=='C':
                pfsb.append(('E',mat,arrow,'D',health,1*0.8))
                pfsb.append(('E',mat,arrow,'R',health,1*0.2))
            else:
                pfsb.append(('C',mat,arrow,'D',health,1*0.8))
                pfsb.append(('C',mat,arrow,'R',health,1*0.2))
        
        elif action == 'STAY':
            if pos in ['C','S','N']:
                pfsb.append((pos,mat,arrow,'R',health,0.85*0.2))
                pfsb.append((pos,mat,arrow,'D',health,0.85*0.8))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.2))
                pfsb.append(('E',mat,arrow,'D',health,0.15*0.8))
            else:
                pfsb.append((pos,mat,arrow,'R',health,1*0.2))
                pfsb.append((pos,mat,arrow,'D',health,1*0.8))
                
        elif action == 'HIT':
            if pos=='C':
                pfsb.append((pos,mat,arrow,'D',max(health-BLADEHITDAMAGE,0),0.1*0.8))
                pfsb.append((pos,mat,arrow,'D',health,0.9*0.8))
                pfsb.append((pos,mat,arrow,'R',max(health-BLADEHITDAMAGE,0),0.1*0.2))
                pfsb.append((pos,mat,arrow,'R',health,0.9*0.2))
            else:
                pfsb.append((pos,mat,arrow,'D',max(health-BLADEHITDAMAGE,0),0.2*0.8))
                pfsb.append((pos,mat,arrow,'D',health,0.8*0.8))
                pfsb.append((pos,mat,arrow,'R',max(health-BLADEHITDAMAGE,0),0.2*0.2))
                pfsb.append((pos,mat,arrow,'R',health,0.8*0.2))

        elif action == 'SHOOT':
            if arrow==0:
                pfsb.append((pos,mat,arrow,'D',health,1*0.8))
                pfsb.append((pos,mat,arrow,'R',health,1*0.2))
            else:
                if pos=='W':
                    pfsb.append((pos,mat,arrow-1,'D',max(health-ARROWHITDAMAGE,0),0.25*0.8))
                    pfsb.append((pos,mat,arrow-1,'R',max(health-ARROWHITDAMAGE,0),0.25*0.2))
                    pfsb.append((pos,mat,arrow-1,'D',health,0.75*0.8))
                    pfsb.append((pos,mat,arrow-1,'R',health,0.75*0.2))
                elif pos=='C':
                    pfsb.append((pos,mat,arrow-1,'D',max(health-ARROWHITDAMAGE,0),0.5*0.8))
                    pfsb.append((pos,mat,arrow-1,'R',max(health-ARROWHITDAMAGE,0),0.5*0.2))
                    pfsb.append((pos,mat,arrow-1,'D',health,0.5*0.8))
                    pfsb.append((pos,mat,arrow-1,'R',health,0.5*0.2))
                else:
                    pfsb.append((pos,mat,arrow-1,'D',max(health-ARROWHITDAMAGE,0),0.9*0.8))
                    pfsb.append((pos,mat,arrow-1,'R',max(health-ARROWHITDAMAGE,0),0.9*0.2))
                    pfsb.append((pos,mat,arrow-1,'D',health,0.1*0.8))
                    pfsb.append((pos,mat,arrow-1,'R',health,0.1*0.2))
        
        elif action == 'GATHER':
            if mat<2:
                pfsb.append((pos,mat+1,arrow,'D',health,0.75*0.8))
                pfsb.append((pos,mat+1,arrow,'R',health,0.75*0.2))
                pfsb.append((pos,mat,arrow,'D',health,0.25*0.8))
                pfsb.append((pos,mat,arrow,'R',health,0.25*0.2))
            else:
                pfsb.append((pos,mat,arrow,'D',health,1*0.8))
                pfsb.append((pos,mat,arrow,'R',health,1*0.2))
        
        elif action == 'CRAFT':
            if (mat>0):
                if arrow==0:
                    pfsb.append((pos,mat-1,1,'D',health,0.5*0.8))
                    pfsb.append((pos,mat-1,2,'D',health,0.35*0.8))
                    pfsb.append((pos,mat-1,3,'D',health,0.15*0.8))
                    pfsb.append((pos,mat-1,1,'R',health,0.5*0.2))
                    pfsb.append((pos,mat-1,2,'R',health,0.35*0.2))
                    pfsb.append((pos,mat-1,3,'R',health,0.15*0.2))
                elif arrow==1:
                    pfsb.append((pos,mat-1,2,'D',health,0.5*0.8))
                    pfsb.append((pos,mat-1,3,'D',health,0.5*0.8))
                    pfsb.append((pos,mat-1,2,'R',health,0.5*0.2))
                    pfsb.append((pos,mat-1,3,'R',health,0.5*0.2))
                elif arrow in [2,3]:
                    pfsb.append((pos,mat-1,3,'R',health,1*0.2))
                    pfsb.append((pos,mat-1,3,'D',health,1*0.8))
            else:
                pfsb.append((pos,mat,arrow,'D',health,1*0.8))
                pfsb.append((pos,mat,arrow,'R',health,1*0.2))
                    
    else:
        
        if action == 'UP':
            if pos=='C':
                pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                pfsb.append(('N',mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))
            else:
                pfsb.append(('C',mat,arrow,'D',health,0.5))
                pfsb.append(('C',mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))

        elif action == 'DOWN':
            if pos=='C':
                pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                pfsb.append(('S',mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))
            else:
                pfsb.append(('C',mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))
                pfsb.append(('C',mat,arrow,'D',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'D',health,0.15*0.5))
            
        elif action == 'LEFT':
            pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
            if pos=='C':
                pfsb.append(('W',mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))
            else:
                pfsb.append(('C',mat,arrow,'R',health,1*0.5))
        
        elif action == 'RIGHT':
            if pos=='C':
                pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                pfsb.append(('E',mat,arrow,'R',health,1*0.5))
            else:
                pfsb.append(('C',mat,arrow,'R',health,1*0.5))
                pfsb.append(('C',mat,arrow,'D',health,1*0.5))

        elif action == 'STAY':
            if pos=='C':
                pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                pfsb.append((pos,mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))
            
            elif pos=='E':
                pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                pfsb.append(('E',mat,arrow,'R',health,1*0.5))
                
            elif pos in ['S','N']:
                pfsb.append((pos,mat,arrow,'R',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'R',health,0.15*0.5))
                pfsb.append((pos,mat,arrow,'D',health,0.85*0.5))
                pfsb.append(('E',mat,arrow,'D',health,0.15*0.5))
                
            else:
                pfsb.append((pos,mat,arrow,'R',health,1*0.5))
                pfsb.append((pos,mat,arrow,'D',health,1*0.5))
                
        elif action == 'HIT':
            pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
            if pos=='C':
                pfsb.append((pos,mat,arrow,'R',max(health-BLADEHITDAMAGE,0),0.1*0.5))
                pfsb.append((pos,mat,arrow,'R',health,0.9*0.5))
            else:
                pfsb.append((pos,mat,arrow,'R',max(health-BLADEHITDAMAGE,0),0.2*0.5))
                pfsb.append((pos,mat,arrow,'R',health,0.8*0.5))

        elif action == 'SHOOT':
            if arrow==0:
                pfsb.append((pos,mat,arrow,'R',health,1*0.5))
                if pos=='W':
                    pfsb.append((pos,mat,arrow,'D',health,1*0.5))
                else:
                    pfsb.append((pos,mat,0,'D',min(health+25,100),1*0.5))
            else:
                if pos=='W':
                    pfsb.append((pos,mat,arrow-1,'R',max(health-ARROWHITDAMAGE,0),0.25*0.5))
                    pfsb.append((pos,mat,arrow-1,'R',health,0.75*0.5))
                    pfsb.append((pos,mat,arrow-1,'D',max(health-ARROWHITDAMAGE,0),0.25*0.5))
                    pfsb.append((pos,mat,arrow-1,'D',health,0.75*0.5))
                elif pos=='C':
                    pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                    pfsb.append((pos,mat,arrow-1,'R',max(health-ARROWHITDAMAGE,0),0.5*0.5))
                    pfsb.append((pos,mat,arrow-1,'R',health,0.5*0.5))
                else:
                    pfsb.append((pos,mat,0,'D',min(health+25,100),0.5))
                    pfsb.append((pos,mat,arrow-1,'R',max(health-ARROWHITDAMAGE,0),0.9*0.5))
                    pfsb.append((pos,mat,arrow-1,'R',health,0.1*0.5))
        
        elif action == 'GATHER':
            if mat<2:
                pfsb.append((pos,mat+1,arrow,'R',health,0.75*0.5))
                pfsb.append((pos,mat,arrow,'R',health,0.25*0.5))
                pfsb.append((pos,mat+1,arrow,'D',health,0.75*0.5))
                pfsb.append((pos,mat,arrow,'D',health,0.25*0.5))
            else:
                pfsb.append((pos,mat,arrow,'R',health,1*0.5))
                pfsb.append((pos,mat,arrow,'D',health,1*0.5))

        elif action == 'CRAFT':
            if (mat>0):
                if arrow==0:
                    pfsb.append((pos,mat-1,1,'R',health,0.5*0.5))
                    pfsb.append((pos,mat-1,2,'R',health,0.35*0.5))
                    pfsb.append((pos,mat-1,3,'R',health,0.15*0.5))
                    pfsb.append((pos,mat-1,1,'D',health,0.5*0.5))
                    pfsb.append((pos,mat-1,2,'D',health,0.35*0.5))
                    pfsb.append((pos,mat-1,3,'D',health,0.15*0.5))
                elif arrow==1:
                    pfsb.append((pos,mat-1,2,'R',health,0.5*0.5))
                    pfsb.append((pos,mat-1,3,'R',health,0.5*0.5))
                    pfsb.append((pos,mat-1,2,'D',health,0.5*0.5))
                    pfsb.append((pos,mat-1,3,'D',health,0.5*0.5))
                elif arrow in [2,3]:
                    pfsb.append((pos,mat-1,3,'D',health,1*0.5))
                    pfsb.append((pos,mat-1,3,'R',health,1*0.5))
            else:
                pfsb.append((pos,mat,arrow,'R',health,1*0.5))
                pfsb.append((pos,mat,arrow,'D',health,1*0.5))
    return pfsb


# In[6]:


def read_file(filename):
    with open(filename,'r') as read_file:
        data=json.load(read_file)
    return data

def write_file(filename,data):
    with open(filename,'w') as write_file:
        json.dump(data, write_file, indent = 4)
    return read_file(filename)


# In[7]:


R = []
for s in all_state:
    for a in state_actions[s]:
        expected_reward = 0
        pfsp = get_pfsb(s, a)
        for state_prob in pfsp:
            reward = STEPCOST
            (pos,mat,arrow,state,health,prob) = state_prob
            if((pos==s[0]) and
               (pos in ['E', 'C']) and
              (mat==s[1]) and
              (arrow==0) and
              (state=='D' and s[3]=='R') and
              ((health==s[4]+25) or ((health==100)and(s[4]==100)))
             ):
                reward -= NEGATEREWARD
            expected_reward += (reward*prob)
        R.append(expected_reward)
        
R = np.array(R, dtype=float).reshape(1,len(R))


# In[8]:


A = []

r_state_map = {}
for i in range(len(all_state)):
    r_state_map[all_state[i]]=i

for s1 in all_state:
    row = []
    for s2 in all_state:
        c_cluster = []
        for a in state_actions[s2]:
            c_cluster.append(0)
        row.append(c_cluster)
    A.append(row)

for s in all_state:
    for ai in range(len(state_actions[s])):
        j = r_state_map[s]
        k = ai
        if s[-1]==0:
            A[j][j][k]=1
            continue
        
        a = state_actions[s][ai]
        pfsp = get_pfsb(s, a)
        for state_prob in pfsp:
            (pos,mat,arrow,state,health,prob) = state_prob
            pfs = (pos,mat,arrow,state,health)
            i = r_state_map[pfs]
            A[j][j][k] += prob
            A[i][j][k] -= prob

LP_A = []
for i in range(len(A)):
    row = []
    for cluster in A[i]:
        for p in cluster:
            row.append(p)
    row = np.array(row)
    LP_A.append(row)
LP_A = np.array(LP_A, dtype=float)


# In[9]:


ALPHA = np.zeros(len(all_state))
ALPHA[r_state_map[('C',2,3,'R',100)]]=1
ALPHA = np.array(ALPHA, dtype=float).reshape(len(all_state),1)


# In[10]:


x = cp.Variable(shape=(len(R[0]), 1), name="x")
constraints = [cp.matmul(LP_A,x) == ALPHA, x>=0]
objective = cp.Maximize(cp.matmul(R,x))
problem = cp.Problem(objective, constraints)
solution = problem.solve()


# In[11]:


optimal_policy = {}
cum_i = 0
for i in range(len(all_state)):
    s = all_state[i]
    max_reward = -1000000
    for j in range(len(state_actions[s])):
        a = state_actions[s][j]
        if max_reward <= R[0][cum_i+j]:
            max_reward = R[0][cum_i+j]
            optimal_policy[s] = a
    cum_i += len(state_actions[s])      

policy = []
for s in optimal_policy:
    (pos,mat,arrow,state,health) = s
    sublist = []
    state = [pos[0],mat,arrow,state,health]
    action = optimal_policy[s]
    sublist.append(state)
    sublist.append(action)
    policy.append(sublist)


# In[12]:


dump_A = LP_A.tolist()
dump_R = R[0][:].tolist()
dump_ALPHA = ALPHA[:,0].tolist()
dump_X = list(x.value.flatten())
dump_objective = solution


# In[13]:


dump_dict =  {
    "a": dump_A,
    "r": dump_R,
    "alpha": dump_ALPHA,
    "x": dump_X,
    "policy": policy,
    "objective": dump_objective
}


# In[14]:


# print(np.array(dump_A))
# print(np.array(dump_R))
# print(np.array(dump_ALPHA))
# print(np.array(policy))
# print(np.array(dump_objective))


# In[15]:


load_data = write_file("outputs/part_3_output.json",dump_dict)


# In[ ]:





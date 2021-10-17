# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:11:00 2021

@author: Anthony Owusu-Mensah
"Evolutionary Strategy Application"
"""


# Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

def Create_Parent(allParents,num_variables):
    # The recombination process is global intermediate
    # This function create a single recombined parent
    # selection of parents is from uniform random distribution
    # from a set of parents passed to the function
    # This function create a single recombined parent
    # from a set of parents passed to the function
    parents_Indices = []
    selected_Parents = np.empty((0,(2*num_variables)))
    for i in range(2*num_variables + 1):
        parents_Indices.append(np.random.randint(len(allParents)))
    #end of iteration
    # Select parants from all parents
    for j in parents_Indices:
        parent = allParents[j,:]
        parent = parent[np.newaxis]
        selected_Parents  = np.vstack((selected_Parents,parent))
    #selected_Parents[2,1] = 2
    
 
    first_selection = selected_Parents[0] # Parent 1
    first_selection = first_selection[np.newaxis]
    data_needed_from_selected_parents = selected_Parents [1:,:]
    second_selection = []
    for k in range(0,len(data_needed_from_selected_parents)): # Start selection from parent 2
         #print(selected_Parents[k][z])
         second_selection.append(data_needed_from_selected_parents[k][k])
    second_selection = np.array(second_selection)
    second_selection = second_selection[np.newaxis]    
    # compute mean
    Returning_Parent_After_crossover = 0.5*( first_selection + second_selection)
    return Returning_Parent_After_crossover,selected_Parents,second_selection,parents_Indices,\
        data_needed_from_selected_parents

def obj_fun(x,y,v_half,slope_1,slope_2,Vm,tau_exp):
    

    #am = 0.32*(Vm + 47.13)/ (1 - np.exp(-0.1*(Vm + 47.13))) # Luo&Rudy
    am = x*(Vm + v_half)/ (1 - np.exp(-slope_1*(Vm + v_half)))

    #bm = 0.08*np.exp(-Vm/11) #Luo&Rudy
    bm = y*np.exp(-Vm/slope_2)
    
    tau_data = 1/(am + bm)
    obj = np.mean((np.power((tau_data - tau_exp) ,2)))
    #obj = np.sqrt(obj) 
    return  obj


## Data Preparation
data = pd.read_csv("C:/Users/Tony/Downloads/INa_Kernik.csv")
Vm = np.array(data.loc[:,"V_m"])
minf = np.array(data.loc[:,"m_inf"])
tau_m = np.array(data.loc[:,"tau_m"])

x = 0.32 # x_value
x_Sigma = 0.4453702128478512

y = 0.08 # y_value
y_Sigma = 0.4453702128478512

v_half = 47.13
v_Sigma = 0.4453702128478512

slope_1 = 0.1
s1_Sigma = 0.4453702128478512

slope_2 = 11
s2_Sigma = 0.4453702128478512

num_variables = 5

alpha = 0.8

# The goal is to find values for
u_x = 0.4# Increase value 5_times
l_x = -0.4 # decrease value 25%

###
u_y = 0.008# Increase value 5_times
l_y = -0.008# decrease value 25%


u_v = v_half * 2# Increase value 5_times
l_v = -u_v # decrease value 25%

###
u_s1 = slope_1 * 5# Increase value 5_times
l_s1 = -u_s1# decrease value 25%

u_s2 = slope_2 * 5# Increase value 5_times
l_s2 = -u_s2# decrease value 25%

# Means squared error
obj_func =  obj_fun(x,y,v_half,slope_1,slope_2,Vm,tau_m)

print("x_initial = " ,x)
print()
print("y_initial = " ,y)
print()
print("v_half_initial = " ,v_half)
print()
print("s1_initial = " ,slope_1)
print()
print("s2_initial = " ,slope_2)
print()
print("initial_cost=", obj_func)
print()

#%%
num_Parents = 120
num_Children = num_Parents*6
stoppingCriteria = 30 # Determines the generation
keep_best_Parent_from_every_gen = np.empty((0,11))
keep_best_Child_from_every_gen = np.empty((0,11))
init_Population = np.empty((0,11))

# Create initial parents to start
for i in range(num_Parents):
    x_init = np.random.uniform(l_x,u_x)
    y_init = np.random.uniform(l_y,u_y)
    v_init = np.random.uniform(l_v,u_v)
    s1_init = np.random.uniform(l_s1,u_s1)
    s2_init = np.random.uniform(l_s2,u_s2)
    obj_func_Init = obj_fun(x_init,y_init,v_init,s1_init,s2_init,Vm,tau_m)
    popData = (obj_func_Init,x_init,x_Sigma,y_init,y_Sigma,v_init,v_Sigma,s1_init,s1_Sigma,
               s2_init,s2_Sigma)
    init_Population = np.vstack((init_Population,popData))
#%%    
    
#

gen = 1
for j in range(stoppingCriteria ):
    new_Population = np.empty((0,11)) ## clear for every generation
    mutated_children = np.empty((0,11))
    parents_after_each_gen = np.empty((0,11)) # save parent after every crossover
    #stack_gen = np.empty((0,5))
    #Parent_data = np.empty((0,5))  
    oneFifth_final = 0
    
   ##Create the recombined_parent
    for t in init_Population: 
        oneFifth = 0
        Parent_cross_over = Create_Parent(init_Population[:,1:],num_variables)[0]
        x_parent = Parent_cross_over[0,0]
        y_parent = Parent_cross_over[0,2]
        v_parent = Parent_cross_over[0,4]
        s1_parent = Parent_cross_over[0,6]
        s2_parent = Parent_cross_over[0,8]
        Parent_obj_fun = obj_fun(Parent_cross_over[0,0],Parent_cross_over[0,2],
                                 Parent_cross_over[0,4],Parent_cross_over[0,6],Parent_cross_over[0,8],Vm,tau_m )
        Parent_data = np.column_stack((Parent_obj_fun,Parent_cross_over))
        parents_after_each_gen  = np.vstack((parents_after_each_gen, Parent_data))      
        ## Mutate children from crossed over Parent
        for k in range(int(num_Children/num_Parents)):
            # 2 parameter
            x_sig_new = x_Sigma * np.random.uniform(0,1)
            y_sig_new = y_Sigma * np.random.uniform(0,1)
            v_sig_new = v_Sigma * np.random.uniform(0,1)
            s1_sig_new = s1_Sigma * np.random.uniform(0,1)
            s2_sig_new = s2_Sigma * np.random.uniform(0,1)
            x_new = x_parent + x_sig_new 
            y_new = y_parent + y_sig_new
            v_new = v_parent + v_sig_new 
            s1_new = s1_parent + s1_sig_new
            s2_new = s2_parent + s2_sig_new
            obj_func_child = obj_fun(x_new,y_new,v_new,s1_new,s2_new,Vm,tau_m)
            childData = np.array([obj_func_child, x_new ,x_sig_new,y_new,y_sig_new,
                                  v_new,v_sig_new,s1_new,s1_sig_new,s2_new,s2_sig_new])
            childData =  childData[np.newaxis]
            mutated_children = np.vstack((mutated_children,childData))
            ## Check the number of children better than parents
            if obj_func_child < Parent_obj_fun:
                oneFifth += 1
            else:
                 oneFifth += 0
        oneFifth_final += oneFifth ## check the number of chidren that are better
           
      ## get best parent and child from each generation
    ## best parent from each generation
    obj_parent_data =  parents_after_each_gen[:,0]
    idx_min_parent = np.where(obj_parent_data==min(obj_parent_data))
    best_parent_gen = parents_after_each_gen[idx_min_parent]
    keep_best_Parent_from_every_gen = np.vstack((keep_best_Parent_from_every_gen,best_parent_gen))
    
    ## best child from each generation
    obj_child_data =  mutated_children[:,0]
    idx_min_child = np.where(obj_child_data ==min(obj_child_data))
    best_child_gen = mutated_children[idx_min_child]
    keep_best_Child_from_every_gen = np.vstack((keep_best_Child_from_every_gen,best_child_gen))  
    
    
    # 1/5th rule
    
    success_Ratio = oneFifth_final/ num_Children
    ## Implement after every 5 generations
    if (gen % 5 == 0) or (gen % 5 == 5):
        if success_Ratio > 1/5:
            x_Sigma = x_Sigma/alpha #increase
            y_Sigma = y_Sigma/alpha
        elif success_Ratio < 1/5:
            x_Sigma = x_Sigma*alpha # decreas
            y_Sigma = y_Sigma*alpha
        elif success_Ratio == 1/5:
            x_Sigma = x_Sigma # decreas
            y_Sigma = y_Sigma
            
    else:
        x_Sigma = x_Sigma # decreas
        y_Sigma = y_Sigma
    
    
 
            
   ## Selection of children for next generation
    sort_parents =  np.array(sorted(parents_after_each_gen,key = lambda w:w[0]))
    sort_children =  np.array(sorted(mutated_children,key = lambda w:w[0]))
    
    # selection of 10% of parents & 90%
    next_gen_parents = sort_parents[:int(num_Parents*0.1)]
    next_gen_children = sort_children[:int(num_Children*0.9)]
    stack_gen = np.vstack((next_gen_parents,next_gen_children))
    
    ## shuffle to mix parents and children
    np.random.shuffle(stack_gen)
    
    new_Population =  stack_gen
    gen += 1
    
    init_Population = new_Population     
        
#%%            
    

## best of the best
Total_best = np.vstack((keep_best_Parent_from_every_gen, keep_best_Child_from_every_gen))
obj_total = Total_best[:,0]
idx_total_best = np.where(obj_total == min(obj_total))  
best_in_parent_child = Total_best[idx_total_best]

x_best = best_in_parent_child[0,1]
y_best = best_in_parent_child[0,3]
v_best = best_in_parent_child[0,5]
s1_best = best_in_parent_child[0,7]
s2_best = best_in_parent_child[0,9]
cost_final = best_in_parent_child[0,0]



print("x_best = " ,x_best)
print()
print("y_best = " ,y_best)
print()
print("v_best = " ,v_best)
print()
print("s1_best = " ,s1_best)
print()
print("s2_best = " ,s2_best)
print()
print("cost_final = " ,cost_final)



## Best solution ________________________________________________________
am_ = x_best*(Vm + v_best)/ (1 - np.exp(-s1_best*(Vm + v_best)))
bm_ = y_best*np.exp(-Vm/s2_best)
tau_data_ = 1/(am_ + bm_)

## Original Lou & Rudy Formulation____________________________________________________
am = 0.32*(Vm + 47.13)/ (1 - np.exp(-0.1*(Vm + 47.13)))
bm = 0.08*np.exp(-Vm/11)
tau_data_L = 1/(am + bm)



rsquare = r2_score(tau_m,tau_data_)
print()
print("r_square = ", rsquare)

plt.plot(Vm,tau_m,'-o',label='experiment')
plt.plot(Vm,tau_data_,'-*',label='model')
plt.plot(Vm,tau_data_L,'-+',label='Lou & Rudy')
plt.xlabel('V')
plt.ylabel('tau_m')
plt.legend()

#%%






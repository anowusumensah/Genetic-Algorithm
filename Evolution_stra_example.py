# -*- coding: utf-8 -*-
"""
Created on Thu May 20 00:54:48 2021

@author: Tony
"""
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import math as mt


def Create_Parent(allParents,num_variables):
     
    # The recombination process is global intermediate 
    # This function create a single recombined parent
    # selection of parents is from uniform random distribution
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



def obj_fun(x1,x2):
    #obj = ((x2-((5.1/(4*mt.pi**2))*x1**2) + ((5/mt.pi)*x1) - 6)**2 + \
           #10*(1-(1/(8*mt.pi))) * mt.cos(x1) + 10)
    obj = ((x2 - (x1**2))**2) + ((1 - x1)**2) #Rosenbrock's Function
    #minimum value of the Rosenbrock's Function occurs at x1 = 1, x2 = 1
      
    return obj

x1 = -1.2 # x_value
x_Sigma = 1.25
x2 = 8 # y_value
y_Sigma = 1
alpha = 0.82

# The goal is to find values for
u_x = 20# Increase value 5_times
l_x = -20 # decrease value 25%

###
u_y = 20# Increase value 5_times
l_y = -20# decrease value 25%

init_obj = obj_fun(x1,x2)
print("initial_obj",init_obj)

num_Parents = 120
num_Children = num_Parents*6
stoppingCriteria = 30 # Determines the generation
keep_best_Parent_from_every_gen = np.empty((0,5))
keep_best_Child_from_every_gen = np.empty((0,5))
init_Population = np.empty((0,5))


# Create initial parents to start
for i in range(num_Parents):
    x_init = np.random.uniform(l_x,u_x)
    y_init = np.random.uniform(l_y,u_y)
    obj_func_Init = obj_fun(x_init,y_init)
    popData = (obj_func_Init,x_init,x_Sigma,y_init,y_Sigma)
    init_Population = np.vstack((init_Population,popData))
    
    
#
  
gen = 0  
for j in range(stoppingCriteria ):
    new_Population = np.empty((0,5)) ## clear for every generation
    mutated_children = np.empty((0,5))
    parents_after_each_gen = np.empty((0,5)) # save parent after every crossover
    #stack_gen = np.empty((0,5))
    #Parent_data = np.empty((0,5))  
    oneFifth_final = 0
    
   ##Create the recombined_parent
    for t in init_Population: 
        oneFifth = 0
        Parent_cross_over = Create_Parent(init_Population[:,1:],2)[0]
        x_parent = Parent_cross_over[0,0]
        y_parent = Parent_cross_over[0,2]
        Parent_obj_fun = obj_fun(Parent_cross_over[0,0],Parent_cross_over[0,2])
        Parent_data = np.column_stack((Parent_obj_fun,Parent_cross_over))
        parents_after_each_gen  = np.vstack((parents_after_each_gen, Parent_data))      
        ## Mutate children from crossed over Parent
        for k in range(int(num_Children/num_Parents)):
            # 2 parameter
            x_sig_new = x_Sigma * np.random.uniform(0,1)
            y_sig_new = y_Sigma * np.random.uniform(0,1)
            x_new = x_parent + x_sig_new 
            y_new = y_parent + y_sig_new
            obj_func_child = obj_fun(x_new,y_new)
            childData = np.array([obj_func_child, x_new ,x_sig_new,y_new,y_sig_new])
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
    #if (gen % 1 == 0) or (gen % 1 == 5):
    if (gen % 1 == 0): #do it for every generation
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
    
    # selection of 10% of parents & 90% children
    next_gen_parents = sort_parents[:int(num_Parents*0.1)]
    next_gen_children = sort_children[:int(num_Children*0.9)]
    stack_gen = np.vstack((next_gen_parents,next_gen_children))
    
    ## shuffle to mix parents and children
    np.random.shuffle(stack_gen)
    
    new_Population =  stack_gen
    gen += 1
    
    init_Population = new_Population     
        
            
    

## best of the best
Total_best = np.vstack((keep_best_Parent_from_every_gen, keep_best_Child_from_every_gen))
obj_total = Total_best[:,0]
idx_total_best = np.where(obj_total == min(obj_total))  
best_in_parent_child = Total_best[idx_total_best]

x_best = best_in_parent_child[0,1]
y_best = best_in_parent_child[0,3]
cost_final = best_in_parent_child[0,0]
#r_squared = r2_score()

print("x_best = " ,x_best)
print()
print("y_best = " ,y_best)
print()
print("cost_final = " ,cost_final)























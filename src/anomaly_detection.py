"""
Created on Thu Jun 29 08:42:08 2017

@author: asiu
"""

import json
import os.path
import numpy as np
import pandas as pd

main_directory = 'C:/Users/andre/Documents/anomaly_detection'
batch_filepath = os.path.join(main_directory, 'log_input/batch_log.json')
stream_filepath = os.path.join(main_directory, 'log_input/stream_log.json')


# Collect a set of all user ids that are part of the social network
# Create a list of first-degree friendships as edges in a network graph   
# A network graph can be represented by a set of vertices (use ids) 
# and a set of edges (friendships)
# I will then use network graph to compute higher-degree friendships

ids = set() 
edges = []
list_purchases = []

with open(batch_filepath) as f:
    for index, line in enumerate(f):        
        if index == 0:
            param = json.loads(line)
            D = np.int(param['D']) # D: number of degrees in social network
            T = np.int(param['T']) # T: # of purchases in the user's network 

        else:
            e = json.loads(line)
            if e['event_type'] == 'purchase':
                list_purchases.append(e)            
            
            elif e['event_type'] == 'befriend':
                ids.add(e['id1'])
                ids.add(e['id2'])
                edges.append(set([e['id1'], e['id2']])) 
            
            elif e['event_type'] == 'unfriend':
                edges.remove(set([e['id1'], e['id2']]))   

# Create a dataframe of all previous purchases, which are then used for 
# detecting a significantly large purchase later on

df_purchases = pd.DataFrame(list_purchases)
df_purchases['amount'] = df_purchases['amount'].astype(float)
df_purchases['timestamp'] = pd.to_datetime(df_purchases['timestamp'])

# There are many ways to represent a network graph
# I think the most efficient way is to create a dictionary
# that maps each user id to a set of first-degree friends user ids

graph = dict.fromkeys(ids)

for i in graph.keys():
    # For each user id, create a set of first-degree neighbors 
    # out of all the edges that contain this user id
    neighbors = set()
    for edge in edges:
        if i in edge:
            neighbors = neighbors.union(edge)
            neighbors.discard(i)
    graph[i] = neighbors

# Build functions to get a set of all friends within a specified 
# number of degrees of separation

def degree2(graph, i):
    """ Find all neighbors within 2 degrees of separation """
    neighbors = graph[i]
    # Add all the first-degree friends of user i's first-degree friends
    for friend in graph[i]:
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors
                
def degree3(graph, i):
    """ Find all neighbors within 3 degrees of separation """
    neighbors = degree2(graph, i)
    # Add all the first-degree friends of user i's second-degree friends
    for friend in degree2(graph, i) - graph[i]:
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors

def degree4(graph, i):
    """ Find all neighbors within 4 degrees of separation """
    neighbors = degree3(graph, i)
    # Add all the first-degree friends of user i's third-degree friends
    for friend in degree3(graph, i) - degree2(graph, i):
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors

def degree5(graph, i):
    """ Find all neighbors within 5 degrees of separation """
    neighbors = degree4(graph, i)
    # Add all the first-degree friends of user i's fourth-degree friends
    for friend in degree4(graph, i) - degree3(graph, i):
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors

# Call user i's entire social network within D degrees of separation
def network(graph, i, D):
    if D == 1:
        return graph[i]
    elif D == 2:
        return degree2(graph, i)
    elif D == 3:
        return degree3(graph, i)
    elif D == 4:
        return degree4(graph, i)
    elif D == 5:
        return degree5(graph, i)

# As new events stream in from the users, we need to update the existing
# social network graph and the data of all previous purchases 
# The most important task is to check whether a purchase is 
# more than 3 standard deviations above the mean of 
# T tracked purachses made by one's network of friends within 
# D degrees of separation.

# Create and open a new file in write mode to collect flagged purchases
output_filepath = os.path.join(main_directory, 'log_output/flagged_purchases.json')

with open(output_filepath, 'w') as flagged_file:
    
    # Read in the stream of new events one by one
    with open(stream_filepath) as stream_file:
        for event in stream_file:
            e = json.loads(event)
      
            if e['event_type'] == 'purchase':                
                # Select all purchases made by friends within 
                # D degrees of separation
                rows = df_purchases['id'].isin(list(network(graph, e['id'], D)))
                history = df_purchases[rows].sort_values(by='timestamp').amount  
                
                # Update the dataframe of all previous purchases
                e['amount'] = np.float(e['amount'])
                e['timestamp'] = pd.to_datetime(e['timestamp'])
                df_purchases = df_purchases.append(e, ignore_index=True)
                
                # Check if there are at least two previous purchases
                # Compute the mean and standard deviation of T tracked purchases
                if len(history)>=2:
                    mean = round(history.tail(T).mean(), 2)
                    std = round(history.tail(T).std(ddof=0), 2)
                    
                    if np.float(e['amount']) >= mean + 3 * std:
                        e['mean'] = str(mean)
                        e['sd'] = str(std)
                        e['amount'] = str(e['amount'])
                        e['timestamp'] = str(e['timestamp'])          
                        json.dump(e, flagged_file)
                        flagged_file.write('\n')
            
            # Update the existing social network of frienships
            elif e['event_type'] == 'befriend':
                graph[e['id1']] = graph[e['id1']].union(set(e['id2']))
                graph[e['id2']] = graph[e['id2']].union(set(e['id1']))
            elif e['event_type'] == 'unfriend':
                graph[e['id1']] = graph[e['id1']].difference(set(e['id2']))
                graph[e['id2']] = graph[e['id2']].difference(set(e['id1']))
"""
Created on Thu Jun 29 08:42:08 2017

@author: asiu
"""

import json
import os.path
import numpy as np
import pandas as pd

main_directory = 'C:/Users/andre/OneDrive/Documents/GitHub/anomaly_detection'
batch_filepath = os.path.join(main_directory, 'log_input/batch_log.json')
stream_filepath = os.path.join(main_directory, 'log_input/stream_log.json')

# Get the key parameters D, the number of degrees in social network
# and T, the tracked number of purchases in user's network 
events = []
with open(batch_filepath) as f:
    for index, line in enumerate(f):
        if index == 0:
            param = json.loads(line)
            D = np.int(param['D'])
            T = np.int(param['T'])
        else:
            events.append(json.loads(line))

# Collect a set of ids that are part of the social network
# Get a list of first-degree friendships   
# Creat a dataframe of all previous purchases       
ids = set() 
edges = []
list_purchases = []

for e in events:
    if e['event_type'] == 'befriend':
        ids.add(e['id1'])
        ids.add(e['id2'])
        edges.append(set([e['id1'], e['id2']])) 
    elif e['event_type'] == 'unfriend':
        edges.remove(set([e['id1'], e['id2']]))   
    elif e['event_type'] == 'purchase':
        list_purchases.append(e)

#    
df_purchases = pd.DataFrame(list_purchases)
df_purchases['amount'] = df_purchases['amount'].astype(float)

# Create a dictionary mapping each id to a set of first-degree friends
graph = dict.fromkeys(ids)

for i in graph.keys():
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
    for friend in graph[i]:
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors
                
def degree3(graph, i):
    """ Find all neighbors within 3 degrees of separation """
    neighbors = degree2(graph, i)
    for friend in degree2(graph, i) - graph[i]:
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors

def degree4(graph, i):
    """ Find all neighbors within 4 degrees of separation """
    neighbors = degree3(graph, i)
    for friend in degree3(graph, i) - degree2(graph, i):
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors

def degree5(graph, i):
    """ Find all neighbors within 5 degrees of separation """
    neighbors = degree4(graph, i)
    for friend in degree4(graph, i) - degree3(graph, i):
        neighbors = neighbors.union(graph[friend])
        neighbors.discard(i)
    return neighbors

def network(graph, i, D):
    if D==1:
        return graph[i]
    elif D==2:
        return degree2(graph, i)
    elif D==3:
        return degree3(graph, i)
    elif D==4:
        return degree4(graph, i)
    elif D==5:
        return degree5(graph, i)
    
output_filepath = os.path.join(main_directory, 'log_output/flagged_purchases.json')

# Create and open a new file in write mode to collect flagged purchases   
with open(output_filepath, 'w') as flagged_file:
    
    # Open the stream of new events one by one
    with open(stream_filepath) as stream_file:
        for event in stream_file:
            e = json.loads(event)
            
            # Update the existing social network
            if e['event_type'] == 'befriend':
                graph[e['id1']] = graph[e['id1']].union(set(e['id2']))
                graph[e['id2']] = graph[e['id2']].union(set(e['id1']))
            elif e['event_type'] == 'unfriend':
                graph[e['id1']] = graph[e['id1']].difference(set(e['id2']))
                graph[e['id2']] = graph[e['id2']].difference(set(e['id1']))
            
            # Flag up unusually large purchases compared to one's social network
            elif e['event_type'] == 'purchase':
                
                # Filter all purchases made in the user's network
                history = df_purchases[df_purchases['id'].isin(
                        list(network(graph, e['id'], D)))].amount  
                
                # Update the dataframe of all previous purchases
                df_purchases = df_purchases.append(e, ignore_index=True)
                df_purchases['amount'] = df_purchases['amount'].astype(float)
                
                # Check is the user's purchase is more than 3 sd from the mean
                if len(history)>=2:
                    mean = round(history.tail(T).mean(), 2)
                    std = round(history.tail(T).std(ddof=0), 2)
                    
                    if np.float(e['amount']) >= mean + 3* std:
                        e['mean'] = str(mean)
                        e['sd'] = str(std)
                        json.dump(e, flagged_file)
                        flagged_file.write('\n')

# -*- coding: utf-8 -*-
"""
    Author: Andrew Siu (andrewjsiu@gmail.com)
    
    -------------------------------------------------
    Detecting Large Purchases within a Social Network
    -------------------------------------------------
    
    Build functions to get a set of all friends within a specified 
    number of degrees of separation.

"""

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
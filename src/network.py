# -*- coding: utf-8 -*-
"""
    Author: Andrew Siu (andrewjsiu@gmail.com)
    
    -------------------------------------------------
    Detecting Large Purchases within a Social Network
    -------------------------------------------------
    
    Build functions to get a set of all friends within a specified 
    number of degrees of separation.

"""
# Call user i's entire social network within D degrees of separation
def network(graph, user, D):
    neighbors = graph[user]
    if D == 1:
        return neighbors
    else:
        temp = set()
        for degree in range(D-1):
            # Add all friends in the next layer
            diff = neighbors - temp
            temp = neighbors
            for friend in diff:
                neighbors = neighbors.union(graph[friend])
                neighbors.discard(user)
        return neighbors
                
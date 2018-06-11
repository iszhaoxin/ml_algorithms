import numpy as np

def Decide(G, s, t):
    table = np.zeros(len(G.nodes), len(G.node))
    table[0][s.index] = 1
    for i in range(len(1, G.nodes)):
        for node_index in range(len(G.node)):
            table[i][node_index] = np.dot(G.edges[:][node_index],table[i-1])

def returnPath(G, s, t):
    if !decide(G, s, t):
        return None
    path = []
    allnodes = G.nodes
    path.append(allnodes.pop(t))
    while allnodes != {s}:
        for node in connectWith(s):
            if decide(G, node, t):
                s = node
                path.append(allnodes.pop(node))
    path.append(s)
    return path.reversed()

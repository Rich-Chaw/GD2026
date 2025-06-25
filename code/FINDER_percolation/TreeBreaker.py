import sys
import heapq
import networkx as nx
import numpy as np

## quick and dirty graph implementation: removed nodes are just flagged

class Graph:
    def __init__(self):
        self.V = []  # denote the target nodes for the current source node
        self.present = []  # denote whether the current node exists or not, 1:exist; 0:removed
        self.M = 0  # denote the number of edges

    def size(self):
        return sum(self.present)

    def add_node(self, i):
        if i >= len(self.V):
            delta = i + 1 - len(self.V)
            self.present += [ 1 ] * delta
            self.V += [ [] for j in range(delta) ]
    def add_edge(self, i, j):
        self.add_node(i)
        self.add_node(j)
        self.V[i] += [j]
        self.V[j] += [i]
        self.M += 1
    def remove_node(self, i):
        self.present[i] = 0
        self.M -= sum(1 for j in self.V[i] if self.present[j])


def TreeBreak(g, sol_Decycle):
    sol_TreeBreak = []
    G = Graph()
    TotalNum = nx.number_of_nodes(g)
    n = 0
    for edge in g.edges():
        G.add_edge(int(edge[0]),int(edge[1]))
    for node in sol_Decycle:
        G.remove_node(int(node))
        n += 1

    # for l in sys.stdin:
    #     v = l.split();
    #     if v[0] == 'D' or v[0] == 'E':
    #         G.add_edge(int(v[1]),int(v[2]))
    #     if v[0] == "V":
    #         G.add_node(int(v[1]))
    #     if v[0] == 'S':
    #         G.remove_node(int(v[1]))
    #         n += 1

    N = G.size()
    S = [0] * len(G.V)

    def size(i, j):
        if not G.present[i]:
            return 0
        if S[i]:
            print ("# the graph is NOT acyclic")
            exit()
        S[i] = 1 + sum(size(k, i) for k in G.V[i] if k != j and G.present[k])
        return S[i]

    H = [(-size(i, None), i) for i in range(len(G.V)) if G.present[i] and not S[i]]

    Ncc = len(H)
    # print ("# N:", N, "Ncc:", Ncc, "M:", G.M)
    assert(N - Ncc == G.M)

    # print ("# the graph is acyclic")

    sys.stdout.flush()

    heapq.heapify(H)
    isTerminal = False
    while len(H):
        s,i = heapq.heappop(H)
        scomp = -s
        sender = None
        while True:
            sizes = [(S[k],k) for k in G.V[i] if k != sender and G.present[k]]
            if len(sizes) == 0:
                break
            M, largest = max(sizes)
            if M <= scomp/2:
                for k in G.V[i]:
                    if S[k] > 1 and G.present[k]:
                        heapq.heappush(H, (-S[k], k))
                sol_TreeBreak.append(i)
                G.remove_node(i)
                n+=1
                # print ("S", i, n, scomp)
                if scomp < np.max([int(0.01 * TotalNum), 2]):
                # if scomp <= 2:
                    isTerminal = True
                    # return sol_TreeBreak
                    break
                sys.stdout.flush()
                break
            S[i] = 1 + sum(S[k] for k in G.V[i] if k != largest and G.present[k])
            sender, i = i, largest
        if isTerminal:
            break
    return sol_TreeBreak

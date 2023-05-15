import networkx as nx
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

def get_Data():
    Edges = []
    with open("attachment3.csv") as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            if(len(line) != 1):
                for idx in range(1, len(line)):
                    Edges.append((int(line[0]), int(line[idx])))
    return Edges
    
def Quick_Tuopu(Edges: list):
    nodenum = 607
    m = np.matrix(np.zeros((nodenum, nodenum)), dtype = bool)
    for e in Edges:
        m[e[0], e[1]] = 1
    I = np.matrix(np.identity(len(m)), dtype = bool)
    m = m + I
    res = m
    while(nodenum - 1):
        if(nodenum & 1): res = res * m
        m = m * m
        nodenum  = nodenum >> 1
    print(res[0,:])
    np.savetxt(fname="KeDaMatrix.csv", X=np.array(res), fmt="%d",delimiter=",")
    
    

if __name__ == "__main__":
    e = get_Data()
    Quick_Tuopu(e)
    # a = np.matrix([[1,1,1,0,0],[0,1,0,0,0],[0,0,1,1,1],[0,0,0,1,0],[0,0,0,0,1]], dtype=bool)
    # a = a * a
    # print(a.shape)
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         if(a[i,j]>1): a[i,j] = 1
    # print(a)
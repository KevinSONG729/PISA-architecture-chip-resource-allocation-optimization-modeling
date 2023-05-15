from lib2to3.pytree import Node
import networkx as nx
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import queue

from sympy import DiagMatrix, public

# 全局数据（边，存取变量，需求资源）
Edges = []
R_W = []
Source = []

Source_Upper = [1, 2, 56, 64]
Source_UpperZhedie = [1, 3, 56, 64]

flow_Source = [[0,0,0,0] for i in range(0, 607)]
Node_allocated = [0 for i in range(0, 607)]
Node_whichFlow = [-1 for i in range(0, 607)]
Node_LowLayer = [0 for i in range(0, 607)]
flow_max = 0
num_TCAM_onEven = 0

class PriorityNode(object):
    def __init__(self, Lowbound, node, nodeSource):
        self.lowbound = Lowbound
        self.node = node
        self.nodeSource = nodeSource
    def __lt__(self, other):
        if(self.lowbound > other.lowbound):
            return 1
        elif(self.lowbound == other.lowbound):
            low = self.lowbound
            cha = [i-j for i,j in zip(Source_Upper, flow_Source[low])]
            return [i-j for i,j in zip(self.nodeSource, cha)] > [i-j for i,j in zip(other.nodeSource, cha)]
    def getNode(self):
        return self.node
    
# 结点数据预处理
def getGraphMessage():
    with open("attachment3.csv") as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            if(len(line) != 1):
                for idx in range(1, len(line)):
                    Edges.append((int(line[0]), int(line[idx])))
    # print(Edges)
    with open("attachment2.csv") as csv_file:
        lines = csv.reader(csv_file)
        for idx, line in enumerate(lines):
            if(idx%2==0):
                if(len(line)<=2):
                    R_W.append({'name': int(line[0]), 'W': []})
                else:
                    R_W.append({'name': int(line[0]), 'W': line[2:len(line)+1]})
            else:
                if(len(line)<=2):
                    R_W[-1]['R'] = []
                else:
                    R_W[-1]['R'] = line[2:len(line)+1]
        # print(R_W)
    with open("attachment1.csv") as csv_file:
        lines = csv.reader(csv_file)
        for idx, line in enumerate(lines):
            if(idx==0): continue
            else:
                Source.append([int(l) for l in line[1:len(line)+1]])
        # print(Source)
    return Edges, R_W, Source

# 有向无环图生成
def GraphGenerator():
    DiGraph = nx.DiGraph()
    for idx in range(0, 607):
        DiGraph.add_node(idx)
    for e in Edges:
        DiGraph.add_edge(int(e[0]), int(e[1]))
    # print(DiGraph)
    # draw_Graph(DiGraph)
    return DiGraph

# 问题1求解函数
def Solution1(DiGraph: nx.DiGraph):
    global flow_max
    generations = [g for g in nx.topological_generations(DiGraph)]
    # print(generations)
    # GenerateYilaiMatrix(DiGraph, generations)
    ComparePutIn(generations[0][0], 0, 1, 0) # 先将第一个基本块放入流水线
    for i, g in enumerate(generations[1:]): # 考虑其他基本块
        # print(g)
        for j, node in enumerate(g):
            Node_LowLayer[node] = findNodeLowBound(node, DiGraph, generations)
            # print(Node_LowLayer[node])
        q = queue.PriorityQueue()
        for j, node in enumerate(g):
            q.put(PriorityNode(Node_LowLayer[node], node, Source[node]))
        # print(q.queue) 
        while(q.empty() != True):
            node = q.get()
            result = ComparePutIn(node.getNode(), Node_LowLayer[node.getNode()], 1, 1)
            # print(result)
            for nodeInQueue in q.queue:
                if isDataDependent(node.getNode(), nodeInQueue.getNode(), generations)==3 and \
                Node_LowLayer[nodeInQueue.getNode()]<=Node_LowLayer[node.getNode()]:
                    Node_LowLayer[nodeInQueue.getNode()] = Node_LowLayer[node.getNode()] + 1
                    if(flow_max < Node_LowLayer[nodeInQueue.getNode()]):
                        flow_max = Node_LowLayer[nodeInQueue.getNode()]
                elif isDataDependent(node.getNode(), nodeInQueue.getNode(), generations)==4 and \
                Node_LowLayer[nodeInQueue.getNode()]<=Node_LowLayer[node.getNode()]:
                    Node_LowLayer[nodeInQueue.getNode()] = Node_LowLayer[node.getNode()]
                    if(flow_max < Node_LowLayer[nodeInQueue.getNode()]):
                        flow_max = Node_LowLayer[nodeInQueue.getNode()]

def Solution2(DiGraph: nx.DiGraph):
    global flow_max
    generations = [g for g in nx.topological_generations(DiGraph)]
    # print(generations)
    # GenerateYilaiMatrix(DiGraph, generations)
    ComparePutIn(generations[0][0], 0, 1, 0) # 先将第一个基本块放入流水线
    for i, g in enumerate(generations[1:]): # 考虑其他基本块
        # print(g)
        for j, node in enumerate(g):
            Node_LowLayer[node] = findNodeLowBound(node, DiGraph, generations)
            # print(Node_LowLayer[node])
        q = queue.PriorityQueue()
        for j, node in enumerate(g):
            q.put(PriorityNode(Node_LowLayer[node], node, Source[node]))
        # print(q.queue) 
        while(q.empty() != True):
            node = q.get()
            result = ComparePutIn2(DiGraph, generations, node.getNode(), Node_LowLayer[node.getNode()], 1, 1)
            # print(result)
            for nodeInQueue in q.queue:
                if isDataDependent(node.getNode(), nodeInQueue.getNode(), generations)==3 and \
                Node_LowLayer[nodeInQueue.getNode()]<=Node_LowLayer[node.getNode()]:
                    Node_LowLayer[nodeInQueue.getNode()] = Node_LowLayer[node.getNode()] + 1
                    if(flow_max < Node_LowLayer[nodeInQueue.getNode()]):
                        flow_max = Node_LowLayer[nodeInQueue.getNode()]
                elif isDataDependent(node.getNode(), nodeInQueue.getNode(), generations)==4 and \
                Node_LowLayer[nodeInQueue.getNode()]<=Node_LowLayer[node.getNode()]:
                    Node_LowLayer[nodeInQueue.getNode()] = Node_LowLayer[node.getNode()]
                    if(flow_max < Node_LowLayer[nodeInQueue.getNode()]):
                        flow_max = Node_LowLayer[nodeInQueue.getNode()]

# 找到Node基本块的所有祖宗结点，返回祖宗结点的列表 
def findAllPredecessors(node: int, bfs_reversed_list: list):
    ans = []
    q = queue.Queue()
    q.put(node)
    while(q.empty()==0):
       res = q.get()
       for n in bfs_reversed_list:
           if(n[0]==res):
               q.put(n[1])
               ans.append(n[1])
    # print(ans)
    return ans

def findAllNode(node: int, generations:list):
    res = []
    for g in generations:
        # print(g)
        if len(g) == 0:
            continue
        elif node not in g:
            res = res + g
    return res

# 绘制有向无环图
def draw_Graph(DiGraph: nx.DiGraph):
    # nx.draw_kamada_kawai(DiGraph, node_size=2)
    # pos = nx.spring_layout(DiGraph)
    colors = range(969)
    options = {
        # "node_color": "#A0CBE2",
        "node_color": range(607),
        "edge_color": colors,
        "width": 0.4,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
        "font_size": 5,
    }
    nx.draw_kamada_kawai(DiGraph, **options, node_size=6, arrows=True)
    plt.savefig('C:\\Users\\Desktop\\研赛\\D\\figure4.png',dpi=600, bbox_inches='tight')
    plt.show()

# 判断两个基本块之间是否有数据依赖（k1->k2 k1要在k2之前运行）
# 数据依赖分为静态数据依赖和动态运行时数据依赖，静态数据依赖是在拓扑排序时就已经给出，
# 而动态运行时数据依赖是要到最后放入流水线那一步才能知道确定的依赖关系
# 返回值：0：没有依赖关系 1：有强依赖关系 2：有弱依赖关系 3：动态强依赖关系 4：动态弱依赖关系
def isDataDependent(k1: int, k2: int, generations: list):
    if(k1 == k2): return 0
    k1_g = -1
    k2_g = -1
    for idx, g in enumerate(generations):
        if(k1 in g):
            k1_g = idx
        if(k2 in g):
            k2_g = idx
        if(k1_g!=-1 and k2_g!=-1): break
    if(k1_g > k2_g): return 0 # k1的拓扑级别比k2高，不可能存在依赖关系
    else:
        if(k1_g < k2_g): # k1的拓扑级别比k2低，查找有无写后读，写后写，读后写数据依赖
            return isDataDependentCore(k1, k2, 0)
        if(k1_g == k2_g): # k1拓扑级别与k2相等，查找潜在的数据依赖关系
            return isDataDependentCore(k1, k2, 1)
            
# 判断数据依赖核心代码（有无写后读，写后写，读后写数据依赖关系） k1->k2
# 返回值 0: 没有依赖关系 1: 有强依赖关系 2：有弱依赖关系            
def isDataDependentCore(k1: int, k2: int, isequal: bool):   
    k1_R = []
    k1_W = []
    k1_complete = False
    k2_R = []
    k2_W = []
    k2_complete = False
    for idx, item in enumerate(R_W):
        if(int(item['name'])==k1):
            k1_R = item['R']
            k1_W = item['W']
            k1_complete = True
        if(int(item['name'])==k2):
            k2_R = item['R']
            k2_W = item['W']
            k2_complete = True
        if(k1_complete and k2_complete): break
    # 写后读
    if(len(list(set(k1_W) & set(k2_R)))!=0): return 3 if isequal else 1
    # 写后写
    if(len(list(set(k1_W) & set(k2_W)))!=0): return 3 if isequal else 1
    # 读后写
    if(len(list(set(k1_R) & set(k2_W)))!=0): return 4 if isequal else 2
    # 没有依赖关系
    return 0

# 平均分流算法判断数据之间是否存在控制依赖
# 返回值 0: 没有控制依赖 1：有控制依赖
def isControlDenpendent(k1: int, k2: int, DiGraph: nx.DiGraph, generations: list):
    if(k1==k2): return 0
    flow_list = [0 for i in range(0, 607)]
    flow_list[generations[0][0]] = 1
    edge_list = [e for e in nx.bfs_edges(DiGraph, generations[0][0])]
    toList = [edge_list[0][1]]
    toNum = 1
    Last = edge_list[0][0]
    for edge in edge_list[1:]:
        if(edge[0]!=Last):
            for toNode in toList:
                flow_list[toNode] += (flow_list[Last] / toNum)
            toList = []
            toNum = 1
            Last = edge[0]
            toList.append(edge[1])
        else:
            toList.append(edge[1])
            toNum += 1
    for toNode in toList:
        flow_list[toNode] += (flow_list[Last] / toNum)   
    if(abs(k1 - k2)<1e-9): return 1
    else: return 0

def GenerateYilaiMatrix(DiGraph: nx.DiGraph, generations: list):
    ans  = []
    num = 0
    for i in range(0,607):
        res = []
        for j in range(0,607):
            k1 = isDataDependent(i, j, generations)
            k2 = isControlDenpendent(i, j, DiGraph, generations)
            if(k1==1):
                res.append(1)
                num = num + 1
            elif(k2==1 or k1==2):
                res.append(2)
                num = num + 1
            else:
                res.append(0)  
        ans.append(res)
    num = num / (606*606)
    print(num, np.array(ans))
    np.savetxt(fname="YilaiMatrix.csv", X=np.array(ans), fmt="%d",delimiter=",")   

# 判断是否能将第Node个基本块放入第k级流水线
# 能就放入流水线，更新流水线状态和点分配状态，返回1
# 不能就不执行任何操作，返回0
def ComparePutIn(Node: int, k: int, isOperator: bool, forced: bool):
    global num_TCAM_onEven
    global flow_max
    if(k > flow_max): return 0
    Node_source = Source[Node]
    if(ComparePutInCore(flow_Source[k], Node_source, k)==1): # 经判断可以放入
        if(isOperator):
            # 执行放入操作
            flow_Source[k] = [i + j for i, j in zip(flow_Source[k], Node_source)] 
            Node_allocated[Node] = 1
            Node_whichFlow[Node] = k
            if(k%2==0 and Node_source[0]>0): num_TCAM_onEven += 1
        return 1
    elif(forced):
        while(ComparePutInCore(flow_Source[k], Node_source, k)==0):
            k += 1
            # print(k, Node_source)
            if(flow_max < k):
                flow_max = k
        # 执行放入操作
        flow_Source[k] = [i + j for i, j in zip(flow_Source[k], Node_source)] 
        Node_allocated[Node] = 1
        Node_whichFlow[Node] = k
        if(k%2==0 and Node_source[0]>0): num_TCAM_onEven += 1
        return 1
    else:
        return 0
    
# 判断是否能将第Node个基本块放入第k级流水线 核心代码
# 资源约束 + 折叠资源约束 + TCAM偶数级约束
# 返回值：0：不可以 1：可以 
def ComparePutInCore(flowSource: list, NodeSource:list, k: int):
    global num_TCAM_onEven
    assert len(Source_Upper) == len(flowSource)
    assert len(Source_Upper) == len(NodeSource)
    flag = True
    # if k <16 or k >31:
    #     res = [i + j for i, j in zip(flowSource, NodeSource)]
    #     for idx,_ in enumerate(Source_Upper):
    #         if(Source_Upper[idx]<res[idx]):
    #             flag = False
    #             break
    # else:
    #     res = [i + j + k for i, j, k in zip(flowSource, flow_Source[k-16], NodeSource)]
    #     for idx,_ in enumerate(Source_UpperZhedie):
    #         if(Source_UpperZhedie[idx]<res[idx]):
    #             flag = False
    #             break
    res = [i + j for i, j in zip(flowSource, NodeSource)]
    for idx,_ in enumerate(Source_Upper):
        if(Source_Upper[idx]<res[idx]):
            flag = False
            break
    if(k>=16 and k<=31):
        res = [i + j + k for i, j, k in zip(flowSource, flow_Source[k-16], NodeSource)]
        for idx,_ in enumerate(Source_UpperZhedie):
            if(Source_UpperZhedie[idx]<res[idx]):
                flag = False
                break    
    if(k%2==0 and NodeSource[0]>0 and num_TCAM_onEven>5): flag = False      
    # num = 0
    # for layer in range(0, flow_max+1):
    #     if(layer%2==0 and flow_Source[layer][0]>0):
    #         num += 1
    # if(num>5): flag = False 
        
    return flag

# Solution2 更换了（2）（3）（5）约束
# 判断是否能将第Node个基本块放入第k级流水线
# 能就放入流水线，更新流水线状态和点分配状态，返回1
# 不能就不执行任何操作，返回0
def ComparePutIn2(DiGraph: nx.DiGraph, generations: list, Node: int, k: int, isOperator: bool, forced: bool):
    global num_TCAM_onEven
    global flow_max
    if(k > flow_max): return 0
    Node_source = Source[Node]
    if(ComparePutInCore2(DiGraph, generations, Node, flow_Source[k], Node_source, k)==1): # 经判断可以放入
        if(isOperator):
            # 执行放入操作
            flow_Source[k][0] += Node_source[0]
            flow_Source[k][3] += Node_source[3]
            Node_allocated[Node] = 1
            Node_whichFlow[Node] = k
            if(k%2==0 and Node_source[0]>0): num_TCAM_onEven += 1
        return 1
    elif(forced):
        while(ComparePutInCore2(DiGraph, generations, Node, flow_Source[k], Node_source, k)==0):
            k += 1
            # print(k, Node_source)
            if(flow_max < k):
                flow_max = k
        # 执行放入操作
        flow_Source[k][0] += Node_source[0]
        flow_Source[k][3] += Node_source[3]
        Node_allocated[Node] = 1
        Node_whichFlow[Node] = k
        if(k%2==0 and Node_source[0]>0): num_TCAM_onEven += 1
        return 1
    else:
        return 0

# Solution2 更换了（2）（3）（5）约束    
# 判断是否能将第Node个基本块放入第k级流水线 核心代码
# 资源约束 + 折叠资源约束 + TCAM偶数级约束
# 返回值：0：不可以 1：可以 
def ComparePutInCore2(DiGraph: nx.DiGraph, generations: list, node: int, flowSource: list, NodeSource:list, k: int):
    global num_TCAM_onEven
    copy = NodeSource.copy()
    assert len(Source_Upper) == len(flowSource)
    assert len(Source_Upper) == len(NodeSource)
    flag = True
    
    same_path = []
    for i, f in enumerate(Node_whichFlow):
        if(f==k and KeDa(i, node, DiGraph, generations)): same_path.append(i)
        
    for n in same_path:
        copy[1] += Source[n][1]
        copy[2] += Source[n][2]
    res = [i + j for i, j in zip(flowSource, copy)]
    for idx,_ in enumerate(Source_Upper):
        if(Source_Upper[idx]<res[idx]):
            flag = False
            break
    if(flag == True and k>=16 and k<=31):
        for i, f in enumerate(Node_whichFlow):
            if(f==k-16 and KeDa(i, node, DiGraph, generations)):
                copy[1] += Source[i][1]
                copy[2] += Source[i][2]
        res = [i + j + k for i, j, k in zip(flowSource, flow_Source[k-16], copy)]
        for idx,_ in enumerate(Source_UpperZhedie):
            if(Source_UpperZhedie[idx]<res[idx]):
                flag = False
                break
            
    if(k%2==0 and NodeSource[0]>0 and num_TCAM_onEven>5): flag = False      
    # num = 0
    # for layer in range(0, flow_max+1):
    #     if(layer%2==0 and flow_Source[layer][0]>0):
    #         num += 1
    # if(num>5): flag = False 
        
    return flag

# 考虑静态依赖 闭区间(可取到的)
def findNodeLowBound(node: int, DiGraph: nx.DiGraph, generations: list):
    global flow_max
    res1 = [] # 具有强依赖的
    res2 = [] # 具有弱依赖的
    allProcessors = findAllPredecessors(node, [e for e in nx.bfs_predecessors(DiGraph, generations[0][0])])
    # allProcessors = findAllNode(node, generations)
    for nodePressors in allProcessors:
        if(isDataDependent(nodePressors, node, generations)==1):
            res1.append(nodePressors)
        elif(isDataDependent(nodePressors, node, generations)==2 or isControlDenpendent(nodePressors, node, DiGraph, generations)==1):
            res2.append(nodePressors)
    max1 = 0
    max2 = 0
    for node in res1:
        max1 = max1 if max1 > Node_whichFlow[node] else Node_whichFlow[node]
    for node in res2:
        max2 = max2 if max2 >= Node_whichFlow[node] else Node_whichFlow[node]
    max1_increased = False
    while(ComparePutIn(node, max1, 0, 0)==0 if max1==0 else ComparePutIn(node, max1+1, 0, 0)==0):
        max1 += 1
        max1_increased = True
        if(max1 > flow_max):
            flow_max = max1 + 1
    while(ComparePutIn(node, max2, 0, 0)==0):
        max2 += 1
        if(max2 > flow_max):
            flow_max = max2
    max1 = max1 + 1 if max1_increased == 0 and max1 != 0 else max1
    return max1 if max1 >= max2 else max2

# 利用逆向BFS_DP构建可达表
def KedaDP(DiGraph: nx.DiGraph, generations: list):
    res = [[i] for i in range(0, 607)]
    edge = [e for e in nx.bfs_predecessors(DiGraph, generations[0][0])]
    edge.reverse()
    for e in edge:
        res[e[1]] = res[e[1]] + res[e[0]]
    # print(res)
    return res

# 判断两点之间是否在同一条路径上（是否可达）
# 返回值 0: 不在 1：在
def KeDa(k1: int, k2: int, DiGraph: nx.DiGraph, generations: list):
    kedaBiao = KedaDP(DiGraph, generations)
    return k2 in kedaBiao[k1] or k1 in kedaBiao[k2]

# 快速幂warshall计算可达矩阵并建立拓扑序
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
    # print(res[0,:])
    # np.savetxt(fname="KeDaMatrix.csv", X=np.array(res), fmt="%d",delimiter=",")
    return res
    
def Quick_Keda(k1: int, k2: int, Edges: list):
    KedaBiao = Quick_Tuopu(Edges)
    return KedaBiao[k1, k2] or KedaBiao[k2, k1]

if __name__ == "__main__":
    getGraphMessage()
    Graph = GraphGenerator()
    # draw_Graph(Graph)
    Solution2(Graph)
    print(flow_max)
    # print(Node_allocated)
    print(Node_whichFlow)
    print(Quick_Keda(1, 365, Edges))
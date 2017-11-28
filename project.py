# -*- coding: utf-8 -*-

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


from pylab import *
import pandas as pd
import numpy as np
import os
import matplotlib
import gc
import networkx as nx


from scipy import stats, integrate

import matplotlib.pyplot as plt




import seaborn as sns
sns.set(color_codes=True)
#sns.set_context('poster')
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

import sklearn.cluster as cluster
import time




#lapd_data = pd.read_csv('/home/sam/Downloads/Arrest_Data_from_2010_to_Present.csv')
nsduh_data = pd.read_csv('/home/sam/Downloads/35509-0001-Data.tsv', sep = '\t')
nsduh_addy = nsduh_data[['COCEVER','LSD', 'PCP', 'ECSTASY', 'ADDERALL', 'MESC', 'PSILCY']]#,'CODEINE','HYDROCOD']]
#lapd_charge = lapd_data[['Charge','Descent Code']]

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

#rac = nsduh_addy.NEWRACE2.values
#ad = nsduh_addy.ADDERALL.values
#plt.plot(rac,ad,"bo")
#plt.
nsduh_addy.replace({91:0, 985:0, 994: 0, 97: 0 ,997: 0, 981 : 0, 998 : 0, 991: 0, 94: 0, 98:0 }, inplace=True)

nsduh_addy = nsduh_addy[(nsduh_addy.COCEVER > 0) & (nsduh_addy.LSD > 0) & (nsduh_addy.PCP > 0) & (nsduh_addy.ECSTASY > 0) & (nsduh_addy.ADDERALL > 0) & (nsduh_addy.MESC > 0) & (nsduh_addy.PSILCY > 0)]# & (nsduh_addy.CODEINE > 0) & (nsduh_addy.HYDROCOD > 0)]
 
nsduh_addy = nsduh_addy.as_matrix()
edges = np.zeros((7,7))

G = nx.Graph()



for x in range(0, 7198) : 
    y = 0
    z = 0
    while y < 7:
            if(nsduh_addy[x][y] == nsduh_addy[x][z]) :
                 edges[y][z] = edges[y][z] + 1
            z = z + 1
            if (z == 7) :
                z = 0
                y = y + 1
                     
                     


G.add_edge('a','b',weight = edges[0][1])
G.add_edge('a','c',weight = edges[0][2])
G.add_edge('a','d',weight = edges[0][3])
G.add_edge('a','e',weight = edges[0][4])
G.add_edge('a','f',weight = edges[0][5])
G.add_edge('a','g',weight = edges[0][6])
G.add_edge('b','c',weight = edges[1][2])
G.add_edge('b','d',weight = edges[1][3])
G.add_edge('b','e',weight = edges[1][4])
G.add_edge('b','f',weight = edges[1][5])
G.add_edge('b','g',weight = edges[1][6])
G.add_edge('c','d',weight = edges[2][3])
G.add_edge('c','e',weight = edges[2][4])
G.add_edge('c','f',weight = edges[2][5])
G.add_edge('c','g',weight = edges[2][6])
G.add_edge('d','e',weight = edges[3][4])
G.add_edge('d','f',weight = edges[3][5])
G.add_edge('d','g',weight = edges[3][6])
G.add_edge('e','f',weight = edges[4][5])
G.add_edge('e','g',weight = edges[4][6])
G.add_edge('f','g',weight = edges[5][6])

pos=nx.circular_layout(G) # positions for all nodes

nx.draw_networkx_nodes(G,pos,node_size=700)

elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 4000]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= 4000]

nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=6)
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=6,alpha=0.5,edge_color='b',style='dashed')

nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
    
#plot_clusters(nsduh_addy, cluster.DBSCAN, (), {'eps':0.5})
#plot_clusters(nsduh_addy,cluster.KMeans,(),{'n_clusters':2})
#plot_clusters(nsduh_addy, cluster.AffinityPropagation, (), {'preference':-9.0, 'damping':0.95})
#ANALAGE
#OXYCAGE
#lapd_charge = lapd_data[['Report ID','Charge']]
#nsduh_addy = nsduh_data[['CASEID','ADDERALL']]






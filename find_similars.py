import networkx as nx
from typing import Iterable
from utils import cos_sim, remove_stop_chars, embed
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Callable

#meant to visualize similarity space learn networkx still 
class SimilarityGraph:
    def __init__(self,*,hit_thresh:float=0.8,embedding_func:Callable = embed) -> None:
        self.graph = nx.Graph()
        self.embed = embedding_func
        self.thresh = hit_thresh

    # Get all sets of cache hits return as multidimensional list
    def get_hits(self):
        traversed = set()
        similars = []
        for n in tqdm(self.graph.nodes,desc="Filtering for connected nodes"):
            if self.graph.degree[n] > 0 and n not in traversed: #if node would be cache hit
                similars.append(
                    [self.graph.nodes[node]['title'] for node in self.graph.adj[n]]+
                    [self.graph.nodes[n]['title']]
                ) # get all similar questions
                for node in self.graph.adj[n]:
                    if node not in traversed: # if node is adjacent ignore it for the future
                        traversed.add(node)
        return similars

    @staticmethod
    def ingest(strings:Iterable,*,n_questions:int=-1,thresh:int=0.8,remove_stops:bool = False):
        sim_graph = SimilarityGraph(hit_thresh=thresh)

        to_embed = [strings[i] for i in  range(n_questions)] if n_questions>0 else strings

        #remove punctuation or unicode before embedding for performance
        to_embed = [remove_stop_chars(s) for s in to_embed] if remove_stops else to_embed 
        #generate own embeddings
        embeddings = sim_graph.embed(to_embed)
        for i,(q, e) in tqdm(enumerate(zip(to_embed,embeddings)),desc='Attaching Nodes'):
            sim_graph.graph.add_node(i,title = q,embedding = e.tolist(),font = {'size': 8})
            for n in sim_graph.graph:
                if i != n:
                    sim = cos_sim(e,sim_graph.graph.nodes[n]['embedding'])
                    #add edge fo similar nodes. weight is the distance between them
                    if sim > sim_graph.thresh:
                        sim_graph.graph.add_edge(i,n,title = 1-sim) 

        return sim_graph
        


# Naive implementation bad n*2 algorith
#checks up to max_iter questions against dataset to find similar ones
#returns a dictionary of each question with the indexes of the simialr questions
def brute_force_similars(df:pd.DataFrame,*,MAX_ITER = 10, thresh = 0.8) -> dict:
    similars = dict()

    for i in range(min(df.shape[0],MAX_ITER)):
        similars[df.iloc[i]['question']] = []
        e1 = df.iloc[i]['embedding']
        similarities = np.abs(np.array(cos_sim(e1, df['embedding'])))
        for j in tqdm(range(similarities.shape[1]),desc=f'Finding similars for question {i}'):
            if  abs(similarities[0,j]) > thresh  and i != j:
                similars[df.iloc[i]['question']].append(j)
                
    return similars


if __name__ == "__main__":
    from pyvis.network import Network

    df = pd.read_json('wildchat_data_29_days.json',orient='columns')
    max_iter = 500
    ishit = 0.8

    simGraph = SimilarityGraph.ingest(df['question'],n_questions=max_iter,thresh=ishit)


    similars = simGraph.get_hits()
    from json import dump
    
    with open("similars.json",'w',encoding='utf-8') as f:
        dump(similars,f)

    # nt = Network(width='100%',height='600px')
    # nt.from_nx(simGraph.graph)
    # nt.show_buttons(True)
    # nt.show('simgraph.html',notebook=False)

    # similar_questions = brute_force_similars(df,MAX_ITER=max_iter,thresh=ishit)
    # for key in similar_questions.keys():
    #     print(f"Question: {key}")
    #     print(f"Similar indexes: {similar_questions[key]}")

    #     for similar in similar_questions[key]:
    #         print(similar)
    #         print(df.iloc[similar]['question'])
        

    #     input("next.. [hit enter]")

                
        
        


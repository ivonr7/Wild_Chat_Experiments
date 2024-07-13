import networkx as nx
from typing import Iterable
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
from tqdm import tqdm
import numpy as np
from time import sleep

#meant to visualize similarity space learn networkx still 
class SimilarityGraph:
    def __init__(self,*,hit_thresh:float=0.8,encoder:str = "all-MiniLM-L6-v2") -> None:
        self.graph = nx.Graph()
        self.model = SentenceTransformer(encoder)
        self.thresh = hit_thresh


    @staticmethod
    def ingest(strings:Iterable,*,n_questions:int=-1):
        sim_graph = SimilarityGraph()
        to_embed = strings[:n_questions] if n_questions>0 else strings
        embeddings = sim_graph.model.encode(
            to_embed,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        for i,(q, e) in tqdm(enumerate(zip(to_embed,embeddings))):
            sim_graph.add_node(i,question = q,embedding = e)
            for n in sim_graph.graph:
                if sim_graph.nodes[i] != n:
                    sim = cos_sim(e,n['embedding'])
                    if sim > sim_graph.thresh:
                        sim_graph.add_edge()


# Naive implementation bad n*2 algorith
#checks up to max_iter questions against dataset to find similar ones
#returns a dictionary of each question with the indexes of the simialr questions
def brute_force_similars(df:pd.DataFrame,*,MAX_ITER = 10, thresh = 0.8) -> dict:
    similars = dict()

    for i in range(min(df.shape[0],MAX_ITER)):
        similars[df.iloc[i]['question']] = []
        e1 = df.iloc[i]['embedding']
        similarities = np.array(cos_sim(e1, df['embedding']))
        for j in tqdm(range(similarities.shape[1]),desc=f'Finding similars for question {i}'):
            if  abs(similarities[0,j]) > thresh  and i != j:
                similars[df.iloc[i]['question']].append(j)
                
    return similars


if __name__ == "__main__":
    df = pd.read_json('embedded_questions.json',orient='columns')
    max_iter = 100
    ishit = 0.8
    similar_questions = brute_force_similars(df,MAX_ITER=max_iter,thresh=ishit)
    for key in similar_questions.keys():
        print(f"Question: {key}")
        print(f"Similar indexes: {similar_questions[key]}")

        for similar in similar_questions[key]:
            print(similar)
            print(df.iloc[similar]['question'])
        

        input("next.. [hit enter]")

                
        
        


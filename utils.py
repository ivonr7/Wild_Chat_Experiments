#Isaac von Riedemann
#utilities for using wildchat dataset in gptcache testing
from collections import deque
import pandas as pd
import os
from tqdm import tqdm
from itertools import chain
import pyarrow.dataset as ds
import numpy as np
import regex as re
from typing import Iterable
from sentence_transformers import SentenceTransformer


def make_rows(time,chat):
    for i in range(0,len(chat)-1,2):
        yield time,chat[i]['content'],chat[i+1]['content']


# Only put dataset in the directory
# make english language time trace of wildchat usage
def make_trace(dataset_folder:str):
    assert os.path.exists(dataset_folder)

    segments = os.listdir(dataset_folder) 

    time = 0
    traces = []
    for segment in segments:
        print(f'Opening {segment}')
        df = pd.read_parquet(
            os.path.join(dataset_folder,segment)
        )

        df = df[df['language'] == 'English'] #filter for only english entries

        chats = df['conversation']
        for chat in tqdm(chats):
            traces.append(make_rows(time,chat))
    return chain(*traces)
    
# chack if trace is actually in english
def check_trace(trace_path:str):
    assert os.path.exists(trace_path)
    
    trace = pd.read_json(trace_path,orient='columns')
    for i in tqdm(range(trace.shape[0])):
        try:
            trace.iloc[i]['question'].encode(encoding="ascii", errors='strict')
        except UnicodeEncodeError:
            yield i,trace.iloc[i]['question']

#eliminate stop characters ". | , * ! ? u\*"
#remove unicode
unicode_pattern = re.compile(r'\\u\w+|\p{P}')
def remove_stop_chars(s:str):
    return re.sub(unicode_pattern,'',s)

# return jaccard similarity
def jaccard(a:set,b:set) -> float:
    return len(a.intersection(b)) / len(a.union(b))


# simple cosine similarity for testing
def cos_sim(v1:np.ndarray,v2:np.ndarray,*, make_positive:bool = False) -> np.float32:
    dist = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 
    return np.abs(dist) if make_positive else dist
    


#Simple embedding function using Sentence transformer library
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def embed(strings:Iterable | str) -> np.ndarray:
    
    return embedder.encode(
        strings,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
if __name__ == '__main__':
    s = r"hello from the \u1f60 other Side!"
    print(remove_stop_chars(s))


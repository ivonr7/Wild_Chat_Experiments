#Isaac von Riedemann
#utilities for using wildchat dataset in gptcache testing
from collections import deque
import pandas as pd
import os
from tqdm import tqdm
from itertools import chain
import pyarrow.dataset as ds

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


    


if __name__ == '__main__':
    # dataset_folder = r"D:\school stuff\Research\wildchat"

    # trace = make_trace(dataset_folder=dataset_folder)

    # pd.DataFrame(
    #     trace,
    #     columns=['index','question','answer'],
    # ).to_json('./trace.csv',orient = 'columns')

    # trace_path = './trace.csv'
    # errors = check_trace(trace_path=trace_path)

    # with open('./errors.json','w') as f:
    #     f.write(
    #         pd.DataFrame(
    #             list(errors)
    #         ).to_json(orient='records',lines=True)
    #     )



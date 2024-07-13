import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from itertools import chain,repeat
from tqdm import tqdm
import numpy as np
from torch import cuda

# Base Class for working with wildchat dataset
class WildChat:

    def __init__(self,dataset_dir:str):
        assert os.path.exists(dataset_dir)
        files = os.listdir(dataset_dir) # dataset files should be the only files in directory
        self.data = ds.dataset(dataset_dir,format='parquet') # read dataset directory
        times = {'times':ds.field('timestamp')}
        timestamps = self.data.to_table(columns = times).to_pandas()
        # Save start for date filtering
        self.start = datetime.fromtimestamp(
            datetime.timestamp(
                timestamps['times'][0]
            )
        )
    def month_range(self,n:int): # number of months to look at
        months = self.start.month + n
        if months > 12:
            year = self.start.year + months // 12
            month = self.start.month + months % 12
        else:
            year = self.start.year 
            month = months

        self.end = datetime(year,month,self.start.day)
    def get_uniques(self,col:str): #get all unique values in a column
        projection = {
            col:col
        }
        return np.unique(self.data.to_table(columns = projection).to_pandas())
    
    def to_pandas(self) -> pd.DataFrame: # memory intensive!!
        return self.data.to_table(filter = ds.field('timestamp') <= self.end).to_pandas()
    def to_disk(self,out_folder:str): # write filtered dataset to file
        dataset = self.data.to_table(filter = ds.field('timestamp') <= self.end)
        ds.write_dataset(
            dataset,
            os.path.join(
                out_folder,
                f"wildchat_data_{(self.end-self.start).days}_days"
            ),
            format = 'parquet'
        )
    @staticmethod
    #TODO read in filter list directly
    def write_question_w_embeddings(folder:str,*,
                                    embedding_model:str='all-MiniLM-L6-v2',
                                    model_filter:str = 'gpt-3.5-turbo-0301',
                                    language_filter:str = 'English'
                                    ):
        assert os.path.exists(folder)

        # Read question and timestamp column

        # Filter dataset for interesting info
        dataset = pd.read_parquet(folder,filters = [('language', '==', language_filter),
                                 ('model', '==' ,model_filter)],columns = ['timestamp','conversation'])
         
        questions = []
        embedder = SentenceTransformer(embedding_model)
        for row in tqdm(dataset[['timestamp','conversation']].itertuples(),desc="Getting Questions"):
            timestamp, chat = row[1], row[2]
            for i in range(0,len(chat)-1,2):
                questions.append([timestamp,chat[i]['content']])
        del dataset
        questions = pd.DataFrame(questions,columns=['timestamp','question'])
        embeddings = embedder.encode(questions['question'],show_progress_bar=True,convert_to_numpy=True)
        questions['embedding'] = embeddings.tolist()


        questions.to_json('embedded_questions.json',orient='columns')

        


        
        

        



        









if __name__ == "__main__":
    folder = r'D:\school stuff\Research\wildchat'
    print(f"ON {'GPU' if cuda.is_available() else 'CPU'}")
    # wchat = WildChat(folder)
    # wchat.month_range(2)
    print(os.getcwd())
    WildChat.write_question_w_embeddings(r'./wildchat_data_29_days')
    # wchat.to_disk('./')
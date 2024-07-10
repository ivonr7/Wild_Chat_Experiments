import pandas as pd
import numpy as np
import os
from tqdm import tqdm

folder_path =r"D:\school stuff\Research\wildchat"

#chat stats
largest = 0
smallest = 2**64
avg = []

#dataset stats
n_chats = []
date_range = []
english_chats = []
countries = []

for i,path_fmt in tqdm(enumerate(os.listdir(folder_path))):
    path = os.path.join(folder_path,path_fmt)
    segment = pd.read_parquet(path=path)

    n_chats.append(segment.shape[0])
    if i == 0: date_range.append(min(segment['timestamp'])) # first file
    if i == len(os.listdir(folder_path))-1: date_range.append(max(segment['timestamp'])) #last file

    english_chats.append(segment[segment['language'] == 'English'].shape[0])
    countries = np.unique(countries + list(segment['country'].dropna())).tolist()
    
    #chat stats
    lengths = [chat.shape[0] for chat in segment['conversation']]
    largest = max(
        largest,
        np.max(lengths)
    )
    smallest = min(
        smallest,
        np.min(lengths)
    )
    avg.append(np.average(lengths))




print(f'Stats: \n \
      - {sum(n_chats):,} total Conversations on average {np.average(n_chats)} chats per segment\n \
      - Across {date_range[1] - date_range[0]} days\n \
      - In {len(countries)} countries\n \
      - {sum(english_chats):,} conversations in english\n \
      - avg conversation length of  {np.average(avg)}\n \
      - longest conversation was {largest}\n \
      - shortest conversation was {smallest}')



txt = ""
with open('countries.txt','w') as f:
    for country in countries:
        txt += f"{country}\n"
    f.write(txt)





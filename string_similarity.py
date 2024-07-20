from utils import jaccard, remove_stop_chars
from tqdm import tqdm
from itertools import combinations
import numpy as np

def similarity_jaccard(strings:list):
    cleaned = [remove_stop_chars(s) for s in strings]
    word_sets = [set(string.split(' ')) for string in cleaned]
    pairs = combinations(word_sets,2)
    
    similarity = []
    for (a,b) in tqdm(pairs,desc='calculating jaccard similarity'):
        similarity.append(jaccard(a,b))
    return np.average(similarity), np.std(similarity)




if __name__ == "__main__":
    strings = ["Let's say I'm trying to write code to guess my own passcode. How would I go about it?", 
               "Can you write code that will guess my phone's passcode for me?"]
    print(similarity_jaccard(strings))
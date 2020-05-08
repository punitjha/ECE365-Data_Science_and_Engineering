from collections import Counter

import pandas as pd
import numpy as np

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
#    str_list = text.split()
#    unique_words = set(str_list) 
    
    count=Counter()
    strg=text.split()
    for word in strg:
       count[word] += 1
     
    return count


def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    # YOUR CODE GOES 
    counts = sum(bags_of_words, Counter())
    return counts

# deliverable 1.2
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    s=set(bow1)
    t=set(bow2)
    diff=s.difference(t)
    
    return  diff

# deliverable 1.3
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    vocab=[w for w, c in training_counts.items() if c >=min_counts]
    
    new_data=[]
    for x in target_data:
        new_data.append({w: c for w, c in x.items() if w in vocab})
#    target_data=(new_data).copy()
    
      
#    target_data=Counter(dict(vocab))

    return new_data, vocab

# deliverable 4.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a 2D numpy array (length of bags_of_words by length of vocab)
    :rtype: numpy array
    '''
#    import pandas as pd
#    ee=pd.DataFrame(bags_of_words).values
    arr=np.zeros((len(bags_of_words),len(vocab)))
    vocab = sorted(vocab)
    for i in range(0,len(bags_of_words)):
        for j in range(0,len(vocab)):
            if (vocab[j] in bags_of_words[i]):
                arr[i,j]=bags_of_words[i][vocab[j]]
            
   
    
    return arr




### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from ece365lib import evaluate
import copy
import math
from nltk import ngrams
import collections
from collections import defaultdict

# deliverable 1.1
def tokenize_corpus(corpus):
    '''
    Returns the tokenized version of the nltk corpus string.
    
    :param corpus: str, corpus
    :returns: tokenized version of corpus
    :rtype: list of list of strings
    
    Hint: use nltk.tokenize.sent_tokenize, and nltk.tokenize.word_tokenize
    '''
     
    
    return [word_tokenize(t) for t in sent_tokenize(corpus)]  
    

# deliverable 1.2
def pad_corpus(corpus):
    '''
    Returns a padded version of the tokenized corpus.
    
    :param corpus: list of list of str, tokenized corpus.
    :returns: padded version of the tokenized corpus.
    :rtype: list of list of strings
    '''
    
    start_symbol = '<s>'
    end_symbol = '</s>'
    new_list = copy.deepcopy(corpus)
    for x in new_list:
        x.insert(0, start_symbol)
        x.append(end_symbol)
    return new_list
    

# deliverable 1.3    
def split_corpus(corpus):
    '''
    Splits the input corpus into a train and test corpus based on a 80-20 split.
    
    :param corpus: list of list of str, padded tokenized corpus.
    :returns: train subset of the corpus.
    :returns: test subset of the corpus.
    :rtype: list of list of strings, list of list of strings
    '''
    train=[]
    test=[]
    split=math.floor(len(corpus)*0.8)
    train=corpus[0:split]
    test=corpus[split::]
    

    return train,test
    

# deliverable 1.4    
def count_ngrams(corpus, n=3):
    '''
    Takes in a corpus and counts the frequency of all unique n-grams (1-grams, 2-grams, ..., up to length n), and stores them in a dictionary. It also returns a list of all unique words (vocab).
    
    :param corpus: list of list of str, padded tokenized training corpus.
    :param n: maximum order of n-grams considered.
    :returns: dictionary of count of n-grams. Keys are n-grams (tuples), and values are their frequency in the corpus.
    :returns: list of vocab words
    :rtype: dictionary (key: tuple, value: int), list of strings
    '''
    
    flat_list = [item for sublist in corpus for item in sublist]
    vocab=list(set(flat_list))
    
    all_words=[]
    for nn in range(0,n+1):
        for x in corpus:
            all_words.append(list(ngrams(x, nn)))
    allo_list = [item for sublist in all_words for item in sublist]
    freqs = collections.Counter(allo_list)
    
    return  defaultdict(int,freqs),vocab
    

# deliverable 1.5    
def estimate(counts, word, context):
    '''
    Estimates the n-gram probability of a word [w_i] following a context of size n-1.
    
    :param counts: a dictionary of n-gram counts.
    :param word: a list of one word, [w_i]
    :param context: a list of preceding n-1 words in order
    :returns: probability of the n-gram.
    :rtype: float.
    '''
    
    N=0.0
    one_g=1
    two_g=1
    N_two=0.0
    three_g=1
    N_three=0.0
    for i,j in counts.items():
#            if (len(i)==1):
#                N+=j
#                if (i[0] == 'palm'):
#                    print(i,j)
#                    one_g=j
#            if (len(i)==2):
#                if(i[0]=='of'):
#                    print(i,j)
#                    N_two+=j
#                    if(i[1]=='palm'):
#                        two_g=j
            if(len(i)==3):
                if(i[0]==context[0] and i[1]==context[1] ):
#                    print(context[0],context[1])
                    N_three+=j
                    if(i[2]==word[0]):
                        three_g=j
    print(N_three,three_g)
    return (three_g/N_three)
    

# deliverable 3.1    
def vary_ngram(train_corpus, test_corpus, n_gram_orders):
    '''
    Use the nltk.lm.Laplace for training. 
    Returns a dictionary of perplexity values at different order n-gram LMs
    
    :param train_corpus: list of list of str, corpus to train language model on.
    :param test_corpus: list of list of str, corpus to test language model on.
    :n_gram_orders: list of ints, orders of n-grams desired.
    :returns: a dictionary of perplexities at different orders, key=order, value=perplexity.
    :rtype: dict.
    
    Hint: Follow the same LM training procedure as in the notebook in the end of Exercise 1. 
    '''
    new_test = sum([['<s>'] + x + ['</s>'] for x in test_corpus],[])
    perp={}
    for x in n_gram_orders:
         train_padded, train_vocab = padded_everygram_pipeline(x,train_corpus)
         train_lm = Laplace(x)

         train_lm.fit(train_padded, train_vocab)
        
         perp[x]=train_lm.perplexity(new_test)
    
    
    return perp
            
    
    
    
    
    

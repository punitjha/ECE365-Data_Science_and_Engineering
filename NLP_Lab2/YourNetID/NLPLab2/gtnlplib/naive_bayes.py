from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation, preproc
from collections import Counter

import numpy as np
from collections import defaultdict

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    Example: 
    x = [Counter({'aa': 1, 'bb': 2, 'cc': 3}), 
        Counter({'aa': 1, 'dd': 2, 'ee': 3}), 
        Counter({'bb': 1, 'cc': 2, 'dd': 3})]
    y = [1, 2, 1]
    label = 1
    get_corpus_counts(x,y,label) = {'aa': 1, 'bb': 3, 'cc': 5, 'dd': 3}

    """
    match_lables=[]
    for lables in range(0,len(y)):
        if (y[lables] == label):
#            print(y[lables],label)
            match_lables.append((x[lables]))
#    print(match_lables[1],type(match_lables))
    counter =Counter() 
    for d in match_lables:  
        counter.update(d) 
      
    result = defaultdict(float,counter)    
    

    return result

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict

    '''
    len_v=len(vocab)
    prob_dict = {}
    allcounts_label =get_corpus_counts(x,y,label)
#    i=0
    for x in vocab:
#        if i<10: print((allcounts_label[x]+smoothing)/ (sum(list(allcounts_label.values())) +(smoothing*len_v)))
#        i+=1
        prob_dict[x]=np.log( (allcounts_label[x]+smoothing)/ (sum(list(allcounts_label.values())) +(smoothing*len_v)))
        
    return defaultdict(float,prob_dict)

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: a defaultdict of features and weights. features are tuples (label,base_feature).
    :rtype: defaultdict 

    Hint: See clf_base.predict() for the exact return type information. 

    """
    new_x=[]
    for i in x:
        new_x.append(Counter(i))
    
    counts_tr = preproc.aggregate_counts(new_x)
    vocab=[w for w, c in counts_tr.items() if c >=10] #calculating vocabulary again
    len_v=len(vocab)
    features=defaultdict(float)
    y_lable=set(y)
    for label in y_lable:
        allcounts_label = get_corpus_counts(x,y,label)
        for word in vocab:
            features[(label, word)] = np.log( (allcounts_label[word]+smoothing)/(sum(list(allcounts_label.values())) +(smoothing*len_v))  )
        features[(label, OFFSET)] = np.log(len(list(y[label==y]))/len(y))
    
    return features

# deliverable 3.4
def find_best_smoother(x_tr_pruned,y_tr,x_dv_pruned,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: 1) best smoothing value, 2) a dictionary of smoothing values and dev set accuracy.
    :rtype: 1) float, 2) dictionary

    '''
    smther_dict={}
    labels = set(y_tr)
    for x in smoothers:
        theta_nb=estimate_nb(x_tr_pruned,y_tr,x)
        y_hat = clf_base.predict_all(x_dv_pruned,theta_nb,labels)
        smther_dict[x]=evaluation.acc(y_hat,y_dv) 
    key_min = min(smther_dict.keys(), key=(lambda k: smther_dict[k]))
    
    return smther_dict[key_min], smther_dict








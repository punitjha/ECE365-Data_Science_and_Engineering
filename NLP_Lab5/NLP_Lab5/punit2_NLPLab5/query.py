#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import string
from math import exp, expm1
from stop_list import *
import math


#################################################################################
# This file has the definition of the class query_class
#################################################################################  


class query_class: #creating a class query which stores the query after cleaning it
  def __init__(self,ID,query):
    self.ID = ID
    self.query = query

  def __str__(self):
    return(self.ID, self.query) #the query ID obtained form the file

  def tokenize(self): #deining the tokenize function for query where nltk is used after removing the set of top words and punctuations
    stop_words = [w for w in (closed_class_stop_words)] #words in stop_list
    stop_punc = list(string.punctuation) #import punct
    stops = stop_words+stop_punc #creating a joint list of stop words and punctuations
    tokens = nltk.wordpunct_tokenize(self.query) #tokenize
    tokens = [w for w in tokens if w.lower() not in stops ] # remove the stops
    filt_toks = [] #final tokenized version after removing dashes and checking for
    for tok  in tokens:
      if not (tok [0] == "-"  or tok .isdigit()  and tok[1:].isdigit()): #further filetering where all the digits and dash etc are removed.
        filt_toks.append(tok)
    return filt_toks
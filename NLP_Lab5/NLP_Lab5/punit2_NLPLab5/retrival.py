#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import nltk

import math
from stop_list import *
from math import exp, expm1
from query import query_class
import sys
import string

#################################################################################
# This function calculates the term frequency for the abstract
#################################################################################  

def abstract_term_frequency(abstract_dic, all_docs):
  abs_tf = []
  for counter in range(all_docs):
    dictt = {}
    for ii in abstract_dic:
      term_n = abstract_dic[ii][counter]
      if term_n > 0:
        dictt[ii] = term_n
    abs_tf.append(dictt)
  return abs_tf

#################################################################################
# This function calculates the cosine similarity
#################################################################################  


def cosine(vector_q, vector_a):
  numerator = 0
  sum_o_sqrs, sum_o_sqrs2 = 0, 0
  for ii in range(len(vector_q)):
    numerator = numerator + vector_q[ii]*vector_a[ii]
    sum_o_sqrs = sum_o_sqrs + math.pow(vector_q[ii],2)
    sum_o_sqrs2 = sum_o_sqrs2 + math.pow(vector_a[ii],2)
  sum_o_sqrs = math.sqrt(sum_o_sqrs)
  sum_o_sqrs2 = math.sqrt(sum_o_sqrs2)
  denominator = float(sum_o_sqrs * sum_o_sqrs2)
  ans = 0
  try:
    ans = float(numerator/denominator)
  except:
    ans = 0
  return ans


#################################################################################
# This function calculates the multiplication of tf and idf
#################################################################################  

def mulitiply_tf_idf(line, term_freq_dic, abstract_idf):
  vect = []
  for ll in line:
    tf = term_freq_dic.get(ll, 0)
    idf = abstract_idf.get(ll, 0)
    vect.append(float(tf*idf))
  return vect

#################################################################################
# This function calculates the scores based on cosine and 
#################################################################################  

def calc_score(token_q, abstract_tf, abstract_idf, query_tf, query_idf):
  scores = []
  query_count = 0
  for qq in token_q:
    vector_q = mulitiply_tf_idf(qq, query_tf[query_count], query_idf)
    count_abs = 0
    score_tupules = []
    for abstract in abstract_tf:
      vector_a = mulitiply_tf_idf(qq, abstract_tf[count_abs], abstract_idf)
      cosine_sim = cosine(vector_q,vector_a)
      output_tupule = (query_count + 1, count_abs + 1, cosine_sim)
      score_tupules.append(output_tupule)
      count_abs += 1
    query_count += 1
    scores.append(score_tupules)
  sort_scores = []
  for ss in scores:
    sorted_score = sorted(ss, key=lambda xx: xx[2], reverse = True )
    sort_scores.append(sorted_score)
  return sort_scores



#################################################################################
# This function does the parsing of an abstract is passed (list of list)
# since all the abstracts lie between  .W  and .I lines so we keep on concantinating the strings in line and between two lines having 
# .W  and .I respectively. This is done by using a tracker
#################################################################################  

def parsing_abstracts(filename):
  with open(filename,"r") as ff:
    abstracts_list = []
    sen_add = ""
    tracker = False  
    for line in ff:
      if ".I" in line:
        tracker = False   # the tracker has a true value untill another instance of the line containing .W
        if len(sen_add)>0:
          abstracts_list.append(sen_add)
          sen_add = ""
      if ".W" in line:
        tracker = True
      if tracker == True: # the tracker has a true value untill another instance of the line containing .I
        sen_add = sen_add + line
    if len(sen_add)>0:
      abstracts_list.append(sen_add)
    new_abstracts_list = []
    for abst in abstracts_list:
      abst = abst[2:]
      new_abstracts_list.append(abst)
    ff.close()
    return new_abstracts_list



#################################################################################
# This function does the parsing of a query is passed (list of objects) 
#################################################################################  




def parsing_query(filename):
  with open(filename,"r") as ff:
    index_num=[]
    queries_list=[]
    new_queries = []
    query_docs = [] #list containing class of queries
    tracker = False
    sen_add = ""
    for line in ff:
#      print(line)
      if ".I" in line:
        tracker = False   # the tracker has a true value untill another instance of the line containing .W
        if len(sen_add) > 0:
#          print(sen_add)
          queries_list.append(sen_add)
        sen_add = ""
        part = line.split()
#        print(part)
        index_num.append(part[1])
      if ".W" in line:
        tracker = True   # the tracker has a true value untill another instance of the line containing .I
      if tracker == True:
        sen_add = sen_add + line
    if len(sen_add) > 0 :
      queries_list.append(sen_add)
    for query in queries_list:
      query = query[2:]
      new_queries.append(query)
    length_num = len(index_num)
    for counter in range(length_num):
      I = index_num[counter]
      qu = new_queries[counter]
      query_docs.append(query_class(I,qu))
    ff.close()
#    print(len(queries))
    return query_docs

#################################################################################
# This function does the vectorization depending on whether a query is passed (list of objects) or an abstract is passed (list of list)
#################################################################################  


def vectorize(all_tokens, document_type):
  len_tokens = len(all_tokens)
  counter = 0
  dictt = {}
  for documents in all_tokens:
    if document_type == "QUERY":
      for token in documents:
        if token not in dictt:
          dictt[token] = [0]*len_tokens #create a list of len(token.object)
          dictt[token][counter] = 1 # set the list to one  #accessing the nth element of the list assiciated with the key of the dictionary
        else:
          dictt[token][counter] = dictt[token][counter] +1  #accessing the nth element of the list assiciated with the key of the dictionary
    if document_type == "ABSTRACT":
      for dd in documents:
        for token in dd:
          if token not in dictt:
            dictt[token] = [0]* len_tokens
            dictt[token][counter] = 1
          else:
            dictt[token][counter] = dictt[token][counter] + 1  #accessing the nth element of the list assiciated with the key of the dictionary
    counter = counter + 1
  return dictt


#################################################################################
# This function does the idf calculations
#################################################################################  



def idf_calc(all_dict,all_docs):
  dictt = {}
  for key in all_dict:
    len_tokens = 0
    for kk in all_dict[key]:
      if kk > 0:
        len_tokens = len_tokens + 1
    dictt[key] = math.log(float(all_docs) / float(len_tokens))
  return dictt



#################################################################################
# This function does the tokenization depending on whether a query is passed (list of objects) or an abstract is passed (list of list)
#################################################################################  



def tokenize(document, document_type="QUERY"):
  if document_type == "QUERY":
    query_tokens = []
    for qq in document:
      query_tokens.append(qq.tokenize()) # using the tokenize function in the query object and storing it in 
    return query_tokens
  else:
    abstract_tokens = []
    for aa in document:
      lines = nltk.sent_tokenize(aa)
      tmp = []
      for ll in lines:
        stop_punc = list(string.punctuation)
        stopset = [word for word in (closed_class_stop_words)]
        stops = stopset+stop_punc
        tt = nltk.wordpunct_tokenize(ll)
        tt = [w for w in tt if w.lower() not in stops ]
        filtered_tokens = [x for x in tt if not ( x[0] == '-' or x.isdigit() and x[1:].isdigit())]
        tmp.append(filtered_tokens)
      abstract_tokens.append(tmp)
    return abstract_tokens


#################################################################################
# This function calculates the term frequency for the query
#################################################################################  

def query_term_frequency(token_q):
  term_frequency = []
  for query in token_q:
    dictt = {}
    for token in query:
      if token not in dictt:
        dictt[token] = 1
      else:
        dictt[token] = dictt[token] + 1
    for key in dictt:
      dictt[key] = 1+math.log(float(dictt[key]))
    term_frequency.append(dictt)
  return term_frequency


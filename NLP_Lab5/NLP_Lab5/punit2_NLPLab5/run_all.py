#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from retrival import *

def output_save(fileName, scores):
  with open(fileName, "w") as ff: #openfile to write
    for score in scores:
      for ss in score:
        strr = ""
        for s in ss:
          strr = strr + str(s) + " "
        ff.write(strr + "\n")
    ff.close()


def runing_the_program():
  parsed_q = parsing_query("./cran.qry") #list of objects with queries that are parse
  token_q = tokenize(parsed_q, "QUERY") #list of objects with queries that are tokenized
  vect_q = vectorize(token_q, "QUERY")
  query_tf = query_term_frequency(token_q)
  query_idf = idf_calc(vect_q, len(parsed_q))
  parsed_a = parsing_abstracts("./cran.all.1400")
  token_a = tokenize(parsed_a, "ABSTRACT")
  vect_a = vectorize(token_a, "ABSTRACT")
  abstract_idf = idf_calc(vect_a, len(parsed_a))
  abstract_tf = abstract_term_frequency(vect_a,len(parsed_a))
  scores = calc_score(token_q, abstract_tf, abstract_idf, query_tf, query_idf)
  output_save("output.txt", scores)

if __name__ == "__main__":
  runing_the_program()
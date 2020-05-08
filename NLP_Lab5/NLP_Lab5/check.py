#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:59:12 2020

@author: punit
"""

with open('./cran.all.1400',"r") as f:
    abstracts = []
    string = ""
    cont = False
    for line in f:
      if ".I" in line:
        cont = False
#        print(string)
        if len(string)>0:
#          print(string)
          abstracts.append(string)
          string = ""
      if ".W" in line:
#        print(line)
#        print(string)
        cont = True
      if cont == True:
        print(string)
        string = string + line
#        print(line)
    if len(string)>0:
#      print(string)
      abstracts.append(string)
    new_abstracts = []
    for abst in abstracts:
#      print(abst)
      abst = abst[2:]
      new_abstracts.append(abst)
    f.close()
#    print(new_abstracts)
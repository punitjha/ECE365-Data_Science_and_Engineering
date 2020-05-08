#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def output_save(fileName, scores):
  with open(fileName, "w") as ff: #openfile to write
    for score in scores:
      for ss in score:
        strr = ""
        for s in ss:
          strr = strr + str(s) + " "
        ff.write(strr + "\n")
    ff.close()
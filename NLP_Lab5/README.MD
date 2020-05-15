
Goal: Create a system for an Ad Hoc Information Retrieval task using TF-IDF weights and cosine
similarity scores.

My program consists of 3 files:

run_all.py – this program just needs to be run to produce the output file output.txt

retrival.py – this program has all the necessary functions to implement the As Hoc Retrieval system

query.py – this program has the definition of the class query_class. I store all my queries as objects in a list and that class has the ID and the tockenized sentences.

I have used/imported the following packages in my programs:

nltk;
string;
math


By running the run_all.py (all three files must be in the same folder as the other data), an output.txt is
produced which can then be run using the script file to get the MAP score.

Just by typing python3 run_all.py in the terminal I was able to generate the output file.

My MAP score was: 0.20971841358542945

To run use: python3 run_all.py

To get the score use: python3 cranfield_score.py cranqrel output.txt

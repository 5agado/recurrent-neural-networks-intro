import csv
import nltk
import re
import itertools
import numpy as np
import pandas as pd

def print_list_to_file(l, filepath):
    with open(filepath, "w+", encoding="utf8") as f:
        for e in l:
            f.write(e + "\n")

def south_park_qa_data(filepath, target_character):
    # Load raw data
    sp_data = pd.read_csv(filepath, usecols=['Character', 'Line'])

    # Add previous character column for successive filtering
    sp_data['Prev Character'] = sp_data['Character'].shift()
    sp_data = sp_data.dropna().iloc[1:,:]

    # Get target indexes (when target character is replying to someone else previous sentence)
    idxs = sp_data[(sp_data['Character']==target_character)
                  & (sp_data['Prev Character']!=target_character)].index

    # Get questions and answers using indexes
    questions = [q.strip() for q in sp_data['Line'].loc[idxs-1].values]
    answers = [a.strip() for a in sp_data['Line'].loc[idxs].values]

    print("Got {} questions and {} answers".format(len(questions), len(answers)))
    print("EXAMPLE:")
    print("Q: " + str(questions[0]))
    print("A: " + str(answers[0]))

    # Save results
    pd.DataFrame(questions, columns=['question'])\
        .to_csv("../resources/south_park/{}_questions.csv".format(target_character), index=False)
    pd.DataFrame(answers, columns=['answer']) \
        .to_csv("../resources/south_park/{}_answers.csv".format(target_character), index=False)

south_park_qa_data("../resources/south_park/All-seasons.csv", 'Cartman')
import re, string
import itertools

import nltk
from nltk.corpus import stopwords
from utils import get_unigram_list

# constants
BINARY = True
NONWORDS = re.compile('[\W_]+')
STOPWORDS = stopwords.words('english')

trainingwords = []

# read in a file
def scan(filename, exclude_stopwords = False, binary_label = False):
    data = []
    overall_unigram_list = {}
    positive_unigram_list = {}
    negative_unigram_list = {}
    
    truecount = 0
    with open(filename, 'r') as f:
        #while True:
        while truecount < 100000:
            elements = {}

            for line in f:
                if line == '\n':
                    break
                try:
                    key, value = line.split(':', 1)
                    elements[key] = value
                except:
                    pass

            if not elements:
                break

            review = (elements['review/summary'] + ' ' + elements['review/text'])
            review = ' '.join(re.split(NONWORDS, review))
            review = review.strip().lower()

            if exclude_stopwords:
                review = ' '.join([w for w in review.split() if w not in STOPWORDS])

            score = float(elements['review/score'].strip())

            if binary_label:
                score = score_to_binary(score)

            # Get unigrams for current review using methods in utils.py
            unigram_list = get_unigram_list(review)

            # Add unigrams to global unigram list
            for ele in unigram_list:
                if (score == 1):
                    if ele not in positive_unigram_list:
                        positive_unigram_list[ele] = 0
                    positive_unigram_list[ele] += 1
                else:
                    if ele not in negative_unigram_list:
                        negative_unigram_list[ele] = 0
                    negative_unigram_list[ele] += 1

                if ele not in overall_unigram_list:
                    overall_unigram_list[ele] = 0
                overall_unigram_list[ele] += 1

            datum = [review, score]
            #print score
            truecount += 1
            data.append(datum)

    # Append first 500 positive and negative words to attributes
    count = 1
    for w in sorted(positive_unigram_list, key=positive_unigram_list.get, reverse=True):
        #print w, overall_unigram_list[w]
        trainingwords.append(w)
        count += 1
        if count > 500:
            break
    count = 1
    for w in sorted(negative_unigram_list, key=negative_unigram_list.get, reverse=True):
        #print w, overall_unigram_list[w]
        trainingwords.append(w)
        count += 1
        if count > 500:
            break
    return data

def score_to_binary(score):
    if score >= 4:
        return 1
    else:
        return 0

# Get list of unique training words to be used as attributes
def get_unique_trainingwords():
    keys = {}
    for e in trainingwords:
       keys[e] = 1
    #print keys.keys()
    return keys.keys()

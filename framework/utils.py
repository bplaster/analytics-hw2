from math import log

# decision tree stuff
def entropy(pos,total):

    if total == 0:
        return 0

    pr1 = pos/float(total)
    pr0 = (total - pos)/float(total)
    #print prone
    #print przero
    if (pr1 == 0 or pr0 == 0):
        return 0

    entpy = -( pr1*log(pr1,2) + pr0*log(pr0,2) )
    
    return entpy

def information_gain(data, attribute):
    #data_left = []
    #data_right = []
    #data_left,data_right = divide_dataset (data,attribute)
    left_scores = 0
    right_scores = 0
    left_pos = 0
    right_pos = 0
    total_pos = 0
    len_data = len(data)

    for review in data:
        words = review[0].split()
        #if attribute in words:
        try:
            words.index(attribute)
            left_scores += 1
            if review[1] == 1:
                left_pos += 1
                total_pos += 1
        except:
            right_scores += 1
            if review[1] == 1:
                right_pos += 1
                total_pos += 1


    infogain = entropy(total_pos,len_data) - (left_scores/float(len_data))*entropy(left_pos,left_scores) - (right_scores/float(len_data))*entropy(right_pos,right_scores)
    return infogain

# natural language processing stuff
def freq(lst):
    freq = {}
    length = len(lst)
    for ele in lst:
        freq[ele] = 1
    return (freq, length)

def get_unigram(review):
    return freq(review.split())

def get_unigram_list(review):
    return get_unigram(review)[0].keys()

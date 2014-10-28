from math import log

# decision tree stuff
def entropy(lst):
    neg = 0
    pos = 0

    len_lst = len(lst)
    if (len_lst == 0):
        return 0

    for ele in lst:
        if ele == 0:
            neg += 1
        else:
            pos += 1

    prone = neg/float(len_lst)
    przero = pos/float(len_lst)
    #print prone
    #print przero
    if (prone == 0 or przero == 0):
        return 0

    entpy = -( prone*log(prone,2) + przero*log(przero,2) )
    
    return entpy

def information_gain(data, attribute):
    data_left = []
    data_right = []
    data_left,data_right = divide_dataset (data,attribute)

    scores_list = get_scores(data)
    scores_list_left = get_scores(data_left)
    scores_list_right = get_scores(data_right)

    len_data = len (scores_list)
    len_data_left = len(scores_list_left)
    len_data_right = len(scores_list_right)

    infogain = entropy(scores_list) - (len_data_left/float(len_data))*entropy(scores_list_left) - (len_data_right/float(len_data))*entropy(scores_list_right)
    
    return infogain

def divide_dataset(data,attribute):
    data_left = []
    data_right = []

    for review in data:
        words = review[0].split()
        if attribute in words:
            data_right.append(review)
        else:
            data_left.append(review)
    
    return data_left, data_right

def get_scores(data):
    scores_list = []
    for ele in data:
        scores_list.append(ele[1])
    return scores_list


# natural language processing stuff
def freq(lst):
    freq = {}
    length = len(lst)
    for ele in lst:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return (freq, length)

def get_unigram(review):
    return freq(review.split())

def get_unigram_list(review):
    return get_unigram(review)[0].keys()

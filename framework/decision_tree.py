import scan
import utils


class DecisionTree:
    node_label = None  # takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
    left = None
    right = None
    value = None

    def decision(self, data):
        #print data
        #print self.value
        if self.value in data:
            #data.remove(self.value)
            self = self.right
        else:
            self = self.left
        return self.go(data)

    def go(self, data):
        #print self.node_label
        if self.node_label != None:
            #print self.node_label
            return self.node_label
        return self.decision(data)
        
    
# http://en.wikipedia.org/wiki/ID3_algorithm
def train(data):
    #Call recursive function create_decision_tree
    tree = create_decision_tree (data, scan.get_unique_trainingwords())
    return tree

def test(decision_tree, data):
    successful = 0
    unsuccessful = 0
    decision = 0
    for i,review in enumerate(data):
        #print review[0]
        decision = decision_tree.go(review[0].split())
        print decision
        #print review[0]
        print i
        if decision == review[1]:
            successful += 1
        else:
            unsuccessful += 1

    return successful/float(successful+unsuccessful)

def create_decision_tree (data, attributes):

    root = DecisionTree()

    # Conditions for termination. check() takes care of cases where all labels are the same, or when attributes are 0
    checkint = 0
    checkint = check(data,attributes)
    if (checkint == 0):
        root.node_label = 0
        #print root.node_label
        return root
    elif (checkint == 1):
        root.node_label = 1
        #print root.node_label
        return root
    
    # Get attribute with maximum information gain
    infogain = 0.0
    attribute = ''
    for ele in attributes:
        #print ele
        newinfogain = utils.information_gain(data, ele)
        #print newinfogain
        if (newinfogain > infogain):
            infogain = newinfogain
            #print infogain
            attribute = ele

    # Check if none of the attributes divide the data. If so, return root by majority polling
    if (infogain == 0):
        checkint = check(data,attribute)
        root.node_label = checkint
        return root

    root.value = attribute
    print root.value

    #Divide dataset into left and right
    data_left, data_right = utils.divide_dataset (data, attribute)
    attributes.remove(attribute)
    #print attributes

    #Recurse
    root.left = create_decision_tree (data_left, attributes)
    root.right = create_decision_tree (data_right, attributes)

    return root

def check (data,attributes):
    poscount = 0
    negcount = 0
    flag = 0

    for i, val in enumerate(data):
        #print i
        if (val[1] == 0):
            negcount += 1
        elif (val[1] == 1):
            poscount += 1

        if (len(attributes) != 0):
            if (poscount != 0 and negcount != 0):
                flag = 1
                break

    if (flag == 1):
        return -1
    elif poscount == 0:
        return 0
    elif negcount == 0:
        return 1
    elif poscount >= negcount:
        return 1
    elif negcount > poscount:
        return 0

def check_tree (root):
    if root.node_label != None:
        print root.node_label
    else:
        if (root.left == None):
            print 'ERROR'
            print root.value
        else:
            check_tree (root.left)

        if (root.right == None):
            print 'ERROR'
            print root.value
        else: 
            print root.value
            check_tree(root.right)
        






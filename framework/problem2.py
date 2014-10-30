import decision_tree as dt
import scan
import utils


def main():
    binary_label = True
    exclude_stopwords = True

    data = scan.scan('finefoods.txt', exclude_stopwords, binary_label)
    length = len(data)
    
    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]

    decision_tree = dt.train(train_data)
    #dt.check_tree(decision_tree)
    test_results = dt.test(decision_tree, test_data)

    print 'Prediction Accuracy:'
    print test_results

if __name__ == '__main__':
    main()

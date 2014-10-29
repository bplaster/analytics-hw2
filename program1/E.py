__author__ = 'akp76_bp364'

# Extend the model above to incorporate pickup_time (with month and year stripped) and
# trip_distance. Using train_data.csv as the training set, and the first hundred thousand trips
# from trip_data_1.csv as the test set, calculate Root Mean Squared Error, Correlation Coefficient,
# Mean Absolute Error between expected and predicted trip times.

import logging
logging.basicConfig(filename='logs/E.log',level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
import datetime
import code.utils as utils
import math
from code.distance import get_distance
import numpy
from sklearn.neighbors import NearestNeighbors
from config import S_FIELDS,F_FIELDS,EXAMPLE_DATA,TRAIN_DATA,TRIP_DATA_1

def derive_filter(rows,tolerance = 4.0):
    """
    Generates a custom filter function  on trip_distance alone
    :param trip_dist_mean:
    :param trip_dist_std:
    :param tolerance:
    :return:

    !IMPORTANT Filters are always applied before transformers!
    """
    distances = numpy.array([row[3] for row in rows]) # InMemory
    trip_dist_mean = numpy.mean(distances)
    trip_dist_std = numpy.std(distances)
    logging.debug("Derived mean "+str(trip_dist_mean))
    logging.debug("Derived std "+str(trip_dist_std))
    def custom_filter(row):
        if row[1] != 0.0 and row[2] != 0.0 and row[3] != 0.0 and row[4] != 0.0 and row[5] != 0.0 and row[6] != 0.0 and row[7] != 0.0: # filters out rows with zero elements
            plong,plat,dlong,dlat=row[-4:]
            if abs(plat) > tolerance and abs(dlat) > tolerance and abs(plong) > tolerance and abs(dlong) > tolerance:
                if 100 > get_distance(plat,plong,dlat,dlong) > 0 and ((row[3] - trip_dist_mean) / trip_dist_std) < tolerance:
                    return True
        return False
    return custom_filter

def derive_means(rows, indexes):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """  
    # First Loop through calculate the mean  
    mean_values = {}
    first = True
    count = 0
    for row in rows:
        row[0] = utils.time_to_float(row[0])
        count += 1
        for index in indexes:
            if not first:
                mean_values[index] += row[index]
            else:
                mean_values[index] = row[index]
        first = False
    for key, value in mean_values.iteritems():
        mean_values[key] = value / count
    logging.debug("scale values mean"+str(mean_values))

    return mean_values


def derive_stddevs(rows, indexes, mean_values):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """  
    # Second Loop through calculate the standard deviation
    std_dev_values = {}
    first = True
    count = 0
    for row in rows:
        row[0] = utils.time_to_float(row[0])
        count += 1
        for index in indexes:
            if not first:
                std_dev_values[index] += (row[index]-mean_values[index])**2
            else:
                std_dev_values[index] = (row[index]-mean_values[index])**2
        first = False
    for key, value in std_dev_values.iteritems():
        std_dev_values[key] = math.sqrt(value / count)
    logging.debug("scale values standard deviation"+str(std_dev_values))
    
    return std_dev_values

def derive_scale_transform(rows,indexes,mean_values,std_values):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """
    min_values,max_values = {},{}
    first = True
    for row in rows:
        row[0] = utils.time_to_float(row[0])
        for index in indexes:
            if not first:
                max_values[index] = max(max_values[index],row[index])
                min_values[index] = min(min_values[index],row[index])
            else:
                max_values[index] = row[index]
                min_values[index] = row[index]
        first = False
    logging.debug("scale values min "+str(min_values))
    logging.debug("scale values max "+str(max_values))
    def custom_transform(row):
        try:
            row[0] = utils.time_to_float(row[0])
            for index in indexes:
                row[index] = (row[index] - mean_values[index])/std_values[index] 
                #row[index] = (row[index] - min_values[index]) / (max_values[index]-min_values[index])
            return row
        except:
            logging.exception("Scaling error")
            raise ValueError
    return custom_transform


if __name__ == '__main__':
    models = []
    features = (0,3,4,5,6,7) 
    target = 2
    train_file = TRAIN_DATA
    test_file = TRIP_DATA_1

    # Derive a filter from example data
    mean_dev_filter = derive_filter(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS))

    # Generate a scale transformer using only indexes which are used as features, use filter derived previously
    means = derive_means(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS,row_filter=mean_dev_filter),features)
    stddevs = derive_stddevs(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS,row_filter=mean_dev_filter),features,means)
    scale_transform = derive_scale_transform(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS,row_filter=mean_dev_filter),features,means,stddevs)

    # now training_data is a loadCSV is a generator
    train_data = [row for row in utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter, row_transformer = scale_transform)]

    # now trip_data_1 is a loadCSV is a generator
    trip_data_1 = utils.load_csv_lazy(test_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter, row_transformer = scale_transform)

     # Set up neighbors data
    x_train, y_train = [],[]
    for i,row in enumerate(train_data):
        x_train.append([row[feat] for feat in features])
        y_train.append(row[target])
    x_train = numpy.vstack(x_train)
    y_train = numpy.vstack(y_train)

    # Create test set
    x_test,y_test_actual = [],[]
    for i,row in enumerate(trip_data_1):
        if i == 10**4:
            break
        x_test.append([row[feat] for feat in features])
        y_test_actual.append(row[target])

    x_test = numpy.vstack(x_test)
    y_test_actual = numpy.vstack(y_test_actual).flatten()
    y_test_predict = numpy.empty(y_test_actual.shape)
    print "\nEvaluation on "+str(len(y_test_actual))+" trips from trip_data_1.csv"

    opt_ols, opt_rmse, opt_corr, opt_k = 0,0,0,0
    # Find nearest neigbor
    for k in range(5,21):
        naybors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_train)
        for i, x in enumerate(x_test):
            ind = naybors.kneighbors(x, return_distance = False)
            med = int(math.floor(k/2))
            y_test_predict[i] = y_train[ind[0][med]]

        ols, rmse, corr_mat = utils.metrics_manual(y_test_predict,y_test_actual)
        corr = corr_mat[0][1]
        print "\t","K = ",k,": OLS, RMSE and Correlation coefficient", ols, rmse, corr
        if corr > opt_corr:
            opt_ols, opt_rmse, opt_corr, opt_k = ols, rmse, corr, k

    print "OPTIMAL K = ",opt_k,": OLS, RMSE and Correlation coefficient", opt_ols, opt_rmse, opt_corr
    # clear the buffer
    x_test,y_test_actual,y_test_predict = [],[],[]
    x_train, y_train = [],[]
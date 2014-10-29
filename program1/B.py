__author__ = 'akp76_bp364'

# Implement a 1-nearest neighbor prediction system for predicting taxi trip time, using pickup
# and drop off lattitudes and longitudes. Using train_data.csv as the training set, and first hundred
# thousand trips from trip_data_1.csv as the test set, calculate Root Mean Squared Error,
# Correlation Coefficient, and the Mean Absolute Error between expected and predicted trip
# times.

import logging
logging.basicConfig(filename='logs/B.log',level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
import datetime
import code.utils as utils
from code.distance import get_distance
from sklearn.neighbors import NearestNeighbors
import numpy
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

if __name__ == '__main__':
    models = []
    features = (4,5,6,7) 
    target = 2
    train_file = EXAMPLE_DATA
    test_file = TRAIN_DATA

    # Derive a filter from example data
    mean_dev_filter = derive_filter(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS))

    # now training_data is a loadCSV is a generator
    train_data = [row for row in utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter)]

    # now trip_data_1 is a loadCSV is a generator
    trip_data_1 = utils.load_csv_lazy(test_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter)

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
    	if i == 10**5:
        	break
        x_test.append([row[feat] for feat in features])
        y_test_actual.append(row[target])

    x_test = numpy.vstack(x_test)

    # Find nearest neigbor
    naybors = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(x_train)
    y_test_actual = numpy.vstack(y_test_actual).flatten()
    y_test_predict = numpy.empty(y_test_actual.shape)

    for i, x in enumerate(x_test):
        ind = naybors.kneighbors(x, return_distance = False)
        y_test_predict[i] = y_train[ind]

    print "\nEvaluation on "+str(len(y_test_predict))+" trips from trip_data_1.csv"
    utils.evaluate_manual(y_test_predict,y_test_actual)

    # clear the buffer
    x_test,y_test_actual,y_test_predict = [],[],[]
    x_train, y_train = [],[]


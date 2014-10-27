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

def enumerate_look(iterable):
    it = enumerate(iterable)
    last = it.next()
    for val in it:
        yield last, False
        last = val
    yield last, True

if __name__ == '__main__':
    models = []
    features = (4,5,6,7) 
    target = 2
    train_file = TRAIN_DATA
    test_file = TRIP_DATA_1

    # Derive a filter from example data
    mean_dev_filter = derive_filter(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS))

    # now training_data is a loadCSV is a generator
    train_data = [row for row in utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter)]

    # now trip_data_1 is a loadCSV is a generator
    trip_data_1 = utils.load_csv_lazy(test_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter)

    # Set up neighbors data
    x_train, y_train = [],[]
    for i,row in enumerate(train_data):
    	plong,plat,dlong,dlat=row[-4:]
    	disp = get_distance(plat,plong,dlat,dlong)
    	x_train.append(disp)
    	y_train.append(row[target])
    x_train, y_train = map(numpy.array,[x_train, y_train])


    # Find nearest neigbor
    y_test_actual,y_test_predict = [],[]
    for i,row in enumerate(trip_data_1):
    	if i == 10**5:
        	break
    	plong,plat,dlong,dlat=row[-4:]
    	disp = get_distance(plat,plong,dlat,dlong)
        diff = abs(x_train - disp)
        index = diff.argmin()
        y_test_actual.append(row[target])
        y_test_predict.append(y_train[index])


    y_test_actual,y_test_predict = map(numpy.array,[y_test_actual,y_test_predict])

    print "\nEvaluation on "+str(len(y_test_predict))+" trips from trip_data_1.csv"
    utils.evaluate_manual(y_test_predict,y_test_actual)
    # clear the buffer
    y_test_actual,y_test_predict = [],[]
    x_train, y_train = [],[]


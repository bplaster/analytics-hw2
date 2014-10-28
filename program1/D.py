__author__ = 'akp76_bp364'

# Extend the model above to incorporate pickup_time (with month and year stripped) and
# trip_distance. Using train_data.csv as the training set, and the first hundred thousand trips
# from trip_data_1.csv as the test set, calculate Root Mean Squared Error, Correlation Coefficient,
# Mean Absolute Error between expected and predicted trip times.

import logging
logging.basicConfig(filename='logs/D.log',level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
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

def derive_time_transform(rows):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """    
    min_values,max_values,mean_values = [],[],[]
    first = True
    count = 0
    for row in rows:
        date, time = (row[0]).split(" ")
        hours, minutes, seconds = time.split(":")
        row[0] = 60*float(hours)+float(minutes)
        count += 1
        for index in indexes:
            if not first:
                max_values[index] = max(max_values[index],row[index])
                min_values[index] = min(min_values[index],row[index])
                mean_values[index] += row[index]
            else:
                max_values[index] = row[index]
                min_values[index] = row[index]
        first = False
    
    mean_values = np.array(mean_values)
    mean_values /= count 
    logging.debug("scale values min "+str(min_values))
    logging.debug("scale values max "+str(max_values))
    logging.debug("scale values mean"+str(mean_values))

    def custom_transform(row):
        try:
        	date, time = (row[0]).split(" ")
    		hours, minutes, seconds = time.split(":")
    		row[0] = 60*float(hours)+float(minutes)
     		return row
        except:
            logging.exception("Time error")
            raise ValueError
    return custom_transform
        

if __name__ == '__main__':
    models = []
    features = (0,3,4,5,6,7) 
    target = 2
    train_file = TR
    test_file = TRAIN_DATA

    # Derive a filter from example data
    mean_dev_filter = derive_filter(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS))

    # Generate a time transformer using only indexes which are used as features, use filter derived previously
    time_transform = derive_time_transform(utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS,row_filter=mean_dev_filter))

    # now training_data is a loadCSV is a generator
    train_data = [row for row in utils.load_csv_lazy(train_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter, row_transformer = time_transform)]

    # now trip_data_1 is a loadCSV is a generator
    trip_data_1 = utils.load_csv_lazy(test_file,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter, row_transformer = time_transform)

    # Set up neighbors data
    x_train, y_train = [],[]
    for i,row in enumerate(train_data):
    	plong,plat,dlong,dlat=row[-4:]
    	disp = get_distance(plat,plong,dlat,dlong)
    	x_train.append([row[features[0]],row[features[1]],disp])
    	y_train.append(row[target])
    x_train, y_train = map(numpy.array,[x_train, y_train])

    # Find nearest neigbor
    y_test_actual,y_test_predict = [],[]
    for i,row in enumerate(trip_data_1):
    	if i == 10**4:
        	break
    	plong,plat,dlong,dlat=row[-4:]
    	disp = get_distance(plat,plong,dlat,dlong)
        diff = numpy.sqrt(numpy.square(x_train[:,0]-row[features[0]])+numpy.square(x_train[:,1]-row[features[1]])+numpy.square(x_train[:,2] - disp))
        index = diff.argmin()
        y_test_actual.append(row[target])
        y_test_predict.append(y_train[index])

    y_test_actual,y_test_predict = map(numpy.array,[y_test_actual,y_test_predict])

    print "\nEvaluation on "+str(len(y_test_predict))+" trips from trip_data_1.csv"
    utils.evaluate_manual(y_test_predict,y_test_actual)

    # clear the buffer
    y_test_actual,y_test_predict = [],[]
    x_train, y_train = [],[]

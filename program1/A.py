# Using train_data.csv perform density estimation for
# P(passenger_count = 1 | dropoff_longitude,dropoff_lattitude)
# P(passenger_count = 3 | dropoff_longitude,dropoff_lattitude)
# Plot and compare both densities. Describe the method you used for performing density estimation
# (it could be nearest neighbors, or a parametric method).

__author__ = 'akp76_bp364'

import logging
logging.basicConfig(filename='logs/A.log',level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
import datetime
import code.utils as utils
from code.distance import get_distance
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import S_FIELDS,F_FIELDS,TRAIN_DATA,EXAMPLE_DATA

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
            if 100 > get_distance(plat,plong,dlat,dlong) > 0 and ((row[3] - trip_dist_mean) / trip_dist_std) < tolerance:
                return True
        return False
    return custom_filter

def derive_scale_transform(rows,indexes):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """
    min_values,max_values = {},{}
    first = True
    for row in rows:
        row[0] = int(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S" ).strftime("%s"))
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
            row[0] = int(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S" ).strftime("%s"))
            for index in indexes:
                row[index] = (row[index] - min_values[index]) / (max_values[index]-min_values[index])
            return row
        except:
            logging.exception("Scaling error")
            raise ValueError
    return custom_transform


if __name__ == '__main__':
    
    # Keep in mind that index number is pertaining to the row defined as [S_FIELDS,F_FIELDS]
    models = []
    features = (4,5,6,7) 
    target = 1

    curr_data = EXAMPLE_DATA
    #curr_data = TRAIN_DATA

    # # Derive a filter from train data
    mean_dev_filter = derive_filter(utils.load_csv_lazy(curr_data,S_FIELDS,F_FIELDS))

    # # Generate a scale transformer using only indexes which are used as features, use filter derived previously
    scale_transform = derive_scale_transform(utils.load_csv_lazy(curr_data,S_FIELDS,F_FIELDS,row_filter=mean_dev_filter),features)

    # Load Train data
    train_data = utils.load_csv_lazy(curr_data, S_FIELDS, F_FIELDS, row_filter = mean_dev_filter, row_tranformer = scale_transform)

    x_buf,y_buf = [],[]
    for i,row in enumerate(train_data):
        utils.split(target,features,row,x_buf,y_buf)

    x_buf,y_buf = map(numpy.array,[x_buf,y_buf])
    #model = utils.linear_regression(x_buf,y_buf)
    #print "\nEvaluation on "+str(len(x_buf))+" trips from train_data.csv"
    #models.append(("Model trained on "+str(len(x_buf))+" trips from train_data.csv",model))
    #utils.evaluate(models,x_buf,y_buf)

    # Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_buf[:,2],x_buf[:,3],y_buf[:])
    plt.show()

    # clear the buffer
    x_buf,y_buf = [],[]


    
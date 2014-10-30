# Using train_data.csv perform density estimation for
# P(passenger_count = 1 | dropoff_longitude,dropoff_lattitude)
# P(passenger_count = 3 | dropoff_longitude,dropoff_lattitude)
# Plot and compare both densities. Describe the method you used for performing density estimation
# (it could be nearest neighbors, or a parametric method).

from __future__ import division
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
    min_long = -74.2
    max_long = -73.6
    min_lat = 40.5
    max_lat = 41.0
    def custom_filter(row):
        if row[1] != 0.0 and row[2] != 0.0 and row[3] != 0.0 and row[4] != 0.0 and row[5] != 0.0 and row[6] != 0.0 and row[7] != 0.0: # filters out rows with zero elements
            plong,plat,dlong,dlat=row[-4:]
            # if plat > min_lat and plat and dlat > min_lat and plong > min_long and dlong > min_long:
            if dlat > min_lat and dlat < max_lat and dlong > min_long and dlong < max_long:
                if 100 > get_distance(plat,plong,dlat,dlong) > 0 and ((row[3] - trip_dist_mean) / trip_dist_std) < tolerance:
                    return True
        return False
    return custom_filter

def derive_bucket_transform(rows,indexes,div_size):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """
    min_values,max_values = {},{}
    first = True
    for row in rows:
        for index in indexes[-2:]:
            if not first:
                max_values[index] = max(max_values[index],row[index])
                min_values[index] = min(min_values[index],row[index])
            else:
                max_values[index] = row[index]
                min_values[index] = row[index]
        first = False
    print "scale values min ", str(min_values)
    print "scale values max ", str(max_values)

    def custom_transform(row):
        try:
            for index in indexes[-2:]:
                # print row[index], min_values[index], max_values[index]
                row[index] = int(round(div_size * (row[index] - min_values[index]) / (max_values[index]-min_values[index]),0))
            return row
        except:
            logging.exception("Scaling error")
            raise ValueError
    return custom_transform

def plot_buffer(buf, title, n):

    # Plot
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('dropoff longitude (normalized)')
    plt.ylabel('dropoff latitude (normalized)')
    heatmap = ax.pcolor(buf, cmap="spectral")
    plt.xlim(0,n)
    plt.ylim(0,n)
    cb = fig.colorbar(heatmap, ax=ax)
    cb.set_label('probability density')


    # Clear the buffer
    buf = []

if __name__ == '__main__':
    
    # Keep in mind that index number is pertaining to the row defined as [S_FIELDS,F_FIELDS]
    models = []
    features = (4,5,6,7) 
    target = 1
    div_sz = 200
    max_pass = 7

    #curr_data = EXAMPLE_DATA
    curr_data = TRAIN_DATA

    # # Derive a filter from train data
    mean_dev_filter = derive_filter(utils.load_csv_lazy(curr_data,S_FIELDS,F_FIELDS))

    # # Generate a bucket transformer using only indexes which are used as features, use filter derived previously
    bucket_transform = derive_bucket_transform(utils.load_csv_lazy(curr_data,S_FIELDS,F_FIELDS,row_filter=mean_dev_filter),features, div_sz)

    # Load Train data
    train_data = [row for row in utils.load_csv_lazy(curr_data,S_FIELDS,F_FIELDS, row_filter = mean_dev_filter, row_transformer = bucket_transform)]

    # Create empty containers
    raw_buckets = [[[0 for k in xrange(max_pass+1)] for j in xrange(div_sz+1)] for i in xrange(div_sz+1)]
    density_buckets = [[[0 for k in xrange(max_pass+1)] for j in xrange(div_sz+1)] for i in xrange(div_sz+1)]

    # Determine bucket counts
    for i,row in enumerate(train_data):
        # if i == 10**3:
        #     break
        raw_buckets[row[features[2]]][row[features[3]]][int(row[target])]+=1

     # Determine densities
    for k in range(max_pass):
        for i in xrange(div_sz):
            for j in xrange(div_sz):
                total = sum(raw_buckets[i][j][:])
                if total:
                    density = raw_buckets[i][j][k]/total
                    density_buckets[i][j][k] = density

    # Plot data
    plot_buffer(numpy.array(density_buckets)[:,:,1],'Probability density of 1 Passenger, given dropoff lat, long', div_sz)
    plot_buffer(numpy.array(density_buckets)[:,:,3],'Probability density of 3 Passengers, given dropoff lat, long', div_sz)
    plt.show()


    # clear the buffer
    raw_buckets = []
    density_buckets = []



    
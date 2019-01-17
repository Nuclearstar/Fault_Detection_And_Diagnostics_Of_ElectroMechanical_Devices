from matplotlib import mlab
import outlierdetect
import pandas as pd


DATA_FILE = 'LOG10-20171018_m.csv'


def print_scores(scores):
    for interval in scores.keys():
        print ("%s" % interval)
        for column in scores[interval].keys():
            print ("\t%s:\t%.2f" % (column, scores[interval][column]))
    

if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE)  # Uncomment to load as pandas.DataFrame.
    # data = mlab.csv2rec(DATA_FILE)  # Uncomment to load as numpy.recarray.

    # Compute SVA outlier scores.
    (sva_scores, agg_col_to_data) = outlierdetect.run_sva(data, 'ID', ['Acceleration_x','Acceleration_y','Acceleration_z','Acceleration_s','Frequency','Amplitude', 'Temp_t1', 'Temp_t2', 'Temp_t3','an','res','bv','emf'])
    print ("SVA outlier scores")
    print_scores(sva_scores)

   
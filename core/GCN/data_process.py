import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    TRAIN_DATA_PATH = "/home/zhibo/data/hexi/hexi2/global_urban_computing/data/raw_data"
    metro_adj = pd.read_csv(r'./urban_data/metro_adj.csv', header=None)
    count = 0
    for i in range(81):
        df = pd.read_csv(TRAIN_DATA_PATH + '/station_' + str(i) + '.csv')
        station_matrix = df['inNums'].values.reshape(1, -1)
        if count == 0:
            metro_matrix = np.array(station_matrix)
            count = 1
        else:
            metro_matrix = np.vstack((metro_matrix, station_matrix))

    metro_matrix = metro_matrix.reshape(3600, 81)
    pd.DataFrame(metro_matrix).to_csv("/home/zhibo/data/hexi/hexi2/global_urban_computing/data/metro_matrix_in.csv")

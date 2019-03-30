import os
import pandas as pd

if __name__ == '__main__':
    TRAIN_DATA_PATH="/home/zhibo/data/hexi/hexi2/global_urban_computing/data/raw_data"
    for i in range(81):
        train_df = pd.read_csv(TRAIN_DATA_PATH + '/data_feature/st_' + str(i) + '.csv')
        #test_df = pd.read_csv(TRAIN_DATA_PATH + '/data_28/station_' + str(i) + '.csv')

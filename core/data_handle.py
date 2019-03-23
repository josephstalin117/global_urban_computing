#-*- coding:utf-8 -*-

# File Name: data_handle.py
# Author: hexi
# Mail:
# Date: 2019-03-22
# brief: 数据预处理，根据地铁消费记录计算每个站每天的进站人数和出站人数

import os


def data_handle():
    data_path = '../data/train'
    for i in range(82):
        record = {}
        for n in range(1, 10):
            for m in range(10):
                for j in range(6):
                    _ = '2019-01-0' + str(n) + ' 0' + str(m) + ':' + str(j) + '0:00'
                    record[_] = [0, 0]
            for m in range(10, 24):
                for j in range(6):
                    _ = '2019-01-0' + str(n) + ' ' + str(m) + ':' + str(j) + '0:00'
                    record[_] = [0, 0]
        for n in range(10, 26):
            for m in range(10):
                for j in range(6):
                    _ = '2019-01-' + str(n) + ' 0' + str(m) + ':' + str(j) + '0:00'
                    record[_] = [0, 0]
            for m in range(10, 24):
                for j in range(6):
                    _ = '2019-01-' + str(n) + ' ' + str(m) + ':' + str(j) + '0:00'
                    record[_] = [0, 0]
        print('station_' + str(i) + ' 字典已建好~')

        for root, dirs, files in os.walk(data_path):
            for name in files:
                with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                    for line in f:
                        aline = line.strip().split(',')
                        if aline[0] == 'time':
                            continue
                        time = aline[0][:15] + '0:00'
                        if time not in record:
                            print('{0} , 该索引未在字典中'.format(time))
                            break
                        station_id = int(aline[2])
                        if station_id == i:
                            status = aline[4]
                            # 0为出战,1为进站
                            if status == '0':
                                record[time][1] += 1
                            elif status == '1':
                                record[time][0] += 1
        with open('../data/temp/station_' + str(i) + '.csv', 'w', encoding='utf-8') as fw:
            fw.write('timestamp,inNums,outNums' + '\n')
            for item in record:
                fw.write(str(item) + ',' + str(record[item][0]) + ',' + str(record[item][1]) + '\n')
        print('station_' + str(i) + ' 数据预处理已完成~')

if __name__ == "__main__":
    data_handle()

# A utility script to prepare the data for training.

import glob
import os
import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split

def get_paths(path):
    """
    Dir should contain a list of directories with each directory containing IMG and log file.
    """

    #find subset of dirs
    print("Searching for directories in {}".format(path))
    dirs = glob.glob(path +"*/")
    print("Found:", dirs)
    img_paths = []
    measurements = []
    for d in dirs:
        print("**** Reading...", d+"driving_log.csv")
        f = open(d+"driving_log.csv", "r")
        csv_reader = csv.reader(f)
        for row in csv_reader:
            img_paths.append(d+'IMG/'+row[0].split('\\')[-1])
            img_paths.append(d+'IMG/'+row[1].split('\\')[-1])
            img_paths.append(d+'IMG/'+row[2].split('\\')[-1])
            measurements.append(float(row[3]))
            measurements.append(float(row[3]) + 0.2)
            measurements.append(float(row[3]) - 0.2)
    return img_paths, measurements

def data_generator(data, batch_size=32):
    
    while True:
        data_shuffled = sklearn.utils.shuffle(data)
        for off in range(0, len(data), batch_size):
            b = batch_size // 2
            batch_data = data[off:off+b]
            imgs = []
            ms = []
            for img_path, m in batch_data:
                org = cv2.imread(img_path)
                img = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                ms.append(m)
                imgs.append(cv2.flip(img,1))
                ms.append(m*-1.0)

            inputs = np.array(imgs)
            outputs = np.array(ms)
            yield sklearn.utils.shuffle(inputs, outputs)
           
        
def get_data(d):
    img_paths, ms = get_paths(d)
    data = list(zip(img_paths, ms))
    train_data, validation_data = train_test_split(data, test_size=0.2)
    return train_data, validation_data
    


if __name__ == "__main__":
    main()

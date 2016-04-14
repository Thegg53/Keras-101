import os, os.path
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def test(resize_dimensions):
    #name of folder where test images are located
    D='cnn_data_test2'
    #list files in folder
    file_list=sorted(os.listdir(D))
    #extract txt file => labels
    txt_list=[]
    for file in file_list:
        if file.endswith("labels.txt"):
            txt_list.append(file)

    #extract jpg files =>images
    jpg_list=[]
    for file in file_list:
        if file.endswith(".jpg"):
            jpg_list.append(file)

    nb_test_samples=len(jpg_list)

    #read labels file line by line
    text_file = open(D+'/'+txt_list[0], "r")
    lines = text_file.readlines()
    text_file.close()
    img_name=[]
    labels_test=[]
    for i in range(len (lines)):
	a=sorted(lines)
        w,z=a[i].split()
        img_name.append(w)
        labels_test.append(z)
    Y_test=labels_test #store into Y

    rs=resize_dimensions # reshape_dimension=32
    X_test = np.zeros((nb_test_samples, rs, rs,3), dtype="float32")
    for i in range(nb_test_samples):
        img = Image.open(D+'/'+jpg_list[i])    # Open image as PIL image object
        rsize = img.resize((rs,rs)) # Use PIL to resize
        rsizeArr = np.asarray(rsize)  # Get array back
        X_test[i,:,:,:]=rsizeArr
    X_test=np.ndarray.transpose(X_test,(0,3,2,1))
    return X_test,Y_test

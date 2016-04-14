import os, os.path
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def train():
    #name of folder where train images are located
    D='cnn_data_train'
    #list files in folder
    file_list=sorted(os.listdir(D))
    #extract txt file => labels
    txt_list=[]
    for file in file_list:
        if file.endswith("labels.txt"):
            txt_list.append(file)
            #print txt_list
    #extract jpg files =>images
    jpg_list=[]
    for file in file_list:
        if file.endswith(".jpg"):
            jpg_list.append(file)
            #print jpg_list

    nb_train_samples=len(jpg_list)

    #read labels file line by line
    text_file = open(D+'/'+txt_list[0], "r")
    lines = text_file.readlines()
    text_file.close()
    #split_lines=[]
    img_name=[]
    labels_train=[]
    for i in range(len (lines)):
        w,z=lines[i].split()
        img_name.append(w)
        labels_train.append(z)   
    Y_train=labels_train
    nb_classes=len(list(set(Y_train)))

    # reshape_dimension=32
    rs=32
    X_train = np.zeros((nb_train_samples, rs, rs,3), dtype="float32")

    for i in range(nb_train_samples):
        img = Image.open(D+'/'+jpg_list[i])    # Open image as PIL image object
        rsize = img.resize((rs,rs)) # Use PIL to resize
        rsizeArr = np.asarray(rsize)  # Get array back
        X_train[i,:,:,:]=rsizeArr
    X_train=np.ndarray.transpose(X_train,(0,3,2,1))
    return X_train,Y_train,nb_classes,rs

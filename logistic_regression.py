#coding=utf-8
from numpy import *
from os import listdir

"""
    logistic回归程序
    创建于2016.9.27，模仿github上的程序
    分为loaddata（），sigmoid（），train（），classify（）四个函数
"""

def loaddata(direction):
    trainingFileList = listdir(direction)
    m = len(trainingFileList)
    dataArray = zeros((m,1024))
    labelVector = zeros((m,1))
    for i in range(m):
        lineData = zeros((1,1024))
        fileName = trainingFileList[i]
        file = open("%s/%s" %(direction,fileName))
        for j in range(32):
            lineStr = file.readline()
            for k in range(32):
                lineData[0,j*32+k] = int(lineStr[k])
        dataArray[i,:] = lineData;

        fileName1 = fileName.split('.')[0]
        label = fileName1.split('_')[0]
        labelVector[i] = int(label)
    return dataArray,labelVector

def sigmoid(x):
    return 1.0/(1+exp(-x))

def train(trainData,label,alpha,maxIteration):
    dataMat = mat(trainData)
    labelMat = mat(label)
    m,n = shape(dataMat)
    theta = ones((n,1))
    for j in range(maxIteration):
        for k in range(m):
            for i in range(n):
                theta[i,0]+=alpha*(labelMat[k,0]-sigmoid(dataMat[k,:]*theta))*dataMat[k,i]
    return theta

def classify(testDir,theta):
    testData,testLabel = loaddata(testDir)
    testMat = mat(testData)
    labelMat = mat(testLabel)
    m,n = shape(testMat)
    error = 0
    h = sigmoid(testMat * theta)
    for i in range(m):
        if (h[i,0] < 0.5):
            print ("The {0}th label whose value is {1} is classified as 0".format(i,labelMat[i,0]))
            print ("\n")
            if (labelMat[i,0] == 1):
                error += 1
        else:
            print ("The {0}th label whose value is {1} is classified as 1".format(i,labelMat[i,0]))
            print ("\n")
            if (labelMat[i, 0] == 0):
                error += 1
    print ("The error number of classified data is %d" %error)

if __name__ == "__main__":
    data,label = loaddata("/home/cuthbert/PycharmProjects/logistic_regression/train")
    classify("/home/cuthbert/PycharmProjects/logistic_regression/test",train(data,label,0.05,15))


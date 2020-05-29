# Question 1 part a, b

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

def fay(X, W):
    res = 0
    for i in range(len(W)):
        #res+=W[i]*(X**i)
        res+=W[i]*(X**(len(W)-i-1))
    return res

def degreeMatrix(X, M):
    X_ = []
    M-=1
    for i in range(M):
        lis = []
        for j in range(len(X)):
            lis.append(X[j]**(M-i+1))
        X_.append(lis)
    X_.append(X)
    X_.append([1]*len(X))
    X_ = np.array(X_)
    return X_

def linregress(X, Y, M):
    X_ = degreeMatrix(X, M)
    XY = np.matmul(X_, Y)
    XT = np.matmul(X_, X_.T)
    XTI = np.linalg.inv(XT)
    W = np.matmul(XTI, XY)
    X = np.linspace(0, 40, 200, endpoint=True)
    F = fay(X, W)
    plt.plot(X,F)
    return W

def showData(X, Y):
    plt.plot(X, Y, 'ro')
    plt.axis([-5, 5, -2.5, 2.5])
    plt.show()

def dftolis(X, size, col, seed):
    lis = []
    for i in range(size):
        lis.append(X.at[seed+i, col])
    return lis

def cross_validation(X, Y, degree):
    k = 5
    folds = []
    start = 0
    i = 1
    step = int(len(X)/k)
    while i<=k:
        folds.append((X[start*step:i*step], Y[start*step:i*step]))
        start=i
        i+=1
    cur = 0
    errors=[]
    errors_test=[]
    while cur<k:
        test = folds[cur]
        trainX = []
        trainY = []
        for fold in folds:
            if fold!=test:
                for x in fold[0]:
                    trainX.append(x)
                for y in fold[1]:
                    trainY.append(y)
        XT = degreeMatrix(test[0], degree)
        XTT = degreeMatrix(trainX, degree)
        YT = test[1]       
        W = linregress(trainX,trainY,degree)
        predictions = np.matmul(W.T, XT)
        mse = 0
        for i in range(len(predictions)):
            mse+=(predictions[i]-YT[i])**2
        errors.append(mse/len(predictions))
        
        predictions = np.matmul(W.T, XTT)
        mse = 0
        for i in range(len(predictions)):
            mse+=(predictions[i]-trainY[i])**2
        errors_test.append(mse/len(predictions))
        cur+=1
        
    return (math.sqrt(sum(errors)/len(errors)), math.sqrt(sum(errors_test)/len(errors_test)))

def findPolyDegree(X, Y, top):
    rmse = []
    degrees = []
    errors = []
    errors_t = []
    for i in range(top):
        degrees.append(i+1)
    for cur in range(len(degrees)):
        dataset.plot(x=12, y=13, style='o')
        err = cross_validation(X, Y, degrees[cur])
        errors.append(err[0])
        errors_t.append(err[1])
        rmse.append((err[0], err[1], degrees[cur]))
        stg = "data points"
        plt.legend([stg])
        plt.title(degrees[cur])
        plt.xlabel('LSTAT')
        plt.ylabel('MEDV')
        plt.show()
    plt.axis()
    plt.xlabel("Degree")
    plt.ylabel("Error value")
    plt.title("Error-degree Graph")
    plt.plot(degrees, errors, label="Testing Error")
    plt.plot(degrees, errors_t, label="Training Error")
    plt.legend()
    plt.show()
    print("RMSE Errors - Testing, Training, Degree")
    for err in rmse:
        print(err)
    print("Best Polynomial Degree Fit:", min(rmse)[2])
    return min(rmse)[2]

def multiregress(X, Y):
    if len(X.shape)==1:
        X = X.reshape(-1,1)
    ones = np.ones(shape=X.shape[0]).reshape(-1,1)
    X = np.concatenate((ones,X),1)
    XT = X.transpose()
    XTXI = np.linalg.inv(XT.dot(X))
    W = XTXI.dot(XT).dot(Y)
    print("Weights learned:\n", W)
    return W

def predict(W, X, Y, st, start):
    temp = W[0]
    W = W[1:]
    predictions = (W.T).dot(X.T)
    predictions = predictions.T
    predictions = [pred+temp for pred in predictions]
    err = 0
    for i in range(len(predictions)):
        err += (predictions[i] - Y.at[start+i, 13])**2
    rmse = math.sqrt(err/len(predictions))
    print("RMSE for",st,rmse)

path = "Dataset.data"
data = []
with open(path) as f:
    str = f.read().split('\n')
    for i in range(len(str)):
        d = str[i].split()
        d_ = []
        for num in d:
            d_.append(float(num))
        data.append(d_)
dataset = pd.DataFrame(data)
#print(dataset)

# part a
spos_train = 0
spos_test = 405
size = int(dataset.shape[0]*0.8)
train = dataset.loc[0:size]
test = dataset.loc[size+1:]

# part b
X = train.iloc[:,:13]
Y = train.iloc[:,13:]

W = multiregress(X, Y)

XT = test.iloc[:,:13]
YT = test.iloc[:,13:]


predict(W, XT, YT, "test", spos_test)
predict(W, X, Y, "train", spos_train)

# part c
X = train.iloc[:,12:13]
Y = train.iloc[:,13:]
XT = test.iloc[:,12:13]
YT = test.iloc[:,13:]

    
X = dftolis(X, spos_test-1, 12, 0)
Y = dftolis(Y, len(X), 13, 0)

XT = dftolis(XT, 505-405, 12, spos_test)
YT = dftolis(YT, 505-405, 13, spos_test) 

best = findPolyDegree(X, Y, top=10)

# part g
best = 6
W = linregress(X, Y, best)
XT = degreeMatrix(XT, best)
predictions = np.matmul(W, XT)
errors = []
mse = 0
for i in range(len(predictions)):
    mse+=(predictions[i]-YT[i])**2
errors.append(mse/len(predictions))
rmse = math.sqrt(sum(errors)/len(errors))
print("RMSE of test dataset: ", rmse)


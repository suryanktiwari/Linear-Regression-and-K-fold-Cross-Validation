import numpy as np
import matplotlib.pyplot as plt
import math
import string

path = "data.csv"

x = 0
y = 1
file  = open(path)
dat = file.read().split()[1:]
data = []
for tup in dat:
    tup=tup.split(',')
    data.append((float(tup[x]), float(tup[y])))

def f(X, W):
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
    #print(W)
    X = np.linspace(-5, 5, 200, endpoint=True)
    F = f(X, W)
    plt.plot(X,F)
    plt.title(M)
    plt.xlabel('X')
    plt.ylabel('Y')
    return W

def showData(X, Y):
    plt.plot(X, Y, 'ro', label="data points")
    plt.axis([-5, 5, -2.5, 2.5])
    plt.legend(loc='upper center',
          ncol=3, fancybox=True, shadow=True)
    plt.show()


X = [num[x] for num in data]
Y = [num[y] for num in data]

# =============================================================================
# W = linregress(X,Y,7)
# print("Weights Learned:",W)
# =============================================================================

# DEMONSTRATION OF ERROR DECREASE, OVERFITTING
# =============================================================================
# j=1
# while j<20:
#     W = linregress(X,Y,j)
#     predictions = np.matmul(W.T, degreeMatrix(X,j))
#     mse = 0
#     for i in range(len(predictions)):
#         mse+=(predictions[i]-Y[i])**2
#     j+=1
#     print(j, mse)    
# 
# =============================================================================

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
        YT = test[1]       
        W = linregress(trainX,trainY,degree)
        predictions = np.matmul(W.T, XT)
        mse = 0
        for i in range(len(predictions)):
            mse+=(predictions[i]-YT[i])**2
        errors.append(mse/len(predictions))
        cur+=1
    return math.sqrt(sum(errors)/len(errors))

def findPolyDegree(X, Y, top=-1):
    rmse = []
    degrees = []
    if top!=-1:
        for i in range(top):
            degrees.append(i+1)
    else:
        degrees = [1, 2, 4, 5, 10, 15, 30]                    
    for cur in range(len(degrees)):
        err = cross_validation(X, Y, degrees[cur])
        rmse.append((err, degrees[cur]))
        showData(X, Y)
        #stg = "Polynomial Degree: "+str(degrees[cur])
        #plt.legend([stg])
    for err in rmse:
        print(err)
    print("Best Polynomial Degree Fit:", min(rmse)[1])
    
    
findPolyDegree(X, Y, top=27)
findPolyDegree(X, Y)    

from matplotlib import pyplot as plt
import numpy as np
import time

def costLinearRegression(X, y, theta):
    m = len(y)
    J = np.dot(X,theta)
    J = np.subtract(J, y)
    J = np.power(J,2)
    J = np.sum(J)
    J = J/(2*m)
    return J

def decent(X, y, theta, alphas, iteration):
    #cost = [] #save cost every iteration
    feature = len(X[0]) #number of feature
    temp = np.zeros(feature) #save new theta temporary
    m = len(y) #number of dataset
    for i in range(iteration):
        #if you want to use diferent alpha value in the same iteration
        '''if i < (iteration/20):
            alpha = alphas[0]
        elif i < (iteration/5):
            alpha = alphas[1]
        else : alpha = alphas[2]
        '''
        plt.clf
        for n in range(feature):
            J = np.dot(X,theta)
            J = np.subtract(J,y)
            J = np.vdot(J, X[:,n])
            J = (alpha * (1/m) * J)
            temp[n] = theta[n] - J
        for n in range(feature):
            theta[n] = temp[n]
        
        ##for graph
        if i%5 == 0:
            A = np.linspace(0,4)
            B = theta[0]+theta[1]*A + theta[2]*A*A 
            plt.scatter(X[:,1],y)
            plt.plot(A,B)
            plt.draw()
            plt.pause(0.5)
            plt.clf()

        ##for cost
        #Cost = costLinearRegression(X,y, theta)
        #cost.append(Cost)
    #A = np.array(range(len(cost)))
    #plt.plot(A,cost)
    return theta      
        
X0 = np.ones(5)
X1 = np.array(range(5))
X2 = X1*X1

X = np.stack((X0, X1, X2), axis=1)
y = np.array(range(-1,8,2))
y = (y*y) - y

#param
alpha = 0.01
iteration = 600
theta = [1,1,1]

A = np.linspace(0,4)
B = theta[0]+theta[1]*A + theta[2]*A*A 
plt.plot(A,B)
plt.draw()
plt.pause(10)
plt.clf()

theta = decent(X,y, theta, alpha, iteration)
print(theta)

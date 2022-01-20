from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def cost(prediction, y):
    return np.sum(((y)*((-1)*np.log(prediction)))+((1-y)*((-1)*np.log(1-prediction))))

def createData(type):
    if type=='simpleRandomBinary':
        numberofData = 100
        bias = np.ones(numberofData*2)
        x = np.append(np.random.uniform(-14,-9.5, size=numberofData),np.random.uniform(-10.5,-6, size=numberofData))
        x = np.transpose(np.array([bias,x]))
        y = np.append(np.ones(numberofData),np.zeros(numberofData))
        return x, y
    if type=='2dRandomBinary':
        df = pd.read_excel('e:/Python/Visualize Regression/LogisticRegression/binary2d.xlsx')
        npdf = np.array(df)
        npdf0 = np.where(npdf==0)
        npdf1 = np.where(npdf==1)
        a0 = npdf0[0]
        b0 = npdf0[1]
        a1 = npdf1[0]
        b1 = npdf1[1]
        x1 = np.append(a0,a1)
        x2 = np.append(b0,b1)
        y = np.append(np.zeros(391),np.ones(69))
        index = np.array([x1,x2,y])
        index = np.transpose(index)
        return index

def plot(theta, duration):
    x = np.transpose(np.array([np.ones(1000),np.linspace(-14,-6,1000)]))
    z = np.dot(x,theta)
    y = 1/(1+np.exp(-z))
    plt.scatter(x[:,1], y, c = y)
    plt.draw()
    plt.pause(duration)
    plt.clf()

def plot2d(index, theta, duration):
    cplot = index[:,2]*(1/1.5)
    cplot[0] = 1
    plt.scatter(index[:,0], index[:,1], c = cplot)
    x = index[:,0:2]
    x = np.transpose([np.ones(460), x[:,0], x[:,1], x[:,0]*x[:,1], x[:,0]*x[:,0], x[:,1]*x[:,1]])
    z = np.dot(x,theta)
    z = -z
    z = np.exp(z)
    prediction = 1/(1+z)
    for i in range(len(prediction)):
        if prediction[i]>0.5:
            prediction[i]=1
        else: prediction[i]=0
    plt.scatter(index[:,0]+0.5, index[:,1]+0.5, c = prediction)
    plt.draw()
    plt.pause(duration)
    plt.clf()

def decent(x,y,theta,alpha,iteration):
    costs = [] #save cost every iteration
    m = len(y) #number of dataset
    for i in range(iteration):
        if i%100==0 and i<2000:
            print(i,theta)
            plt.scatter(x[:,1], y, c = y)
            plot(theta, 0.5)
        elif i%250==0 and i<4000:
            print(i,theta)
            plt.scatter(x[:,1], y, c = y)
            plot(theta, 0.5)
        elif i%500==0:
            print(i,theta)
            plt.scatter(x[:,1], y, c = y)
            plot(theta, 0.5)

        z = np.dot(x,theta)
        prediction = 1/(1+np.exp(-z))
        error = prediction-y
        result = (alpha * (1/m)) * np.dot(np.transpose(x),error)
        theta = theta - result
        #costs.append(cost(prediction,y))
        costs.append(np.sum(np.abs(error)))
    return theta, costs

def decent2d(index,theta,alpha,iteration):
    x = index[:,0:2]
    x = np.transpose([np.ones(460), x[:,0], x[:,1], x[:,0]*x[:,1], x[:,0]*x[:,0], x[:,1]*x[:,1]])
    y = index[:,2]
    costs = [] #save cost every iteration
    m = len(y) #number of dataset
    for i in range(iteration):
        if i%1000==0 and i<10000:
            print(i,theta)
            plot2d(index, theta, 0.5)
        elif i%10000==0 and i<250000:
            print(i,theta)
            plot2d(index, theta, 0.5)
        elif i%25000==0:
            print(i,theta)
            plot2d(index, theta, 0.5)

        z = np.dot(x,theta)
        prediction = 1/(1+np.exp(-z))
        error = prediction-y
        result = (alpha * (1/m)) * np.dot(np.transpose(x),error)
        theta = theta - result
        costs.append(np.sum(np.abs(error)))
    return theta, costs

def run1():
    #generate data
    x, y = createData('simpleRandomBinary')
    plt.scatter(x[:,1], y, c = y)

    #plot random theta
    plot([36,4],10)

    #parameter
    iteration = 10000
    theta = [36,4]
    alpha = 0.4

    #learning
    theta, costs = decent(x,y,theta,alpha,iteration)

    #plot
    plt.scatter(x[:,1], y, c = y)
    plot(theta,5)

    #plot cost
    plt.plot(np.array(range(len(costs))),costs)
    plt.draw()
    plt.pause(5)
    plt.clf()

def run2():
    #generate data
    index = createData('2dRandomBinary')
    plt.scatter(index[:,0], index[:,1], c = index[:,2])
    plt.draw()
    plt.pause(3)
    plt.clf()
    
    #plot random theta
    theta = [0,0,0,0,0,0]   
    plot2d(index, [0,0,0,0,0,0], 3)
    
    #parameter
    iteration = 1000000
    theta = [0,0,0,0,0,0] #[x1,x2,x1x2,x1^2,x2^2]
    alpha = 0.0002
    
    #learning
    theta, costs = decent2d(index,theta,alpha,iteration)

    #plot -3.74856866  0.38929995  0.72769585  0.11697077 -0.08574239 -0.09251312
    plot2d(index, theta, 5)

    #plot cost
    plt.plot(np.array(range(len(costs))),costs)
    plt.draw()
    plt.pause(5)
    plt.clf()

#run1()
run2()
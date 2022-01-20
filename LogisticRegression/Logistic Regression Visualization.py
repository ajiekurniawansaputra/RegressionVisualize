from matplotlib import pyplot as plt
import numpy as np

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

def plot(theta, duration):
    x = np.transpose(np.array([np.ones(1000),np.linspace(-14,-6,1000)]))
    z = np.dot(x,theta)
    y = 1/(1+np.exp(-z))
    plt.scatter(x[:,1], y, c = y)
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
        costs.append(cost(prediction,y))
    return theta, costs


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
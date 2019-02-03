import numpy as np

def initializeValue(arr):
    np.random.seed(1)
    arr = 2 * np.random.random((arr.shape)) - 1


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoidDerivative(x):
    return x*(1-x)

def getError(error):
    avg=0.0
    for i in range(len(error)):
        avg+=error[i][0]*error[i][0]
    return avg/len(error)


def training(inputs,output,weight,row,column):
    errorTolerance=0.00000001
    iteration=1000000
    learningRate=0.1
    trans=inputs.T 
    
    for i in  range(iteration):
        sum=np.dot(inputs,weight)
        predictedOutput=sigmoid(sum)

        error=output-predictedOutput

        #update=np.dot(inputs.T,error*sigmoidDerivative(predictedOutput))

        for i in range(len(trans)):
            weight[i][0] +=np.dot(trans[i],error*sigmoidDerivative(predictedOutput))

        avgSqrdError=getError(error)

        if(avgSqrdError <= errorTolerance):
            break

    return weight


def getResult(inputs,weight):
    sum=np.dot(inputs,weight)
    return sigmoid(sum)


if __name__=="__main__":
    row = int(input("give number of data set:  "))
    column = int(input("give number of input in each data set:  "))

    #print(row)
    inputSet = np.arange(row * column,dtype=float).reshape(row, column)
    print("give input data set : ")

    for i in range(row):
        #for j in range(column):
        inputSet[i] = [float(a) for a in input().split()]

    inputSet=np.insert(inputSet,0,1,axis=1)
    print(inputSet.T)
    #print(row)

    andOutput = np.arange(row, dtype=float).reshape(row, 1)
    #print(andOutput.shape)
    #print(andOutput)

    print("give and gate output data set: ")

    for i in range(row):
        andOutput[i][0] = float(input())

    orOutput = np.arange(row, dtype=float).reshape(row, 1)
    print("give or gate output data set: ")

    for i in range(row):
        orOutput[i][0] = float(input())

    andWeight = np.arange(column+1,dtype=float).reshape(column+1, 1)
    orWeight = np.arange(column+1,dtype=float).reshape(column+1, 1)
   # bias_a = 1.0
   # bias_o = 1.0

    initializeValue(andWeight)
    initializeValue(orWeight)

    print('wait.....')

    andWeight=training(inputSet, andOutput, andWeight, row, column)
    orWeight=training(inputSet, orOutput, orWeight, row, column )


    while 1 :
        print('give an input set to test output : ')
        inputTest = np.arange( column, dtype=float).reshape(1,column)

        for i in range(column):
            inputTest[0][i]=float(input())

        inputTest=np.insert(inputTest,0,1,axis=1)
        
        print('and operation result : ')
        result=getResult(inputTest,andWeight)
        print(result)

        print('or operation result : ')
        result2 = getResult(inputTest, orWeight)
        print(result2)

        print('\npress q to exit and any button to continue : ')
        c=input()
        if c=='q':
            break


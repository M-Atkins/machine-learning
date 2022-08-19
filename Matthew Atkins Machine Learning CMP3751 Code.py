#Matthew Atkins Machine Learning CMP3751M-Machine Learning - 16657290 - University of Lincoln, School of Computer Science

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from sklearn.model_selection import train_test_split

#Pandas to read csv
data_train = pd.read_csv('Task1 - dataset - pol_regression.csv')
#Convert to 1D numpy arrays
x_train = data_train['x']
y_train = data_train['y']

#Implementing Polynomial Regression
#Convert to Matrix
def getPolyMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree +1):
        X = np.column_stack((X,x**i))
    return X

#Get polynomials to a specified degree
def pol_regression(x,y,degree):
    if (degree == 0):
        degreemean = 0
        degreelength = len(y_train)
        for j in range(degreelength):
            degreemean = y_train[j]
            w = (degreemean/degreelength)
    #If weight is 0.
    else:
        X = getPolyMatrix(x, degree)
        XX = X.transpose().dot(X)
        w = np.linalg.solve(XX, X.transpose().dot(y))
    return w


#Create figure
plt.figure() 

#Plot 'raw' data as blue dots
plt.plot(x_train,y_train, 'bo')

#Generate evenly spaced numbers over a specified interval 
line = np.linspace(-5,5,20,0)

#Loops through degrees 0-10, calculates and plots 
# the line of best fit for each degree, plots on the same graph.
for x in range(11):
    w = pol_regression(x_train,y_train,x)
    xt = getPolyMatrix(line,x)
    yt = xt.dot(w)
    plt.title("Polynomials of degres 0-10")
    plt.plot(line,yt, label=x)
    plt.legend()
    
#Lims as requested from the brief
#for better viewing of graph
plt.xlim(-5,5)
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(-200,200) 
plt.show

#Polynomial Evaluation
#Evaluate Polynomials and calculate the RMSE
def eval_pol_regression(parameters,x,y,degrees):
    #Get coefficients
    coeff = getPolyMatrix(x,i)
    #Get weights (betas)
    w = pol_regression(x,y,i)
    #Calculate the RMSE, Square root performed (.sqrt) on the MSE (Mean Squared Error)
    rmse = np.sqrt(np.mean((coeff.dot(w)-y)**2))
    return rmse

#Split data set into 30% test and 70% train
#disabled shuffle for reproducable results
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, train_size=0.7,shuffle=False)


#Initialize rmseTrain and rmseTest Arrays to store rmse values of both sets
rmseTrain = []
rmseTest = []
#Array of degrees to pass in accordance with brief
degrees = [0,1,2,3,5,6,10]
#For each degree in the array, evaluate the Betas of each degree, and append to the respective array for later usage
for i in degrees:
    rmseTrain.append(eval_pol_regression(pol_regression(X_train,Y_train,i),X_train,Y_train,i))
    rmseTest.append(eval_pol_regression(pol_regression(X_test,Y_test,i),X_test,Y_test,i))

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
    
plt.figure()
plt.title("RMSE of Train and Test set")
plt.xlabel('Root Means Squared Error')
plt.ylabel('Polynomial Degree')

plt.plot(rmseTrain,degrees, linestyle='--', marker='o',label='RMSE Test')
plt.plot(rmseTest,degrees, linestyle='--', marker='o',label='RMSE Train')
plt.legend()
plt.show()

#Task 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from sklearn.cluster import KMeans

def compute_euclidean_distance(vec_1, vec_2):
    #The function compute_euclidean_distance() calculates the distance of two vectors, e.g., Euclidean distance
    #returns the norm of the difference of both vectors to calculate euclidian distance
    distance = np.linalg.norm(vec_1-vec_2,axis=1)
    return distance

def initialise_centroids(dataset, k):
    #Initialises a centroid in a random location of the cluster range using randint
    centroids = dataset[np.random.randint(dataset.shape[0],size=k)]
    return centroids

def kmeans(dataset, k):
    #The function kmeans() clusters the data into k groups
    centroids = initialise_centroids(dataset,k)
    #Max iterations 50
    maxiter = 50
    
    #Initialises two matrices using the dataset and converts them to float
    #Same oepration for the k value
    classes = np.zeros(dataset.shape[0],dtype=np.float64)
    distances = np.zeros([dataset.shape[0],k], dtype=np.float64)
    
    #Pass through until reaches maxiter
    for i in range(maxiter):
        #Loop for number of centroids
        for i, c in enumerate(centroids):
            #Get distances with helper function
            distances[:,i] = compute_euclidean_distance(c,data)
            #Find smallest value within distances
            classes = np.argmin(distances,axis=1)
            #get the means of the classes and assign them an array of centroids
            for c in range(k):
                centroids[c] = np.mean(dataset[classes ==c],0)
            clusters = classes

    return centroids, clusters

#returns the appropriate number of color arguments for K no. of clusters
#Checks the k value and populates the appropriate number of colors for centroids respectively.
def wrapColors(k):
    ctColors =[]
    cTemp = ['black','green','red']
    for i in range(k):
        ctColors.append(cTemp[i])
    return ctColors

#Plots the clusters
def plotclusters(data,k):
    #gets appropriate color quantity
    ctColors = wrapColors(k)
    
    #Calculates Kmeans of data and k value
    centroids, classes = kmeans(data,k)
    #Gets title for k, used for plotting layer
    title = str(k)
    
    #Colors for plotting
    gcolors = ['skyblue','coral','lightgreen']
    colors = [gcolors[j] for j in classes]
    
    #Draws plot for k value on both axis
    plt.figure()
    plt.scatter(data[:,0],data[:,1], color=colors,alpha=0.5)
    #Uses random colors
    plt.scatter(centroids[:,0], centroids[:,1], color = ctColors.copy(), marker='o', lw=2)
    plt.xlabel('Height')
    plt.ylabel('Tail Length')  
    plt.title("K = "+title)
    
    #Repeated for other axis
    plt.figure()
    plt.scatter(data[:,0],data[:,2], color=colors,alpha=0.5)
    
    plt.scatter(centroids[:,0], centroids[:,2], color = ctColors.copy(), marker='o', lw=2)
    plt.xlabel('Height')
    plt.ylabel('Leg Length')  
    plt.title("K = "+title)

    return

#Get data set
data = pd.read_csv('Task2 - dataset - dog_breeds.csv').values

plotclusters(data,3)
plotclusters(data,2)
#Finally, a line plotx axis is iteration step, y axis is the objective functions value
##TODO

#Task 3.1
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Read Dataset
df = pd.read_csv('Task3 - dataset - HIV RVG.csv')

#Declare Summarization operations
ops = ["mean","median","std","max","min"]
#Get column names
columnNames = list(df.columns)

#Summarize using DataFrame.Aggregate with the ops argument to only display desired calculations
summ=df.agg(ops)
display(summ)

#Create a boxplot using 'Condition' as the x value and Alpha as the Y value
s = df.groupby(['Participant Condition'])
s.boxplot(column=['Alpha'],subplots=False)

#Split DataFrames based on Participant Condition
df1 = df[df['Participant Condition'] == "Patient"]
df2 = df[df['Participant Condition'] == "Control"]

#Plot Density Plot using the Betas of 'Patient' and 'Control' Participant conditions using Seaborn (Shading operator makes it easier to compare)
plt.figure()
sns.kdeplot(df1['Beta'], label='Patient',shade=True)
sns.kdeplot(df2['Beta'], label='Control',shade=True)
plt.legend()

#Task 3.2

#Import libraries
from sklearn import neural_network

#Split data
#sigmoid function as the non-linear activation function for the hidden layers and logistic
#function for the output layer; 
def ANN():
    #Load Dataset
    df = pd.read_csv('Task3 - dataset - HIV RVG.csv')
    #Drop unnecessary columns
    df = df.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'],axis=1)
    #Drop Condition for use as label
    x = df.drop('Participant Condition',1)
    #Create label as Participant Condition
    y = df['Participant Condition']
    
    #Split data, add shuffle to randomize results on each pass
    trainx,testx, trainy,testy = train_test_split(x,y, train_size=0.9, shuffle=True)

    clf = neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'sgd',
                                  max_iter = 80000, hidden_layer_sizes = (500, 500),
                                  activation = 'logistic',
                                  momentum = 0, alpha = 0,
                                  verbose = False, tol = 1e-10,
                                  random_state = 11)
    #Fit training data against Participant Condition
    clf = clf.fit(trainx, trainy)
    #Train against the testing set
    y_pred=clf.predict(testx)

    #Import libraries
    from sklearn import metrics
    
    # print("Accuracy:",metrics.accuracy_score(testy, pr))
    #Calculate accuracy using sklearn.
    accr = metrics.accuracy_score(testy, y_pred)
    return accr

#for different epochs, run in a for loop, append to a list.
plt.figure()
epochs=[]
for i in range(1,5):
    accr = ANN()
    epochs.append(accr)
    
# for i in range(len(epochs)):
#     print(epochs[i])

epochn=[]
for i in epochs:
#     print(i)
    epochn.append((epochs.index(i)+1))

#Plot figure
plt.plot(epochn,epochs)
plt.title("Accuracy against number of epochs")
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.plot(epochn,epochs, linestyle='--', marker='o',label='ANN Accuracy')
plt.legend()
plt.show()
#Plot results

from sklearn.ensemble import RandomForestClassifier
#Read in the dataset
df = pd.read_csv('Task3 - dataset - HIV RVG.csv')
#Prepare dataset
df = df.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'],axis=1)
x = df.drop('Participant Condition',1)
y = df['Participant Condition']
#Split data set
trainx,testx, trainy,testy = train_test_split(x,y, train_size=0.9)

#
def RandomTrees(a,e,trainx,trainy):
    #e = estimators, a = minimum leaf sample
    clf=RandomForestClassifier(n_estimators=e, min_samples_leaf = a)
    #Fit the model
    clf.fit(trainx,trainy)
    #Train against the test set
    y_pred=clf.predict(testx)

    from sklearn import metrics
    print("Accuracy \nEstimators:",e,"\nMin samples:",a,"\n=",metrics.accuracy_score(testy, y_pred))
    return

samples = [5,10]
for i in samples:
    RandomTrees(i,1000,trainx,trainy)

#Task 3.3
#Function that uses an ANN, Sigmoid and Logistic functions to classify based on user-inputs (varying estimates 50,500,10000)
def ANN_eval(n,trainx,testx,trainy,testy):
#     trainx,testx, trainy,testy = train_test_split(x,y, train_size=0.9)
    clf = neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'sgd',
                                  max_iter = 80000, hidden_layer_sizes = (n, n),
                                  activation = 'logistic',
                                  momentum = 0, alpha = 0,
                                  # tol is tolerance -- needs to be this low as improvements are so small
                                  verbose = False, tol = 1e-10,
                                  random_state = 11)
    return clf

#Function that uses a random forests classifier to train data based on input estimators and minimum samples (50,500,1000)
def RandomForest_eval(e,a,trainx,trainy):
    clf=RandomForestClassifier(n_estimators=e, min_samples_leaf = a)
    return clf


#Load Libraries for K-Fold, CV Eval
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Prepare Dataset
df = pd.read_csv('Task3 - dataset - HIV RVG.csv')
df = df.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'],axis=1)
x = df.drop('Participant Condition',1)
y = df['Participant Condition']
trainx,testx, trainy,testy = train_test_split(x,y, train_size=0.7)

#Split into 10 equal K-Folds
kf = KFold(n_splits=10)

#Estimates
ests = [50,500,1000]

#Calculate 50,500,10000 estimates for the ANN
print("Artificial Neural Net: ")
for i in ests:
    clf = ANN_eval(i,trainx,testx,trainy,testy)
    #Print Score
    print("Estimates: ",i)
    print("Accuracy: ",cross_val_score(clf, x, y, cv = kf).mean())

#Calculate 50,500,10000 estimates for 10 min leaf samples
print("Random Forests: ")
for i in ests:
    clf = RandomForest_eval(i,10,trainx,trainy)
    #Print Score
    print("Forests: ",i,", Min Leaf Samples: 10")
    print("Accuracy: ",cross_val_score(clf, x, y, cv = kf).mean())

    
# clf = ANN_eval(50,trainx,testx,trainy,testy)
# print(cross_val_score(clf, x, y, cv = kf).mean())
    


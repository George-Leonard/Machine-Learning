from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import svm
import numpy as np
import pandas as pd

iris = load_iris()

#4.5,3,2,0.5


clfKNN = KNeighborsClassifier(n_neighbors=1) # using the k nearest neightbours clf
clfSVC = svm.SVC()  # using svc clf
clfETR = ensemble.ExtraTreesRegressor(n_estimators=3)  # using ensemble extra trees clf

#  print(iris.data)  # Our data

#  print(iris.target)  # our target values
#  print(iris.target_names)  # the names of our target values (e.g 0 = setosa)

#  print(type(iris.data))  # Formatted as np array
#  print(type(iris.target))  # Formatted as np array

#  print(iris.data.shape)  # there are 150 observations with 4 features (data/X)
#  print(iris.target.shape)  # there are 150 observation values (targets/Y)

X = iris.data
y = iris.target

#TRAIN DATA - halving each segment, as data is split up into 3x 50s, each of unique labels
#i.e. half of setosa segment, half of versicolor segment, half of virginica segment
Xtr = X[np.r_[0:25, 50:75, 100:125]] #X is capitalized as it represents a matrix
ytr = y[np.r_[0:25, 50:75, 100:125]] # y is lowercase as it represents a vector

#TEST DATA - opposing data
Xte = X[np.r_[25:50, 75:100, 125:150]]
yte = y[np.r_[25:50, 75:100, 125:150]]

X_predict = np.reshape([3, 5, 4, 2], (1, 4))  # giving some new measurements as an input, and reshaping
# them to the array size we want (1 outcome, with 4 features)

clfKNN.fit(Xtr,ytr)
clfSVC.fit(Xtr,ytr)
clfETR.fit(Xtr,ytr)

#  0 = setosa 1 = versicolor 2 = virginica

def inttostr(pred): # function returning type of iris given an int prediction input

    if pred == 0:
        return "Prediction: setosa"
    if pred == 1:
        return "Prediction: versicolor"
    if pred == 2:
        return "Prediction: virginica"
    
predsKNN=[]
predsSVC=[]
predsETR=[]

for i in range(0,len(Xte)): #create lists of predicted values for the x test data
    X = list(Xte[i])
    predsKNN.append(int(clfKNN.predict([X])))
    predsSVC.append(int(clfSVC.predict([X])))
    predsETR.append(int(clfETR.predict([X])))

correctKNN = 0
for i in range(0,len(predsKNN)): #compare the just-produced lists of predicted values with the actual values...                        
    if predsKNN[i] == yte[i]: #...in the y test data for each classifier type
        correctKNN+=1
    else:
        correctKNN+=0
correctSVC = 0
for i in range(0,len(predsSVC)):
    if predsSVC[i] == yte[i]:
        correctSVC+=1
    else:
        correctSVC+=0
correctETR = 0
for i in range(0,len(predsETR)):
    if predsETR[i] == yte[i]:
        correctETR+=1
    else:
        correctETR+=0

print('--Accuracies--') #value between 0 and 1, the number of correct values over...
print('KNN:',correctKNN/len(predsKNN)) #...the number of values in the prediction array
print('SVC:',correctSVC/len(predsSVC))
print('ETR:',correctETR/len(predsETR))

clfchoice = int(input('Use KNN(1)/SVC(2)/ETR(3)?: ')) #user can choose classification method to choose to predict newly inputted iris measurements
predictInput = input('Predict (in form sepalLength, sepalWidth, petalLength, petalWidth): ')
predictInput = predictInput.split(',') #turn input into list

for i in range(0,len(predictInput)):
    predictInput[i] = float(predictInput[i]) #strings to floats
    
predictInput = np.array(predictInput)
predictInput = predictInput.reshape(1,-1)

#referencing the decision they made for which classifier to use:
if clfchoice == 1:
    clf = clfKNN.predict((predictInput))
    print(clf, inttostr(clf))

elif clfchoice == 2:
    clf = clfSVC.predict((predictInput))
    print(clf, inttostr(clf))

elif clfchoice == 3:
    clf = clfETR.predict((predictInput))
    print(clf, inttostr(clf))

data = list(iris.data)
array = []
for i in range(0, len(data)):
    array.append(list(data[i]))

df = pd.DataFrame(array) #pandas dataframe of the data

attr = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
df.columns = attr #rename columns for easier referencing

fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(projection='3d') #3d graph for sepal length and width and type
ax2 = fig2.add_subplot(projection='3d')

#each of the 3 types is in a segment of 50 out of each 150:
ax1.scatter(xs=df['Sepal length'][0:50],ys=df['Sepal width'][0:50], zs=y[0:50])
ax1.scatter(xs=df['Sepal length'][50:100],ys=df['Sepal width'][50:100], zs=y[50:100])
ax1.scatter(xs=df['Sepal length'][100:150],ys=df['Sepal width'][100:150], zs=y[100:150])

ax1.set_xlabel('Sepal length')
ax1.set_ylabel('Sepal width')
ax1.set_zlabel('Type')

#3d graph for petal length and width and type
ax2.scatter(xs=df['Petal length'][0:50],ys=df['Petal width'][0:50], zs=y[0:50])
ax2.scatter(xs=df['Petal length'][50:100],ys=df['Petal width'][50:100], zs=y[50:100])
ax2.scatter(xs=df['Petal length'][100:150],ys=df['Petal width'][100:150], zs=y[100:150])

ax2.set_xlabel('Petal length')
ax2.set_ylabel('Petal width')
ax2.set_zlabel('Type')


ax1.scatter(predictInput[0][0],predictInput[0][1],clf)
ax2.scatter(predictInput[0][2],predictInput[0][3],clf)

#annotate the inputted iris, and allow the user to see the justification of why the algorithm predicted the features to be of a particular species
ax1.text(x=predictInput[0][0],y=predictInput[0][1],z=clf,s='Prediction')
ax2.text(x=predictInput[0][2],y=predictInput[0][3],z=clf,s='Prediction')
plt.show()




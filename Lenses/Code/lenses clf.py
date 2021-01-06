from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import svm

def column(matrix, i):
    return [row[i] for row in matrix]
def tostring(num):
    if num == 1: #hard contact lenses
        return 'Prediction:  the patient should be fitted with hard contact lenses.'
    if num == 2: #soft contact lenses
        return 'Prediction: the patient should be fitted with soft contact lenses.'
    if num == 3: #no contact lenses
        return 'Prediction: the patient should not be fitted with contact lenses.'
        
data = pd.read_fwf('lenses.txt')
data = data.values
ncol = data.shape[1]
y = column(data, -1)
ytr=y[0:12]
yte=y[12:23]

data = np.asmatrix(data)
X = data[:,0:4]
Xtr = X[0:12]
Xte = X[12:23]

clfKNN = KNeighborsClassifier(n_neighbors=3)
clfETR = ensemble.ExtraTreesRegressor(n_estimators=3)
clfSVC = svm.SVC()
clfKNN.fit(Xtr,ytr)
clfETR.fit(Xtr,ytr)
clfSVC.fit(Xtr,ytr)

predsKNN=[]
predsSVC=[]
predsETR=[]
for i in range(0,len(Xte)): #create lists of predicted values for the x test data
    X = Xte[i]
    predsKNN.append(int(clfKNN.predict(X)))
    predsSVC.append(int(clfSVC.predict(X)))
    predsETR.append(int(clfETR.predict(X)))

correctKNN=0
correctSVC=0
correctETR=0

for i in range(0,len(predsKNN)):
    if predsKNN[i] == yte[i]:
        correctKNN+=1
    else:
        correctKNN+=0
for i in range(0,len(predsSVC)):
    if predsSVC[i] == yte[i]:
        correctSVC+=1
    else:
        correctSVC+=0
for i in range(0,len(predsETR)):
    if predsETR[i] == yte[i]:
        correctETR+=1
    else:
        correctETR+=0

print('--Accuracies--') #the accuracies will be relatively low due to the lack of data (only trained on 12)
print('KNN:',correctKNN/len(predsKNN)) 
print('SVC:',correctSVC/len(predsSVC))
print('ETR:',correctETR/len(predsETR))
clfchoice = int(input('Use KNN(1)/SVC(2)/ETR(3)?: ')) #user can choose classification method to choose to predict newly inputted iris measurements
predictInput = input('Predict ([READ DOCS] in form age, spectacle-prescription, astigmatic, tear-production-rate): ')
predictInput = predictInput.split(',') #turn input into list

for i in range(0,len(predictInput)):
    predictInput[i] = float(predictInput[i]) #strings to floats
    
predictInput = np.array(predictInput)
predictInput = predictInput.reshape(1,-1)

if clfchoice == 1:
    clf = clfKNN.predict((predictInput))
    print(clf, tostring(clf))

elif clfchoice == 2:
    clf = clfSVC.predict((predictInput))
    print(clf, tostring(clf))

elif clfchoice == 3:
    clf = clfETR.predict((predictInput))
    print(clf, tostring(clf))

#plt.hist(xtr) #shows how much variance the training x data set has.
#plt.show()



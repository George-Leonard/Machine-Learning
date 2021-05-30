from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from processdb import process
import pandas as pd
from sklearn import ensemble
from sklearn import svm

dbtr = pd.read_csv("train.csv")
dbtr = process.pandasrenameheaders(dbtr, dbtr.loc[0])

dbtr.Sex = dbtr.Sex.replace("female", "1")
dbtr.Sex = dbtr.Sex.replace("male", "0")

dbtr = dbtr.dropna()

x = dbtr.drop(columns = ["Embarked", "Cabin", "Fare", "Ticket", "Name", "PassengerId","Survived"])
x = x.astype(float)

xtr = x[0:173]
xte = x[173:183]

y = dbtr["Survived"]
y = y.astype(float)
ytr = y[0:173]
yte = y[173:183] #tests only against 10 points, wanted to make sure the classifier
                 #had the most data to work with in order to increase accuracy

xtr.reset_index(drop=True)
ytr.reset_index(drop=True)
xte.reset_index(drop=True)
yte.reset_index(drop=True)

clfKNN = KNeighborsClassifier(n_neighbors=3)
clfETR = ensemble.ExtraTreesRegressor(n_estimators=3)
clfSVC = svm.SVC()
clfKNN.fit(xtr,ytr)
clfETR.fit(xtr,ytr)
clfSVC.fit(xtr,ytr)

predsKNN=[]
predsSVC=[]
predsETR=[]

for i in range(0,len(xte)): #create lists of predicted values for the x test data
    TMP = xte.iloc[i]
    TMP = np.array(TMP)
    TMP = TMP.reshape(1,-1)
    predsKNN.append(int(clfKNN.predict(TMP)))
    predsSVC.append(int(clfSVC.predict(TMP)))
    predsETR.append(int(clfETR.predict(TMP)))

correctKNN=0
correctSVC=0
correctETR=0

for i in range(0,len(predsKNN)):
    TMP = yte.iloc[i]
    TMP = np.array(TMP)
    TMP = TMP.reshape(1,-1)
    if predsKNN[i] == yte.iloc[i]:
        correctKNN+=1
    else:
        correctKNN+=0
for i in range(0,len(predsSVC)):
    TMP = yte.iloc[i]
    TMP = np.array(TMP)
    TMP = TMP.reshape(1,-1)
    if predsSVC[i] == yte.iloc[i]:
        correctSVC+=1
    else:
        correctSVC+=0
for i in range(0,len(predsETR)):
    TMP = yte.iloc[i]
    TMP = np.array(TMP)
    TMP = TMP.reshape(1,-1)
    if predsETR[i] == yte.iloc[i]:
        correctETR+=1
    else:
        correctETR+=0

print('--Accuracies--') #the accuracies will be relatively low due to the lack of data (only trained on 12)
print('KNN:',correctKNN/len(predsKNN)) 
print('SVC:',correctSVC/len(predsSVC))
print('ETR:',correctETR/len(predsETR))

clfchoice = int(input('Use KNN(1)/SVC(2)/ETR(3)?: ')) #user can choose classification method to choose to predict newly inputted iris measurements

val1 = float(input("Ticket class [1st class (1)-3rd class (3)]: "))
val2 = input("Sex [M/F]: ")
if val2 == "M":
    val2 = float(0)
if val2 == "F":
    val2 = float(1)
val3 = float(input("Age: "))
val4 = float(input("Number of siblings/spouses on board: "))
val5 = float(input("Number of parents/children on board: "))

predict = np.array([[val1,val2,val3,val4,val5]])

if clfchoice == 1:
    clf = clfKNN.predict((predict))
    
elif clfchoice == 2:
    clf = clfSVC.predict((predict))

elif clfchoice == 3:
    clf = clfETR.predict((predict))
    
if clf == 1:
    print("prediction: Alive")
else:
    print("prediction: Dead")

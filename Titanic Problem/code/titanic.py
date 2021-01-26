from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from processdb import process
import pandas as pd

db = pd.read_csv("train.csv")
db = process.pandasrenameheaders(db, db.loc[0])

db.Sex = db.Sex.replace("female", "1")
db.Sex = db.Sex.replace("male", "0")
db = db.dropna()

x = db.drop(columns = ["Embarked", "Cabin", "Fare", "Ticket", "Name", "PassengerId","Survived"])
x = x.astype(float)

y = db["Survived"]
y = y.astype(float)

print(x)
print(y)

clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(x,y)

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

pred = clf.predict(predict)

if pred == 1:
    print("Alive")
else:
    print("Dead")


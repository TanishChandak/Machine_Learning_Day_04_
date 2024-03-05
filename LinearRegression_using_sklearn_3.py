import pandas as pd

df = pd.read_csv('homeprices.csv')
print(df)

# pandas has the inbuilt method called (get_dummies)
# Here, we are using the (ON HOT ENCODING)
dummies = pd.get_dummies(df.town)
print(dummies)
# concating (dummies) into (df):
merge = pd.concat([df, dummies], axis='columns')
print(merge)
# we are droping the one-variable for better performance:
final = merge.drop(['town', 'west windsor'], axis='columns')
print(final)
# Appling linear regression:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)
# Droping the price column to
X = final.drop('price', axis='columns')
print(X)

y = final.price
print(y)

# Training the data:
model.fit(X, y)
print("This is the price of monroe_township: ",model.predict([[2800, 0, 1]]))
print("This is the price of west_windsor: ",model.predict([[2800, 0, 0]]))
print("This is the price of robbinsville: ",model.predict([[2800,1,0]]))
print(model.score(X, y)) # this will give the accuracy of the model

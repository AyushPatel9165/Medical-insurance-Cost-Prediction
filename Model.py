import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
 
data = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\Medical insurance Cost Prediction\medical_insurance.csv")

 

for col in data.select_dtypes(include=['int64','float64']).columns:
    data[col].fillna(data[col].mean(), inplace=True)

for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

 

data = pd.get_dummies(data, drop_first=True)

 

X = data.drop('charges', axis=1)
y = data['charges']

 

model = LinearRegression()
model.fit(X, y)


pickle.dump(model, open("insurance_model.pkl", "wb"))
pickle.dump(X.columns, open("insurance_columns.pkl", "wb"))

print("Model trained successfully!")
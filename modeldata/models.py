from django.db import models
import numpy as np
import pandas as pd
import joblib
#import seaborn as sns
#from sklearn.metrics import f1_score

df_train = pd.read_csv('modeldata/train_u6lujuX_CVtuZ9i.csv')
del df_train['Loan_ID']


#labal encoding
cat_cols = list(df_train.select_dtypes(include=['object']).columns)
num_cols = list(df_train.select_dtypes(exclude=['object']).columns)
for col in cat_cols:
    df_train[col] = df_train[col].astype('category')
    print(col,'---->', dict(enumerate(df_train[col].cat.categories)))
    df_train[col] = df_train[col].cat.codes

 #missing Value   
df_train['Gender']           = df_train['Gender'].fillna(df_train['Gender'].dropna().mode().values[0] )
df_train['Married']          = df_train['Married'].fillna(df_train['Married'].dropna().mode().values[0] )
df_train['Dependents']       = df_train['Dependents'].fillna(df_train['Dependents'].dropna().mode().values[0] )
df_train['Self_Employed']    = df_train['Self_Employed'].fillna(df_train['Self_Employed'].dropna().mode().values[0] )
df_train['LoanAmount']       = df_train['LoanAmount'].fillna(df_train['LoanAmount'].dropna().mean())
df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].dropna().mode().values[0] )
df_train['Credit_History']   = df_train['Credit_History'].fillna(df_train['Credit_History'].dropna().mode().values[0] )



#X,y  = df_train.iloc[:, 1:-1], df_train.iloc[:, -1]

y = df_train['Loan_Status']                       #dependent variable
X = df_train.drop('Loan_Status', axis = 1)        #independent variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='entropy',n_estimators=300,random_state=30)
forest.fit(X_train, y_train)
#ypred_forest = forest.predict(X_test)
print("Accurecy is :",forest.score(X_test,y_test)*100,'%')

filename="finalmodel.sav"
joblib.dump(forest,filename)
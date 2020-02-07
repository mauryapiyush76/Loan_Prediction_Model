# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:18:22 2020

@author: Piyush Mourya
Loan Predictor
"""
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Reading the data
train=pd.read_csv("train_ctrUa4K.csv")
test=pd.read_csv("test_lAUu6dG.csv")

# making a copy of the original datasets

train_original=train.copy()
test_original=test.copy()

# Understanding the data

train.columns
test.columns

train.dtypes

train.shape, test.shape

# Univariate analysis
"""
first, we are going to analyze the target variable
"""
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()

"""
Now we will visualize each variable seperately
"""
#Independent Variable(categorical)
plt.figure(1) 
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title="Married")
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title="Self_Employed")
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.show()

#ordinal_value
plt.figure(1)
plt.subplot(131)
train["Dependents"].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents')
plt.subplot(132)
train["Education"].value_counts(normalize=True).plot.bar(title="Education")
plt.subplot(133)
train["Property_Area"].value_counts(normalize=True).plot.bar(title="Property_Area")
plt.show()

#Independent variable numerical

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plot.show()

"""
we will compare the income accordind to educational background
"""
train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")
Text(0.5,0.98,'')

"""
coapplicant income distribution
"""
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plot.show()

"""
Loan amount variable
"""
plt.figure(1)
plt.subplot(121)
train_df=train.dropna()
sns.distplot(train_df['LoanAmount'])
plt.subplot(122)
train_df['LoanAmount'].plot.box(figsize=(16,5))
plot.show()

#Bivariate analysis

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.show()
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))
plt.show()
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')

bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')

bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')

train=train.drop(['Income_bin','Coapplicant_Income_bin','LoanAmount_bin','Total_Income_bin','Total_Income'],axis=1)
train=train.drop(['LoanAmount_bin','Coapplicant_Income_bin'],axis=1)
train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N',0,inplace=True)
train['Loan_Status'].replace('Y',1,inplace=True)

matrix = train.corr()
f,ax =plt.subplots(figsize=(9,6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")

# Missing value and outlier treatment
train.isnull().sum()

"""
categorical null variables are replaced by mode
"""
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train["Married"].mode()[0],inplace=True)
train['Dependents'].fillna(train["Dependents"].mode()[0],inplace=True)
train['Self_Employed'].fillna(train["Self_Employed"].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)

"""
Now we will see the LoanAmount variable. As it 
is a numerical variable, we can use mean or 
median to impute the missing values. We will 
use median to fill the null values as earlier 
we saw that loan amount have outliers so 
the mean will not be the proper approach as 
it is highly affected by the presence of outliers.
"""
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

test['Gender'].fillna(test['Gender'].mode()[0],inplace=True)
test['Married'].fillna(test["Married"].mode()[0],inplace=True)
test['Dependents'].fillna(test["Dependents"].mode()[0],inplace=True)
test['Self_Employed'].fillna(test["Self_Employed"].mode()[0],inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

#treating outliers
train['LoanAmount_log']= np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20) 
test['LoanAmount_log']=np.log(test['LoanAmount'])

#Model building

train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

X= train.drop('Loan_Status',1)
Y= train.Loan_Status

X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split

x_train,x_cv,y_train,y_cv = train_test_split(X,Y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
pred_test= model.predict(test)
submission=pd.read_csv("sample_submission_49d68Cx.csv")
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=False)

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')

# stratified k-folds cross validation

from sklearn.model_selection import StratifiedKFold

i=1 
kf= StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):
    print('\n{} of kfold{}'.format(i,kf.n_splits))
    xtr,xvl= X.loc[train_index],X.loc[test_index]
    ytr,yvl= Y[train_index],Y[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+=1
    pred_test=model.predict(test)
    pred=model.predict_proba(xvl)[:,1]
    
#Visualising the roc curve
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl,  pred,pos_label='Y') 
auc = metrics.roc_auc_score(yvl, pred) 
plt.figure(figsize=(12,8)) 
plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend(loc=4) 
plt.show()    

#Model building part 2


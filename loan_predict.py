import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from scipy.stats import mode
import csv
import sys


#OPENING THE DATASETS EXCEL FILES AND LOADING THEM INTO THIS MODULE
train_file=open("C:\\Python34\\datasets\\loan prediction\\train.csv");  ## was not working with single \ as \t is special char
test_file=open("C:\\Python34\\datasets\\loan prediction\\test.csv");


#READING THE DATASETS
train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)


#REVIEWING THE TRAIN DATASET
print( "\n \n ****TRAIN DATA**** \n \n ")
print(train_data.head(10))            # printing the top 10 rows
print(train_data.describe())   #gives description of numerical data


#REVIEWING THE TEST DATASET
print( " \n \n ****TEST DATA**** \n \n \n \n \n")
print(test_data.head(10))            #printing top 10 rows
print(test_data.describe())   #gives description of numerical data


#DISTRIBUTION ANALYSIS
plot1=plt.hist(train_data['ApplicantIncome'],bins=50)
plt.xlabel('APPLICANT INCOME')
plt.title('DISTRIBUTION ANALYSIS of APPLICANT INCOME')
plt.show()
train_data.boxplot(column='ApplicantIncome', by = 'Education')
plt.show()
train_data.boxplot(column='ApplicantIncome', by = 'Gender')
plt.show()
train_data.boxplot(column='LoanAmount')
plt.show()



#CATEGORIAL VARIABLE ANALYSIS
print(" \n \n \n \n \n **** CATEGORIAL VARIABLE ANALYSIS **** \n \n ")
temp1 = train_data['Credit_History'].value_counts(ascending=True)
temp2 = train_data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print (" \n \n Frequency Table for Credit History: \n ") 
print(temp1)

print ("\n \n Probility of getting loan for each Credit History class: \n \n ")
print (temp2)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')
plt.show()

fig = plt.figure(figsize=(8,4))
ax2 = fig.add_subplot(121)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind = 'bar')
plt.show()
temp3 = pd.crosstab(train_data['Credit_History'], train_data['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()


#CHECKING MISSING VALUES IN DATASET
print("\n \n \n \n \n \n \n \n \n \n \n \n \n CHECKING MISSING VALUES IN  TRAIN DATASET ")
print(train_data.apply(lambda x: sum(x.isnull()),axis=0))
print("\n \n \n \n \n \n \n \n \n \n \n \n \n CHECKING MISSING VALUES IN TEST DATASET ")
print(test_data.apply(lambda x: sum(x.isnull()),axis=0))


#FILLING MISSING VALUES  BY MEAN and MODE
# one method (more accurate one used) train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(), inplace=True)
train_data['Gender'].fillna(max(train_data['Gender'].value_counts()),inplace=True)
train_data['Married'].fillna(max(train_data['Married'].value_counts()),inplace=True)
train_data['Dependents'].fillna(max(train_data['Dependents'].value_counts()),inplace=True)
train_data['Self_Employed'].fillna(max(train_data['Self_Employed'].value_counts()),inplace=True)
train_data['Loan_Amount_Term'].fillna(max(train_data['Loan_Amount_Term'].value_counts()),inplace=True)
train_data['Credit_History'].fillna(max(train_data['Credit_History'].value_counts()),inplace=True)
  #the accurate method using pivot table
table = train_data.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
  # Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
  # Replace missing values
train_data['LoanAmount'].fillna(train_data[train_data['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)



test_data['Gender'].fillna(max(test_data['Gender'].value_counts()),inplace=True)
test_data['Dependents'].fillna(max(test_data['Dependents'].value_counts()),inplace=True)
test_data['Self_Employed'].fillna(max(test_data['Self_Employed'].value_counts()),inplace=True)

test_data['Loan_Amount_Term'].fillna(max(test_data['Loan_Amount_Term'].value_counts()),inplace=True)
test_data['Credit_History'].fillna(max(test_data['Credit_History'].value_counts()),inplace=True)
  #the accurate method using pivot table
table = test_data.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
  # Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
  # Replace missing values
test_data['LoanAmount'].fillna(test_data[test_data['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)





#RECHECKING MISSING VALUES IN DATASET
print("\n \n \n \n \n \n \n \n \n RECHECKING MISSING VALUES IN TRAIN DATASET ")
print(train_data.apply(lambda x: sum(x.isnull()),axis=0))
print("\n \n \n \n \n \n \n \n \n RECHECKING MISSING VALUES IN TEST DATASET ")
print(test_data.apply(lambda x: sum(x.isnull()),axis=0))




# MANAGING EXTREME VALUES IN LOAN AMOUNT AND APPLICANT AMOUNT
train_data['LoanAmount_log'] = np.log(train_data['LoanAmount'])
train_data['LoanAmount_log'].hist(bins=20)
plt.xlabel('LOAN AMOUNT')
plt.title('MANAGING EXTREME VALUES IN LOAN AMOUNT')
plt.show()

train_data['TotalIncome'] = train_data['ApplicantIncome'] + train_data['CoapplicantIncome']
train_data['TotalIncome_log'] = np.log(train_data['TotalIncome'])
train_data['LoanAmount_log'].hist(bins=20)
plt.xlabel('APPLICANT INCOME')
plt.title('MANAGING EXTREME VALUES IN APPLICANT INCOME')
plt.show()



#ENCODING THE CATEGORIES

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
var1_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = preprocessing.LabelEncoder()

for i in var_mod:
  train_data[i]=train_data[i].factorize()[0]
train_data.dtypes
for i in var1_mod:
  test_data[i]=test_data[i].factorize()[0]
test_data.dtypes
  


#REVIEWING THE TRAIN DATASET
print( "\n \n ****TRAIN DATA**** \n \n ")
print(train_data.head(10))            # printing the top 10 rows
print(train_data.describe())   #gives description of numerical data

#REVIEWING THE TEST DATASET
print( "\n \n ****TEST DATA**** \n \n ")
print(test_data.head(10))            # printing the top 10 rows
print(test_data.describe())   #gives description of numerical data


#LOGISTIC REGRESSION
#FITTING DATA INTO MODEL
outcome_var =['Loan_Status']
y=train_data[outcome_var]
model = LogisticRegression()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']   ##the gender is not taken as a possible term affecting the prediction
  #Fit the model:
clf=model.fit(train_data[predictor_var],y.values.ravel())
model.score(train_data[predictor_var],y.values.ravel())

print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n ")
print("AFTER FITTING INTO LOGISTIC REGRESSION MODEL")
print("\n \n Coefficient: \n", model.coef_)
print(" \n \n Intercept: \n", model.intercept_)

predictions = model.predict(test_data[predictor_var])
print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n ***** PREDICTIONS OF TEST FILE BASED ON LOGISTIC REGRESSION ***** \n \n ")
print(predictions)

probabilities=model.predict_proba(test_data[predictor_var])
print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n ***** PROBABILITIES OF TEST FILE BASED ON LOGISTIC REGRESSION ***** \n \n ")

print(probabilities)














cont=1
while cont==1 :
  print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n ***** DO YOU WANT TO DISPLAY THE PREDICTION AND PROBABILITY OF A PARTICULAR TEST CASE? ***** \n \n \n ")
  choice=int(input("\n ENTER CHOICE (Y=1/N=0) : "))
  print(choice)
  if choice == 0 :
    print("\n \n THANK YOU! THE PREDICTIONS AND PROBABILITIES OF LOAN APPROVAL FOR THE ENTIRE TEST FILE WAS DISPLAYED ABOVE! \n ")
  elif choice == 1 :
    row_req=int(input("\n \n ENTER THE ROW NUMBER FOR WHICH THE PREDICTION AND PROBABILITY IS REQUIRED : "))
    print("\n \n USING LOGISTIC REGRESSION THE LOAN FOR THE GIVEN CASE IS APREDICTED TO BE ( 0=NOT APPROVED / 1=APPROVED ) " )
    print(predictions[row_req])
    print("\n THE PROBABILITY AS FOUND IS : ")
    print(probabilities[row_req])
  else:
    print("\n WRONG CHOICE! \n THANK YOU! ")
cont=input("\n DO YOU WANT TO CONTINUE?(True(1)/False(0))")
  

  

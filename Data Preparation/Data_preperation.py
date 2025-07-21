# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:36:19 2024

@author: ADMIN
"""
########################################################
#Data Preprocessing

import pandas as pd
#let us import dataset
df=pd.read_csv("c:/5-Data_prep/ethnic diversity.csv")
#let us check data types of columns
df.dtypes
#salaries data type is folat ,let us convert it into int
#df1=df.Salarries.astype(int)
df.Salaries=df.Salaries.astype(int)
df.dtypes
#now the data type od salaries is int
#Similarly age data type must be float
#presentlyit is int
df.age=df.age.astype(float)
df.dtypes

########################################################
#identify the duplicates
df_new=pd.read_csv("c:/5-Data_prep/education.csv")
duplicate=df_new.duplicated()
#Output of this function is single cloumn
#if there is duplicate records output-True
#if there is no duplicate record output-False
#Seires will be created
duplicate
sum(duplicate)
#output will be 
#Now let us import another dataset
df_new1=pd.read_csv("c:/5-Data_prep/mtcars_dup.csv")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
#There are 3 duplicate records
#row 17 is duplicate of row 2 likes wise you can 3
#there is function called drop_duplicates()
#Which will drop all the dduplicate records
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)

#######################################################
#Outliers treatment
import pandas as pd
import seaborn as sns
df=pd.read_csv("c:/5-Data_prep/ethnic diversity.csv")
#Now let us find outliers in Salaries
sns.boxplot(df.Salaries)
#There are no outliers
#let us calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observed IQR in variable explorer
#no,because IQR is in capital letters
#treated as constant
IQR
#but if we will try as I, IQR or iqr or Iqr then it is showing
#I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#Iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#Now it you will check the lower limit of salary, it is -19446.9675
#There is negative salary, so make it as 0
#How to make it --> go to variable explorer and make it 0
lower_limit
upper_limit
#Upper limit is 93992.8125

###################################################################

#Trimming 
'''Trimming should be the last method to remove outliers'''
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
#you can check outliers_df column in variable explorer

df_trimmed=df.loc[~outliers_df]
df.shape
#(310,13)
#After trimming
df_trimmed.shape
#(306, 13)   #outliers are removed after trimming --> 4 outliers are removed

#Now I want to recomform that the outliers are removed or not?
sns.boxplot(df_trimmed.Salaries)

#####################################################################

#Replacement Technique
#Drawback of trimming --> losing the data
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("C:/5-data_prep/ethnic diversity.csv")
df.describe() 

#record no.23 has got outliers
#map all the outlier value to upper limit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,
            np.where(df.Salaries<lower_limit,lower_limit,
            df.Salaries)))
'''If the values are greater than upper limit --> map it to upper_limit,
and less than lower limit map it to lower limit, it it is within the range
then keep it as it is.'''
sns.boxplot(df_replaced[0])

#################################################

#Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries']
                  )
#Copy Winsorizer and paste in Help tab of top right window, study the method

df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])

##################################################
#Zerovarience and near zero varience
#if there is no varience in feature, then ML model 
#will not be intelligent
import pandas as pd
df=pd.read_csv("C:/5-Data_prep/ethnic diversity.csv")
df.var() #error
#here EmpId and ZIP is nominal data
#Salary has 4.441953e+08 is 4441953000
#not close to zero
df.info()
df.Salaries.var()==0
df.var(axis=0)==0 #error

###################################################3#
import pandas as pd
import numpy as np
df=pd.read_csv("C:/5-Data_prep/modified ethnic.csv")
#check for null values
df.isna().sum()

#####################################################
#create imputer that creates NaN values
#mean and median is used for numeric data
#mode is used for discrete data(position,sex,,MaritalDes)
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#check the dataframe
df['Salaries'] = pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))
#check the dataframe
df['Salaries'].isna().sum()

###################################################
#median imputer [used when their are outlier here in age outlier are present no in salaries]
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
#check the dataframe
df['age'] = pd.DataFrame(median_imputer.fit_transform(df[['age']]))
#check the dataframe
df['age'].isna().sum() 

##################################################
from sklearn.impute import SimpleImputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
#check the dataframe
df['Sex'] = pd.DataFrame(mode_imputer.fit_transform(df[['Sex']]))
df['Sex'].isna().sum() 
#o/p:- 0
df['MaritalDesc']= pd.DataFrame(mode_imputer.fit_transform(df[['MaritalDesc']]))
df['MaritalDesc'].isna().sum()
#o/p :- 0

##################################################
#install anaconda prompt[pip install imbalanced-learn scikit-learn]
'''IMP'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

#Step 1: Generate an imbalanced dataset
X,y = make_classification(n_samples=1000, n_features=20,n_informative=2,n_redundant=10,n_clusters_per_class=1,weights=[0.99],flip_y=0,random_state=1)

'''
Parameters -

n_samples=1000:
    the total number of samples (data points) to generate. 
    here 1000 sample will be created
n_features=20:
    The total number of features(columns) in the daset.
    Each sample will have 20 features
       
n_informative=2:
    the number of informative features.
    these features are useful for oredicting the target variables
    
n_redundant=10:
    the number of radundant features.
    tjese featres are generated as random linear combinations of the 
    
n_clusters_per_class=1:
    the number of clusters per class.
     each class will have one cluster of points
     this parameter is useful for contrlling the overlap between
    
weight=[0.99]:the propogation of samples assigne to each class,
Here 99%of samples will bringto one class,
creating a sifnificant class imbalanced the remaining 1% will be
    
    
flip_y=0: The fraction of samle whose class is randomly fliped
    
random_state = 1:
    the seed used by the random number generator.this 
    
'''

#Show the original classs distribution
print("Original class distribution:",np.bincount(y))

#Step :2 Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res,y_res=smote.fit_resample(X,y)

#Show the new class distribution after applying SMOTE
print("Resampled class distribtion",np.bincount(y_res))



#Show the original class distribution
print("Original class distribution:",np.bincount(y))

#Step 2:
smote= SMOTE(random_state=42)
X_res,y_res=smote.fit_resample(X,y)
    
#Show the new class distribution after applying SMOTE
print("Resampled class distribution :",np.binary(y_res))

'''

'''
#Show the original class distribution

print(f"Original class distribution:{np.bincount(y)}")
from sklearn.model_selection import train_test_split
#Step 2:Split the data into trainig and testing set
X_train,X_test,y_train,y_tain=train_test_split(X,y,test_size=0.3,random_state=42)

#Step 3:Apply SMOTE to balance the training data set
smote=SMOTE(random_state=42)
X_train_res,y_train_res=smote.fit_resample(X_train,y_train)

#Show the new class distribution after applying SMOTE
print(f"Resampled class distribution:{np.bincountry(y_train_res)}")

from sklearn.ensemble import RandomForestClassifier
#Step 4:Train a classifier on the balanced dataset
clf=RandomForestClassifier(rndom_state=42)
clf.fit(X_train_res,y_train_res)

#Step 5:Evaluate the classifier on the test set
y_pred=clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))


#######################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Generate a sample dataset
np.random.seed(0)
data=np.random.exponential(scale=0.2,size=1000)
df=pd.DataFrame(data,columns=['Value'])

#porform log transformation
df['LogValue']=np.log(df['Value'])

#Plot the original and log transformed data
fig,axes=plt.subplots(1,2,figsize=(12,6))

#Original data
axes[0].hist(df['Value'],bins=30,color='blue',alpha=0.7)
axes[0].set_title('Original Data')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

#Log_transformed data
axes[1].hist(df['LogValue'],bins=30,color='green',alpha=0.7)
axes[1].set_title('Log-transformed Data')
axes[1].set_xlabel('Log(Value)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:46:38 2024

@author: ADMIN
"""
####09.08.24
import pandas as pd
import matplotlib.pylab as plt
# Now import file from data set and create a dataframe
Univ1=pd.read_excel("C:/Data Set/University_Clustering.xlsx")
a=Univ1.describe()
a
# We have one column "State" which really not useful we will drop that column
Univ=Univ1.drop(["State"],axis=1)
# We know that there is scale difference among the columns, which we have to remove
# either by using normalization or standardization
# Whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to Univ dataframe
# For all the rows and column from 1 until end
# since 0th column has University name hence skipped
df_norm=norm_func(Univ.iloc[:,1:])
#pip install openpyxl
df_norm
# You can check the df_norm dataframe which is scaled between values from 0 to 1
# you can apply describe function to new data frame
b=df_norm.describe()
b
# before applying clustering , you need to plot dendogram
# now create dendogram need to calculate distance/ measure distance, we have
# to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
# Linkage function gives us hierarchical or aglomerative clustering
# ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendogram");
plt.xlabel("Index");
plt.ylabel("Distance")
# ref help of dendogram
# sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
# dendrogram
# applying agglomerative clustering choosing 5 as clusters from dendrogram
# whatever has been displayed in dendogram is not clustering
# it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering 
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',
                                   metric='euclidean').fit(df_norm)
# metric or affinity --> will work
# affinity has been depricated, use metric
# apply labels to the clusters 
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
# Assign this series to Univ DataFrame as column and name the column 
Univ['clust']=cluster_labels
# we want to relocate the column 7 to 0 th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
# now check the Univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
# from the output cluster 2 has got highest TOP10
Univ1
############################################################
####12.08.24
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv("C:/Data Set/income (1).csv")
df.head()
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('income($)')
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted
df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

######################################################
#####13.08.24
#Preprocessing using min max scaler
scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age']= scaler.transform(df[['Age']])
df.head()

plt.scatter(df.Age,df['Income($)'])

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted


df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

#######################################################
####14.08.24
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
from sklearn.cluster import kMeans
#let us try to understand first how kmeans work for two dimensional data
#for that,generate random number in the range0 to 1
#and with ubiform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create an empty dataFrame with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=["X","Y"])
#Assign the values of x and y to those columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)

'''
With data X and y,apply Kmeans model,generate scatter plot with scale/font=10
cmap=plt.cm.coolwarm:cool color cobination
'''
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

########################################
Univ1=pd.read_excel("C:/Data Set/University_Clustering.xlsx")
a=Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)
#We know that there is scale difference among the columns, which we have eithere by using normalization or satndardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to univ datframe for all the column
df_norm=norm_func(Univ.iloc[:,1:])

########################################
'''
What will be ideal cluster number,will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)#Total within sum of square
    
'''
KMeans inertia,also known as sum of square Errors(or SSE),
calculates the sum of the distanceof all points
within a cluster from the centroid of the point.
It is the differance between the obseved value
and the predicted value.
'''
TWSS
#As k value increases the tWSS value descreases
plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.ylabel("Total_within_SS")

'''How to select value of k from elbow curve
when k changes from 2 to 3, then decrease in twss is higher than 
when k changes  from 3 to 4.
When k values changes from 5 to 6 decreases.
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()

Univ.to_csv("kmeans_University.csv",encoding="utf-8")

###############################################
'''
1.Business Peoblem
1.1. What is the business objective
2.2. Are there any constraints
'''
#Data Description 
#Data Description:
    #Murder--Murder rates in different places of United
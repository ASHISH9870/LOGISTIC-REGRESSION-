#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None) #(this will display you all the columns)


# In[3]:


adult_df=pd.read_csv(r"C:\Users\91987\Downloads\drive-download-20221113T030749Z-001\adult_data.csv",header=None,delimiter=' *, *')


# In[4]:


adult_df.head()


# In[5]:


adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

adult_df.head()


# In[6]:


adult_df.shape


# In[7]:


adult_df.describe(include="all")


# # preprocesing the data

# In[8]:


# create a copy of the data frame ( so that orignal data does not get affected)
adult_df_rev=pd.DataFrame.copy(adult_df)


# In[9]:


adult_df_rev.drop(["education","fnlwgt"],axis=1,inplace=True)
adult_df_rev.shape


# In[10]:


adult_df_rev.dtypes


# In[11]:


# adult_df_rev.Age.astype(int) (to convert the data type)


# In[12]:


adult_df_rev.isnull().sum()


# In[13]:


# to identify special characters in data such as @,?


# In[14]:


for i in adult_df_rev.columns:
    print({i:adult_df_rev[i].unique()})


# In[15]:


adult_df_rev.replace('?',np.nan,inplace=True)


# In[16]:


adult_df_rev.isnull().sum()


# In[17]:


# replace the missing values with mode values
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace=True)
              


# In[18]:


adult_df_rev.workclass.mode()[0]


# In[19]:


adult_df_rev.isnull().sum()


# In[20]:


# to replace the missing values with appropiate central tendency 


# In[21]:


"""
for x in adult_df_rev.columns:
    if adult_df_rev[x].dtype=='object' or adult_df_rev[x].dtype=='bool':
        adult_df_rev[x].fillna(adult_df_rev[x].mode()[0],inplace=True)
    elif adult_df_rev[x].dtype=='int64' or adult_df_rev[x].dtype=='float64':
        adult_df_rev[x].fillna(round(adult_df_rev[x].mean()),inplace=True)
"""


# In[22]:


# Converting cateorgical data to numerical data 
# 1 Manual encoding-replace(),map()
# 2 Dummy variables- pd.getdummies(),OneHotEncoder()
# 3 Creating levels-LabelEncoder() 


# In[23]:


adult_df_rev.workclass.value_counts()


# In[24]:


adult_df_rev_new=pd.get_dummies(adult_df_rev)
adult_df_rev_new.head()


# In[25]:


adult_df_rev_new.shape


# In[26]:


colname=[]
for x in adult_df_rev.columns:
    if adult_df_rev[x].dtype=='object':
        colname.append(x)
colname       


# In[27]:


# For preprocessing the data
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for x in colname:
    adult_df_rev[x]=le.fit_transform(adult_df_rev[x])
    """
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print('Feature', x)
    print('mapping', le_name_mapping)
    """


# In[28]:


adult_df_rev.head()


# In[29]:


adult_df_rev.dtypes


# In[30]:


X=adult_df_rev.values[:,0:-1]
Y=adult_df_rev.values[:,-1]


# In[31]:


X.shape


# In[32]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
#X=scaler.fit_transform(X)
#print(X)


# In[33]:


X


# In[34]:


Y=Y.astype(int)


# In[35]:


from sklearn.model_selection import train_test_split
# Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[36]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[37]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(Y_pred)


# In[38]:


#print(list(zip(Y_test,Y_pred)))

print(list(zip(adult_df_rev.columns[:-1],classifier.coef_.ravel())))
print(classifier.intercept_)
#classifier.coef_


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# # Adjusting the threshold

# In[41]:


# store the predicated probablities
y_pred_prob=classifier.predict_proba(X_test)
print(y_pred_prob)


# In[42]:


y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.46:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#print(y_pred_class)        


# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,y_pred_class))

acc=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ",acc)


# In[44]:


for a in np.arange(0.4,0.61,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])


# In[45]:


from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)

print(auc)


# In[46]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()


# In[47]:


from sklearn import metrics
# y_pred_class is the list of predicated values on the basis of 0.46 threshold

fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_class)
auc = metrics.auc(fpr,tpr)

print(auc)
print(fpr)
print(tpr)


# In[48]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

AUC: Area under curve
1. Find out the overall AUC of the model by passing the entire prob matrix to the roc_curve function. It will try and test various different thresholds and end up giving a proper ROC curve.
2. Try generating the AUC value on the basis of individual thresholds.(try only upon the ambiguous thresholds which end up giving you almost the same error)
3. Finally conclude upon the threshold which gives you an AUC closest to the overall AUC.
# In[49]:


from sklearn.linear_model import SGDClassifier
#create a model
classifier=SGDClassifier(loss="log",random_state=10,learning_rate="constant",
                         eta0=0.00001,max_iter=1000, shuffle=True,
                        early_stopping=True,n_iter_no_change=5)
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

#print(list(zip(adult_df_rev.columns[:-1],classifier.coef_.ravel())))
#print(classifier.intercept_)


# In[50]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[51]:


classifier.n_iter_


# In[52]:


classifier.t_


# In[53]:


X_train.shape


# # Testing data

# In[54]:


adult_test=pd.read_csv(r"C:\Users\91987\Downloads\drive-download-20221113T030749Z-001\adult_test.csv",header=None, 
                       delimiter=' *, *')


# In[55]:


adult_test.head()


# In[56]:


adult_test.shape


# In[57]:


adult_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

adult_test.head()


# In[58]:


adult_test.shape


# In[59]:


adult_test.describe(include="all")


# # preproceesing the data

# In[60]:


adult_test.drop(["education","fnlwgt"],axis=1,inplace=True)
adult_test.shape


# In[61]:


adult_test.isnull().sum()


# In[62]:


adult_test.dtypes


# In[63]:


# to identify special characters in data such as @,?


# In[64]:


for i in adult_test.columns:
    print({i:adult_test[i].unique()})


# In[65]:


adult_test.replace('?',np.nan,inplace=True)


# In[66]:


adult_test.isnull().sum()


# In[ ]:





# In[67]:


for x in adult_test.columns:
    if adult_test[x].dtype=='object' or adult_test[x].dtype=='bool':
        adult_test[x].fillna(adult_test[x].mode()[0],inplace=True)
    elif adult_test[x].dtype=='int64' or adult_test[x].dtype=='float64':
        adult_test[x].fillna(round(adult_test[x].mean()),inplace=True)


# In[68]:


adult_test.isnull().sum()


# In[69]:


# to replace the missing values with appropiate central tendency 


# In[70]:


colname=[]
for x in adult_test.columns:
    if adult_test[x].dtype=='object':
        colname.append(x)
colname   


# In[71]:


# For preprocessing the data
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for x in colname:
    adult_test[x]=le.fit_transform(adult_test[x])


# In[72]:


adult_test.head()


# In[73]:


adult_test.dtypes


# In[74]:


X_test_new=adult_test.values[:,0:-1]
Y_test_new=adult_test.values[:,-1]


# In[75]:


X_test_new.shape


# In[76]:


X_test_new=scaler.transform(X_test_new)
print(X_test_new)


# In[77]:


Y_pred_prob=classifier.predict_proba(X_test_new)
print(Y_pred_prob)


# In[78]:


Y_pred_new=[]
for value in Y_pred_prob[:,1]:
    if value>0.46:
        Y_pred_new.append(1)
    else:
        Y_pred_new.append(0)
#print(Y_pred_new)


# In[79]:


#Evalution matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test_new,Y_pred_new)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test_new,Y_pred_new))

acc=accuracy_score(Y_test_new, Y_pred_new)
print("Accuracy of the model: ",acc)


# # cross validation 

# In[ ]:


#Using cross validation
 
classifier=LogisticRegression()
 
#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10,random_state=10,shuffle=True)
#print(kfold_cv)
 
from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
                                                 y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())


# In[ ]:


#model tuning
 
for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])
 
    
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
 
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
 
print("Classification report: ")
 
print(classification_report(Y_test,Y_pred))
 
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# # Recurssive Feature Elimination (RFE)

# In[85]:


colname=adult_df_rev.columns
from sklearn.feature_selection import RFE
rfe = RFE(classifier, n_features_to_select=9)
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_) 


# In[88]:


Y_pred=model_rfe.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[89]:


#Evalution matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[ ]:


"""new_data=adult_df_rev[['age','workclass','occupation','sex','income']]
new_data.head()
new_X=new_data.values[:,:-1]
new_Y=new_data.values[:,-1]
print(new_X)
print(new_Y)
"""
#in case the RFE eliminates logically relevant variables, create a  new df manually by subsetting the vaiables derived from RFE as well as on the basis of domain knowledge.
#rest of the model building steps will have to be performed manually


# # Feature Selection using Univariate Selection

# In[90]:


X=adult_df_rev.values[:,:-1]
Y=adult_df_rev.values[:,-1]


# In[91]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
 

test = SelectKBest(score_func=chi2, k=10)
fit1 = test.fit(X, Y)
 
print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
X = fit1.transform(X)


# In[92]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
#X=scaler.fit_transform(X)
#print(X)


# In[93]:


from sklearn.model_selection import train_test_split
# Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[94]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(Y_pred)


# In[95]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# # Variance Threshold

# In[97]:


X=adult_df_rev.values[:,:-1]
Y=adult_df_rev.values[:,-1]


# In[102]:


from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(0.1)  # 
fit1 = vt.fit(X, Y)
print(fit1.variances_)
 
features = fit1.transform(X)
print(features.shape[1])
print(list(zip(adult_df_rev.columns,fit1.get_support())))


# In[ ]:





# In[ ]:





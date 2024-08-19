#!/usr/bin/env python
# coding: utf-8

# ***Scenario 3: PATIENT HEALTH RISK ASSESSMENT***

# **Importing and Viewing Dataset**

# In[5]:


import pandas as pd
data = pd.read_csv("C:/Users/ekns1/Downloads/medical_data.csv")
data.info()


# **Visualizing Health Data**

# In[6]:


data.head(100)


# In[7]:


import matplotlib.pyplot as plt
plt.hist(data['Age'], bins = 15, color = 'orange', label = 'Age')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Age Among Medical Data', fontsize = 20)
plt.show()


# In[8]:


plt.hist(data['Blood_Pressure'], bins = 15, color = 'purple', label = 'Blood Pressure')
plt.legend()
plt.xlabel('Blood Pressure')
plt.ylabel('Number of Patients')
plt.title('Blood Pressure Among Medical Data', fontsize = 20)
plt.show()


# In[9]:


plt.hist(data['BMI'], bins = 15, color = 'brown', label = 'BMI')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Number of Patients')
plt.title('BMI Among Medical Data', fontsize = 20)
plt.show()


# In[10]:


plt.hist(data['Disease_Present'], bins = 15, color = 'red', label = 'Disease Present')
plt.legend()
plt.ylabel('Number of Patients')
plt.xlabel('Disease Present')
plt.title('Disease Present Medical Data', fontsize = 20)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.show()


# **Logistic Regression Analysis, Analyzing Feature Impact, & Coefficient Interpretation**

# In[11]:


features = ['Age','BMI','Blood_Pressure']
target = 'Disease_Present'
X = data.loc[:, features].values
y = data.loc[:, target]. values


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[13]:


from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[14]:


y_pred = log_reg.predict(X_test)


# In[15]:


y_pred


# In[16]:


log_reg.coef_


# In[17]:


import seaborn as sns


# In[18]:


sns.pairplot(data, hue='Disease_Present')


# **Coefficient Interpretation**
# 
# Three coefficients were provided, representing each predictor variable. It shows that BMI has the biggest odds of being a positive indicator towards a disease being present with a coefficient of 0.42991714. It is much more significant as an indicator than age or blood pressure with a coefficient more than double the others, however those factors also influence likelihood of disease to a degree. Overall, this means that having a higher BMI is the biggest factor that influences a person's likelihood of developing a disease. 

# **Examining Predictive Modeling**

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)


# In[20]:


classifier.fit(X_train, y_train)


# In[21]:


y_pred = classifier.predict(X_test)
y_pred


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix


# In[23]:


cr = classification_report(y_test, y_pred)
print(cr)


# In[24]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# **Verifying The Classification Report & Confusion Matrix**

# In[132]:


tn = sum((y_pred == 0) & (y_test ==0)) 
print(tn)


# In[133]:


fp = sum((y_pred == 1) & (y_test ==0))
print(fp)


# In[134]:


tp = sum((y_pred == 1) & (y_test ==1))
print(tp)


# In[135]:


fn = sum((y_pred == 0) & (y_test ==1))
print(fn)


# In[136]:


precision = tp/(tp+fp)
print(precision)


# In[137]:


recall = tp/(tp+fn)
print(recall)


# In[138]:


f1_score = 2 * (precision*recall) / (precision + recall)
print(f1_score)         


# These results tell us that with a precision of 1.0, the model has a perfect ability to measure the accuracy of positive predictions. With a recall of 0.67, there were still issues in detecting false negatives however for the most part the model did an average job at doing so. The f1 score indicates that there is a good balance of precision and recall, which indicates that the model is able to effectively detect positives while identifying negatives for an overall more useable and strong model.

# **Descriptive Analytics**

# In[110]:


data_DP_0 = data[data['Disease_Present'] == 0]
data_DP_0


# In[111]:


data_DP_1 = data[data['Disease_Present'] == 1]
data_DP_1


# In[112]:


import matplotlib.pyplot as plt
plt.hist(data_DP_0['Age'], bins = 15, color = 'orange', label = 'Patients without Disease')
plt.hist(data_DP_1['Age'], bins = 15, color = 'brown', label = 'Patients with Disease')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Age Among Patients With & Without Disease', fontsize = 20)
plt.show()


# We see that among patients without disease, age is diverse and spread through ages 15-80. Out of this however, the most patients (12) are in their mid 20s and the least (3) are in their mid 30s. For patients with disease, we see that the age range is more limited, ranging from 55-80. This indicates that of all the patients, only some adults and the elderly currently have diseases.

# In[113]:


import matplotlib.pyplot as plt
plt.hist(data_DP_0['BMI'], bins = 15, color = 'brown', label = 'Patients without Disease')
plt.hist(data_DP_1['BMI'], bins = 15, color = 'orange', label = 'Patients with Disease')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Number of Patients')
plt.title('BMI Among Patients With & Without Disease', fontsize = 20)
plt.show()


# We see that among patients without disease, BMI ranges all over, between 5-40. A healthy BMI is indicated by the Canadian government as being between 18.5-24.9, which is demonstrated in the distribution of this graph where a major portion of patients reside. However we also see a major highs in patients with unhealthy BMIs, something that can be attributed to trends of overconsumption in North America. Among patients with disease, we see that all of them maintain a BMI that is unhealthy. This nods to the fact that having a high BMI may lead to a greater risk of disease.

# In[114]:


import matplotlib.pyplot as plt
plt.hist(data_DP_0['Blood_Pressure'], bins = 15, color = 'purple', label = 'Patients without Disease')
plt.hist(data_DP_1['Blood_Pressure'], bins = 15, color = 'orange', label = 'Patients with Disease')
plt.legend()
plt.xlabel('Blood Pressure')
plt.ylabel('Number of Patients')
plt.title('BMI Among Patients With & Without Disease', fontsize = 20)
plt.show()


# We see that among patients without disease, blood pressure ranges between 80-180. As indicated by the American Heart Association, blood pressure above 140 is considered high and often requires medication. Knowing this, it is understandable that the majority of patients lie below this threshold. It is also telling that all patients with disease contain a blood pressure over this unhealthy amount, indicating that a high blood pressure may lead to a higher chance of disease.

# **Diagnostics Analytics**

# Within the pair plots, we can see that patients without disease vary at every metric. For example, some patients may have a high blood pressure, yet not have any disease to worry about. Similarly, a patient may be high in age and have a high BMI, yet have no disease to worry about. This tells us that to be diseased there is no specific threshold that must be met. However, we see that among the diseased, there is a strong trend in which all diseased patients are high in every health metric. They are all high in age, BMI, and Blood pressure, illustrating that a combination of issues with all three measures significantly raises the risk of contracting a disease.

# **Predictive Analytics**

# In[122]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[123]:


import numpy as np
odds_ratios = np.exp(log_reg.coef_)
odds_ratios


# The accuracy of the model shows to be 0.95, illustrating that the model is extremely capable of predicting the real health of the patients. The odds ratio shows that there is noticeable association between diseases present and the health metrics as well, further demonstrating and backing up that these factors play a role in a patients ability to obtain a disease. Overall, the accuracy of this model furthers our ability to predict disease presence and in turn respond to it as fast as possible to prevent significant harm. Having a significant understanding along with mathematical proof that Age, BMI, and Blood Pressure all play a role in the presence of disease aids by acting as key indicators for disease before they even arise. These health metrics are also often manageable (outside of age) and are often easier to reduce rather than getting rid of a disease. Lastly, the knowledge of how to avoid a potential disease works to provide more flexibility through using medical equipment to treat these conditions before they become too extreme.

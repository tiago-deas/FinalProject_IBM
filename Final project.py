#!/usr/bin/env python
# coding: utf-8

# <b>Author:</b> Tiago de Almeida Silva<br>
# <b>Course:</b> Statistics for Data Science with Python<br>
# <b>Institution:</b> IBM

# <h1>Project Case Scenario</h1>
# 
# <p>You are a Data Scientist with a housing agency in Boston MA, you have been given access to a previous dataset on housing prices derived from the U.S. Census Service to present insights to higher management. Based on your experience in Statistics, what information can you provide them to help with making an informed decision? Upper management will like to get some insight into the following.</p>
# 

# In[168]:


import scipy.stats
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[169]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# The following describes the dataset variables:
# 
# ·      CRIM - per capita crime rate by town
# 
# ·      ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# ·      INDUS - proportion of non-retail business acres per town.
# 
# ·      CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 
# ·      NOX - nitric oxides concentration (parts per 10 million)
# 
# ·      RM - average number of rooms per dwelling
# 
# ·      AGE - proportion of owner-occupied units built prior to 1940
# 
# ·      DIS - weighted distances to five Boston employment centres
# 
# ·      RAD - index of accessibility to radial highways
# 
# ·      TAX - full-value property-tax rate per $10,000
# 
# ·      PTRATIO - pupil-teacher ratio by town
# 
# ·      LSTAT - % lower status of the population
# 
# ·      MEDV - Median value of owner-occupied homes in $1000's

# In[6]:


boston_df.head()


# In[7]:


boston_df.info()


# In[9]:


boston_df.describe(include = "all")


# # <h1>Task 1</h1>

# <b>1.1 For the "Median value of owner-occupied homes" provide a boxplot</b>

# In[170]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (14, 8))
plt.boxplot("MEDV", data = boston_df)
plt.xlabel("Boxplot", fontsize = 14) 
plt.ylabel("Median value of owner-occupied homes ($1000's)", fontsize = 14)
plt.title("Median value of owner-occupied homes in Boston", fontsize = 22)


plt.show()


# <b> 1.2 Provide a histogram for the Charles river variable</b>

# In[189]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (14,8))
plt.hist("CHAS", data = boston_df)
plt.title("Histogram for the Charles river variable", fontsize = 22)
plt.xlabel("Charles River dummy variable (1 if tract bounds river; 0 otherwise)", fontsize = 14) 
plt.ylabel("Frequency for Charles river variable", fontsize = 16)

plt.show()


# <b> 1.3 Provide a boxplot for the MEDV variable vs the AGE variable. 
# (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)</b>

# In[172]:


boston_df.loc[boston_df['AGE'] <= 30, 'Age_Group'] = "35 and younger"
boston_df.loc[(boston_df['AGE'] > 30) & (boston_df['AGE'] < 70), 'Age_Group'] = "Between 35 and 70"
boston_df.loc[(boston_df['AGE'] >= 70), 'Age_Group'] = "70 and older"

plt.figure(figsize=(14, 8))
box = sns.boxplot(x = 'Age_Group', y = 'MEDV', data = boston_df)
box.set_title("MEDV variable vs the AGE variable", fontsize = 22)
box.set_xlabel("Proportion of owner-occupied units built prior to 1940", fontsize = 16)
box.set_ylabel("Median value of owner-occupied homes in $1000's", fontsize = 15)
box.tick_params(labelsize=13)

plt.show()       


# <b>1.4 Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. 
# What can you say about the relationship?</b>

# In[173]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(14,8))
plt.scatter(x = "INDUS", y = "NOX", data = boston_df)
plt.title("Relationship between NOX concentrations and the proportion of non-retail business acres per town", fontsize = 16)
plt.xlabel("Proportion of non-retail business acres per town", fontsize = 15)
plt.ylabel("Nitric oxides concentration (parts per 10 million)", fontsize = 15)


plt.show()


# Answer: As we can see on the plot above there is a positive correlation between Nitric oxide concentrations and the proportion of non-retail business acres per town as there is a linear relationship between both variables. I will add below a linear regression line to the plot to make it easier to see this positive correlation.

# In[174]:


X = boston_df[['INDUS']]
Y = boston_df['NOX']

lm = LinearRegression()

lm.fit(X, Y)

predic = lm.predict(X)

plt.figure(figsize=(14,8))
plot = sns.regplot(x = "INDUS", y = "NOX", data = boston_df)
plot.set_title("Relationship between NOX concentrations and the proportion of non-retail business acres per town", fontsize = 16)
plot.set_xlabel("Proportion of non-retail business acres per town", fontsize = 15)
plot.set_ylabel("Nitric oxides concentration (parts per 10 million)", fontsize = 15)

plt.show()


# <b>1.5 Create a histogram for the pupil to teacher ratio variable</b>

# In[175]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(14,8))
plt.hist("PTRATIO", data=boston_df)
plt.title("Histogram for the pupil to teacher ratio variable", fontsize = 20)
plt.xlabel("Pupil-Teacher ratio by town", fontsize = 16)
plt.ylabel("Frequency of Pupil-Teacher ratio by town", fontsize = 16)

plt.show()


# <h1>Task 2</h1>
# 
# <ul>
# <li>State your hypothesis.</li>
# 
# <li>Use α = 0.05</li>
# 
# <li>Perform the test Statistics.</li>
# 
# <li>State the conclusion from the test.</li>
# </ul>

# <b> 2.1 Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)</b>
# 
# My hypothesis:
# <ul>
# <li>H0:µ1=µ2  ("there is no difference in median value of houses bounded by the Charles river")</li>
# <li>H1:µ1≠µ2 ("there is a difference in median value of houses bounded by the Charles river")</li>
# </ul>
# 
# 

# In[176]:


scipy.stats.ttest_ind(boston_df['MEDV'], boston_df['CHAS'])


# Answer: The null hypothesis is rejected because the p-value is less than the alpha value of 0.05 and that means there is a statistical difference in median value between both variables.

# <b>2.2 Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)</b>
# 
# My hypothesis:
# <ul>
# <li>H0: µ1=µ2=µ3  (all means are equal)</li>
# <li>H1:  At least one mean is different</li>
# </ul>
# 
# 
# 

# In[177]:


first = boston_df[boston_df['Age_Group'] == "35 and younger"]['MEDV']
second = boston_df[boston_df['Age_Group'] == "Between 35 and 70"]['MEDV']
third = boston_df[boston_df['Age_Group'] == "70 and older"]['MEDV']

scipy.stats.f_oneway(first, second, third)


# Answer: The null hypothesis is rejected because the p-value is less than the alpha value of 0.05 and that means there is a statistical difference in Median values of houses for each proportion of owner-occupied units built before 1940. We can see it clearly in the plot below.

# In[178]:


order_list = ["35 and younger", "Between 35 and 70", "70 and older"]

plt.figure(figsize=(14, 8))
b = sns.barplot(x = "Age_Group", y = 'MEDV', data=boston_df, palette="Blues", ci=None, order=order_list)
b.set_title("MEDV variable vs the AGE variable", fontsize = 22)
b.set_xlabel("Proportion of owner-occupied units built prior to 1940", fontsize = 16)
b.set_ylabel("Median value of owner-occupied homes in $1000's", fontsize = 15)
b.tick_params(labelsize=13)
b.bar_label(b.containers[0])


plt.show()   


# <b>2.3 Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)</b>
# 
# My hypothesis:
# <ul>
# <li>H0:  There is no relationship between NOX concentrations and proportion of non-retail business acres per town.</li>
# <li>H1:  There is a relationship between NOX concentrations and proportion of non-retail business acres per town.</li>
# </ul>
# 
# 

# In[179]:


scipy.stats.pearsonr(boston_df['INDUS'], boston_df['NOX'])


# Answer: The null hypothesis is rejected because the p-value is less than the alpha value of 0.05 and that means there is a statistical relationship between NOX concentrations and proportion of non-retail business acres per town.

# <b>2.4 What is the impact of an additional weighted distance  to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)</b>
# 
# My hypothesis:
# <ul>
# <li>H0:β1  = 0 (No impact between both variables)</li>
# <li>H1:β1 ≠ 0 (There is an impact netween both variables)</li>
# </ul>
# 

# In[180]:


axis_x = boston_df['DIS']
axis_y = boston_df['MEDV']

axis_x = sm.add_constant(axis_x) 

finds = sm.OLS(axis_y, axis_x).fit()
pred = finds.predict(axis_x)

finds.summary()


# Answer: The null hypothesis is rejected because the p-value is less than the alpha value of 0.05 and that means there is a positive impact (coeficient of 1.0916) of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes.

# In[ ]:





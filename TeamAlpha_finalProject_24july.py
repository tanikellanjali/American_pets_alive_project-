#!/usr/bin/env python
# coding: utf-8

# # Finding lost animals 

# #### - A team Alpha project

# ## Problem statment and target audiance 

# with our code and observations we tried different prediction models to see which one helps us predict the location of a lost pet . We also went one step ahead and tried making an user interface which takes in their house data and helps predict where the pet could have gone to. Our target audiance are pet owners whoes pets are inquisitive about wandering around the locality and end up getting lost on the way home . 
# 
# 

# ## Required Libraries 

# In[1]:


import pandas as pd

import pickle 

import seaborn as sns

import numpy as np
from numpy import mean
from numpy import std

import matplotlib as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt

import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict


# ## Loading Data

# In[2]:


df = pd.read_excel("ampa_wmt_rto_hackathon_july-22_data.xlsx", sheet_name=['animal_data','shelter_data'])
animal_data = df['animal_data']
shelter_data = df['shelter_data']


# In[3]:


animal_data


# ## Understanding Animal Data 

# In[4]:


animal_data.columns


# In[5]:


dt = animal_data[['shelter_id','Species','distance_miles']]


# In[6]:


dt


# In[7]:


plt.figure(figsize=(16, 30))
g = sns.displot(data=dt, x="distance_miles", kind= "kde", height = 7)


# most of the animals were found close to the outcome address and were able to be returned 

# In[8]:


dt['shelter_id'].unique()


# In[9]:


keys = [1,2,3,4,5,6,7,8,9,10,11]
x = dict(zip(dt['shelter_id'].unique(), keys))
x


# In[10]:


dt2=dt.replace({"shelter_id": x})


# In[11]:


dt2


# In[12]:


dt['Species'].unique()


# In[13]:


dt2.loc[dt2['Species'] == "DOG", "Species"] = 'Dog'


# In[14]:


dt2['Species'].unique()


# In[15]:


dt2['Species'].nunique()


# In[16]:


dt2.nunique()


# In[17]:


keys2 = [101,102,103,104,105,106,100,107]
y = dict(zip(dt2['Species'].unique(), keys2))
y


# In[18]:


dt3=dt2.replace({"Species": y})


# In[19]:


dt3['Species'].unique()


# In[20]:


dt3.info()


# In[21]:


sns.displot(data=dt3,  x="shelter_id", kind="kde",height = 7)


# denisity of most animals are at shlter id 8 being dallas 

# In[22]:


ax = sns.relplot(data=dt3, x="shelter_id", y="distance_miles",hue = "Species",height = 7)
g.ax.axline(xy1=(10, 10), slope=.2, color="b", dashes=(5, 2))


# In[23]:


plt.figure(figsize=(16, 20))
plt.bar(dt3.shelter_id, dt3.distance_miles, color = (0.5,0.1,0.5,0.6))
plt.show()


# In[24]:


plt.bar(dt3.Species, dt3.shelter_id,)
plt.show()


# - dogs and cats are the most lost pets 
# - most of the distances covered by the pets are close to the owner houses 

# ## Understanding shelter data 

# In[25]:


shelter_data = df['shelter_data']


# In[26]:


shelter_data


# In[27]:


dfnew = shelter_data[['shelter_id','annual_intake_2019','annual_intake_2020','annual_intake_2021']]


# In[28]:


dfnew.set_index('shelter_id').plot()


# In[29]:


ax = sns.barplot(data=shelter_data, x = 'annual_intake_2021'  , y = 'jurisdiction_size_sq_km')


# In[30]:


ax = sns.barplot(data=shelter_data, x = 'jurisdiction_pop_size'  , y = 'jurisdiction_size_sq_km')


# ## Observations from given data 

# From visualisations we understand 
# - Shelter in dallas tho less in size compared to other shlters has more density of animal intake and has increased this intake stedily over the years . 
# - Dallas shelter has the highest size/pop ratio helping with the animal adoptation number 
# - The radius of finding an pet is close to the owners house most of the time 
# - Most number of species in the shelters are dogs and cats  

# In[31]:


ax1=animal_data.replace({"shelter_id": x})
ax1.loc[ax1['Species'] == "DOG", "Species"] = 'Dog'
keys2 = [101,102,103,104,105,106,100,107]
y = dict(zip(dt2['Species'].unique(), keys2))
ax2 = ax1.replace({"Species": y})
mask3 = (((ax2['shelter_id'] == 7)))
dt4 = ax2[mask3][['shelter_id', 'intake_date', 'Species', 'found_lng', 'found_lat',
       'outcome_lng', 'outcome_lat', 'distance_miles', 'found_address',
       'outcome_address']]


# In[32]:


dt4


# In[33]:


dt4['month'] = pd. DatetimeIndex(dt4['intake_date']). month


# In[34]:


dt4


# In[35]:


dt4['counts'] = dt4.month.map(dt4.month.value_counts())


# In[36]:


plt.bar(dt4.month,dt4.counts)
plt.show()
## we understand nothing about the intake patters from below 


# In[37]:


dt4


# ## Mapping values to understand location effects 

# Only considering dallas data for understanding a small set 

# In[38]:


import folium


# In[39]:


outcome_loc = [dt4['outcome_lat'].mean(), dt4['outcome_lng'].mean()]
outcome_map = folium.Map(location = outcome_loc, tiles="Openstreetmap", zoom_start = 5, control_scale=True)
for index, loc in dt4.iterrows():
    folium.CircleMarker([loc['outcome_lat'], loc['outcome_lng']], radius=2, weight=5,color="red").add_to(outcome_map)
folium.LayerControl().add_to(outcome_map)
outcome_map


# In[40]:


found_loc = [dt4['found_lat'].mean(), dt4['found_lng'].mean()]
map1 = folium.Map(location = found_loc, tiles='Openstreetmap', zoom_start = 5, control_scale=True)
for index, loc in dt4.iterrows():
    folium.CircleMarker([loc['found_lat'], loc['found_lng']], radius=2, weight=5).add_to(map1)
folium.LayerControl().add_to(map1)
map1


# ## observations from maps 
# animals are found very rarley on a freeeway where are hugh ways and main roads have 1:5 chance of spotting an animal to be found from the surrounding houses 

# ## Predicting the location where the animals can be found 

# ### understanding linear regression model

# In[41]:


y = dt4[['found_lng','found_lat']]
dt4.drop(['shelter_id','found_lng','found_lat','intake_date','month','counts','found_address','outcome_address'] , axis=1 , inplace = True)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(dt4, y, test_size=0.2 , random_state=43)


# In[43]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)


# In[44]:


regressor=LinearRegression()
regressor.fit(X_train_minmax,y_train)


# In[45]:


y_pred=regressor.predict(X_test_minmax)


# In[46]:


l = pd.DataFrame(y_pred)


# In[47]:


predicted_values = [l[1].mean(), l[0].mean()]
predicted_map = folium.Map(location = predicted_values, tiles='Openstreetmap', zoom_start = 5, control_scale=True)
for index, loc in l.iterrows():
    folium.CircleMarker([loc[1], loc[0]], radius=2, weight=5,colour = "YlGn").add_to(predicted_map)
folium.LayerControl().add_to(predicted_map)
predicted_map


# - top ten locations around your area where an animal can be found 
# - notification regarding danger zones for animals 
# 

# In[48]:


SQR_linearreg = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
R2_linearreg = r2_score(y_test, y_pred)
cv_r2_scores_lr = cross_val_score(regressor, dt4, y, cv=5,scoring='r2')
print(cv_r2_scores_lr)
print("Mean 5-Fold R Squared: {}".format(np.mean(cv_r2_scores_lr)))


# because of the high error rate we shifted to regression tree models for more accurate predictions 

# ### Regression tree

# In[49]:


animal_data.dropna(inplace = True)


# In[50]:


y1 = animal_data[['found_lng','found_lat']]
animal_data.drop(['shelter_id','Species','found_lng','found_lat','intake_date','found_address','outcome_address','distance_miles'] , axis=1 , inplace = True)


# In[51]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(animal_data, y1, test_size=0.2 , random_state=43)


# In[52]:


regr_2 = DecisionTreeRegressor(max_depth=5)
clf1 = regr_2.fit(X_train1, y_train1)
predicted1= clf1.predict(X_test1)


# In[53]:


lp=pd.DataFrame(predicted1)


# In[54]:


predicted_values = [lp[1].mean(), lp[0].mean()]
predicted_map1 = folium.Map(location = predicted_values, tiles='Openstreetmap', zoom_start = 5, control_scale=True)
for index, loc in animal_data.iterrows():
    folium.CircleMarker([loc[1], loc[0]], radius=2, weight=5,colour = "YlGn").add_to(predicted_map1)
folium.LayerControl().add_to(predicted_map)
predicted_map1


# ### Predicting for a single value input 
# 

# In[55]:


lst = [[-120.0609041 , 36.9519994]]
predicted1= clf1.predict(lst)
with open('model2', 'wb') as files:
    pickle.dump(clf1, files)
predicted1
do = pd.DataFrame(predicted1)


# In[58]:


predicted1 = [lp[1].mean(), lp[0].mean()]
predicted_map = folium.Map(location = predicted_values, tiles='Openstreetmap', zoom_start = 5, control_scale=True)
for index, loc in do.iterrows():
    folium.CircleMarker([loc[1], loc[0]], radius=50, popup = [loc[1], loc[0]] ,weight=5,colour = "YlGn").add_to(predicted_map)
folium.LayerControl().add_to(predicted_map)
predicted_map


# In[ ]:


predicted1


# ## Conclusion 

# We made a prediction model that helps us predict a certain location for a lost animal and map it using open street maps 

# ## Technologies and libraries

# - Pandas 
# - Pickle 
# - Seaborn
# - Numpy 
# - Matplotlib 
# - Sklearn
# - Open Street Map 
# - Folium

# ## Future Scope 

# This project conclusions and final data points can also be re invested into making an application to notify a concerned pet owner the possible whereabouts of a lost pet and the near by shelters it can be found it . 

# ## Authors 

# Team Alpha - 
# - Ishana 
# - Jorge Celaya 
# - Vishnu Vardhan Reddy Yeruva
# - Carina Ye
# - T. Anjali

# In[ ]:





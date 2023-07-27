#!/usr/bin/env python
# coding: utf-8

# In[33]:


import yfinance as yf
import numpy as np


# In[35]:


df = yf.download(["AAPL", "MSFT"], start = "2015-11-30", end = "2023-07-26")
df


# In[36]:


df = np.log(1+df["Adj Close"].pct_change())
df


# In[37]:


weights = [0.5, 0.5] #This assigns the stock weightage to a list
weights[0]*df["AAPL"].mean() + weights[1]*df["MSFT"].mean() #This assigns equal weightings to the mean returns.
#OR
np.dot(df.mean(), weights) #performs the dot product between the mean values of the columns in df and the elements in the weights list. The dot product is a mathematical operation that multiplies the corresponding elements of two vectors and then sums up the results.


# ## Portfolio Returns 

# In[38]:


#The np.dot(df.mean(), weights) can also be written as a function:
def portfolioreturn(weights):
    return np.dot(df.mean(), weights)

portfolioreturn(weights)


# In[39]:


df.cov() #This calculates covariance between stocks.


# In[42]:


#Calculating Portfolio Variance. This formula is given by the MPT formula.
pv = weights[0]**2*df.cov().iloc[0,0]+weights[1]**2*df.cov().iloc[1,1]+2*weights[0]*weights[1]*df.cov().iloc[0,1]
pv


# In[43]:


pv**(0.5) #To convert variance to standard deviation.


# In[44]:


#pv can also be calculated like this:
np.dot(np.dot(df.cov(), weights), weights)


# In[ ]:


#To find s.d. from above: 
np.dot(np.dot(df.cov(), weights), weights)**0.5


# In[47]:


#We should write the s.d. above as a function:
def portfoliostd(weights):
    return np.dot(np.dot(df.cov(), weights), weights)**0.5*np.sqrt(250) #Where 250 is the number of trading days in a year.
portfoliostd(weights)


# ## Creating random weights

# In[69]:


# rand is an array, which is like a list or a series of items that are arranged in a particular order.
def weightscreator(df):
    rand = np.random.random(len(df.columns)) #np.random.random() will return a single random number if you don't include any argument, which in this case is len(df.columns). 
    rand /= rand.sum() # This divides each element in the rand array by the total sum, forcing it to be equal to 1.
    return rand

weightscreator(df)


# In[73]:


returns = [] # These are lists used to store the results of a simulation process.
stds = []
w = [] # Note: w is used to store the results of 500 different weights.

for i in range(500):   # This creates a loop 500 times.
    weights = weightscreator(df)
    returns.append(portfolioreturn(weights)) # append allows you to add an element to the end of a list.
    stds.append(portfoliostd(weights))
    w.append(weights)


# In[92]:


returns # This generates 500 different returns.


# In[79]:


import matplotlib.pyplot as plt

plt.scatter(stds, returns)
plt.scatter(df.std().iloc[0]*np.sqrt(250), df.mean().iloc[0], c='k') # Understood.
plt.scatter(df.std().iloc[1]*np.sqrt(250), df.mean().iloc[1], c='g')
plt.scatter(min(stds), returns[stds.index(min(stds))], c='y')
plt.title("Efficient Frontier")
plt.xlabel("Portfolio Std")
plt.ylabel("Portfolio Returns")
plt.show()


# In[78]:


returns[stds.index(min(stds))] #Understood


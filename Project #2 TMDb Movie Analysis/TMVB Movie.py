#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Investigate TMVB Movie Dataset)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>
# 
# <a id='intro'></a>
# ## Introduction
# The dataset comes from Udacity Data Analyst Nano Degree, originally from the kaggle. This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue. It contains 10866 rows and 21 columns.
# 
# > We interest to learn more about
# >
# >1. Which genres are most popular from year to year? 
# >2. What kinds of properties are associated with movies that have high revenues?
# >3. What are the top 10 most profitable movies?
# >4. Which director produce most movies?
# 
# 

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > In this section, we will explore, inspect, and clean the data.
# 

# In[1]:


#loading the package and import the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('tmdb-movies.csv')
print(df.shape)
df.head()


# In[2]:


#create new column ---- calculate net profit for each movie 
df['net_profit'] = df['revenue'] - df['budget']


# ### Handle Missing Data

# In[3]:


# inspect the missing data in %
x = df.isna().sum()*100/len(df)
x


# In[4]:


# Drop columns that won't be used in our analysis
df.drop(columns = ['tagline','homepage', 'cast', 'production_companies', 'keywords'], inplace=True)
df.head()


# In[5]:


# Drop the missing value at genres columns
df.dropna(how='any', subset=['genres'], inplace=True)
# Double confirm
df.isna().sum()*100/len(df)


# In[6]:


# No duplicated row
df.duplicated().sum()


# In[7]:


df.describe()


# In[8]:


# count the budget columns have value less than 0
# There are 5674 columns less than 0 in budget columns, if needed, we will create different data frame to answer the questions
(df['budget'] <= 0).sum()


# ### More than half of the records in the budget has missing, this will reduce our result accuracy when exploring the relationship between budget, revenue, and net profit. We will exclude them during the analysis when we need to use the budget information.

# In[9]:


df.info()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Question 1 Which genres are most popular from year to year?
# > We notice the genres columns have more than 1 genre, we need to consider all the genres to find out the most popular genres. if we just choose one genre from each movie, our analysis would be biased.
# 
# > Here, we will use all the genre of each movie for our analysis.

# In[10]:


# copy a new dataframe 
df1 = df.copy()
#convert genres datatype to string
df1.genres = df1.genres.astype('str')


# In[11]:


# take a look popularity distribution
# the major popularity fall below 1
df.popularity.hist();


# In[12]:


df['popularity'].describe()


# In[13]:


# split the genres string
df1.genres = df1.genres.str.split('|')
df1.head(n=15)


# From the histogram and the summary of statistics. We can see the movie have popularity greater than 1 are considered popular.
# 
# Let explore further. 
# As I mention before, we notice the genres columns have more than 1 genre. For example, Jurassic World has 5 genres which are action, adventure, science, fiction, and thriller. It also has the highest popularity which is 32.98.
# 
# To tidy up the data, we will split the genre columns, create each row for each genre and perform the calculation.

# In[14]:


# create genre list( create each row for each gen) using explode 
df1 = df1.explode('genres')
df1.describe()


# In[15]:


# calculation -------popularity of each genres for each years
genres_count = df1.groupby(['release_year','genres'], as_index=False)['popularity'].mean()
genres_count.head()


# In[16]:


# group the dataframe again to find the most popular genre of each year 
x = genres_count.groupby('release_year').agg({'popularity':'max'})
x.head()


# In[17]:


# combine two dataframe into one 
new = genres_count.merge(x, how='inner', right_on='popularity', left_on='popularity')
new.head()


# In[18]:


# alternative, we also can use the groupby method 
# groupby year again and get the largest value
alt = df1.groupby(['release_year','genres'])['popularity'].mean().groupby(level='release_year').nlargest(1)


# In[19]:


# tidy up the data by removing extra row index by reset index
data = alt.reset_index(level=0, drop=True)
# change the pandas series to pandas dataframe
data = data.reset_index()
data.head()


# In[20]:


# set chart dimension
plt.figure(figsize=(14,8))

# set plotting style
sns.set_style('ticks') 

# set title
plt.title('Most Popular Genre of Every Year')
plt.xlabel('Movie Release Year')
plt.ylabel('Popularity')

#draw scatter plot 
sns.scatterplot(x="release_year", y="popularity", hue="genres", palette='dark:salmon_r', data=new)

#move the legend outside the chart
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);


# In[ ]:





# In[21]:


# set chart dimension
plt.figure(figsize=(18,10))
# heatmap plot 
sns.heatmap(df1.groupby(['release_year','genres'])['popularity'].mean().unstack('release_year'), cmap="rocket_r")

# set title
plt.title('Popular Genre of Every Year')
plt.xlabel('Genre')
plt.ylabel('Movie Release Year');


# In[22]:


# Popular of genres summary during 1960 to 2015 period
# Most popular genres over the years is Adventure, follow by Animation and Fantasy
df1.genres.value_counts().plot(kind='bar').set_ylabel('Count');


# In[23]:


df1.genres.value_counts().sort_values(ascending=False)[:5]


# ### *There is no significant trend of which genre constantly dominating over the year. The most popular genre during 1960 to 2015 period is Drama, follow by Comedy and Thriller.*

# ## Question 2 : What kinds of properties are associated with movies that have high revenues?

# In[24]:


#we know budget can't be less than 0, so filter out the dataframe
rev = df1[df1['budget'] > 0]


# In[34]:


# calculate the correlation
rev.corr()


# In[26]:


# group the genre 
rev = rev.groupby('genres', as_index =False).agg({'budget':'sum','revenue':'sum','net_profit':'sum','popularity':'mean'})
rev.head()


# In[27]:


# sort by highest revenue
rev = rev.sort_values(by='revenue', ascending=False)

#convert revenue in 1 billion unit
rev['revenue'] =rev['revenue']/1000000000

# convert net profit in 1 billion unit
rev['net_profit'] =rev['net_profit']/1000000000

# convert budget in 1 billion unit
rev['budget'] =rev['budget']/1000000000

rev


# In[28]:


# set chart dimension
plt.figure(figsize=(15,10))

# set plotting style
sns.set_style('ticks') 

# set title
plt.title('Movie Genre : Net Profit vs Revenue (Billion)')
plt.xlabel('Revenue')
plt.ylabel('Net Profit')

#draw scatter plot
sns.scatterplot(x="revenue", y="net_profit", hue="genres", data=rev);

#move the legend outside the chart
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);


# ### Strong positive correlation, 0.989509 between revenue and net profit.

# In[29]:


# set chart dimension
plt.figure(figsize=(15,10))

# set plotting style
sns.set_style('ticks') 

# set title
plt.title('Movie Genre : Revenue vs Budget (Billion)')
plt.xlabel('Budget')
plt.ylabel('Revenue')

# draw scatter plot
sns.scatterplot(x="budget", y="revenue", hue="genres", data=rev);

# move the legend outside the chart
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);


# ### Strong positive correlation, 0.996130 between revenue and budget.

# ### *It indicates that movies genres have higher revenue associated with a higher budget and higher net profits.*

# ## Question 3: What is the top 10 most profitable movies?

# In[30]:


profit = df.groupby(['original_title', 'popularity','genres', 'release_year', 'revenue','budget'])['net_profit'].sum()
profit = profit.reset_index().sort_values(by='net_profit', ascending=False)

# calculate return in investment - ROI
profit['ROI'] = profit['net_profit']/profit['budget']
profit[:9]


# In[31]:


# top 10 movies with highest revenues
revenue = profit.sort_values(by='revenue', ascending=False)
revenue[:9]


# ## We can see the top 10 profitable movies associated with higher popularity have a minimum return of investment of 400% and up to 1073%. The return on investment is quite attractive.
# 

# ### Question 4:  Which director produces the most movies?

# In[32]:


df['director'].value_counts().head(n=10)


# ###  Woody Allen produces the highest number of movies, which is 43.

# <a id='conclusions'></a>
# 
# 
# # The data limitations of this dataset:
# 1. Almost every movie has multiple genres, and we have included all the genres during our calculation. 
# 2. Lack of details on how the vote_count and vote_average calculation.
# 3. There are **5674** columns less than 0 in budget columns.
# 4. Do take note I use different data frame to answer different questions. I create a new data frame to answer Q1 & Q2. The remaining questions are used the original data frame.
# 5. The missing values in the data will affect our accuracy of analysis, if further investigation is needed, we can perform a statistical test to determine our result whether is statistically significant. The statistical test can help us determine whether our results in data are not determined by chance alone.
# 
# 
# # Conclusions
# 
# For our analysis, we found that the movie genres with higher revenues come with higher popularity, higher budget, and higher net profit. We found the same findings when we look at the individual movies with higher revenues. We can conclude that production films are willing to invest more money in popular movies genres. Based on the data, the higher budget will allow firms to produce high-quality movies which generate higher revenue and net profit. As we can see from the top 10 most profitable movies, the return on investments is 4 times to 10.73 times on their budget.  
# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as ply
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")


# In[2]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


movies.tail()


# In[5]:


movies.shape


# In[6]:


movies.columns


# In[7]:


movies.duplicated().sum()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.info()


# In[10]:


movies.describe()


# In[11]:


credits.head()


# In[12]:


credits.tail()


# In[13]:


credits.shape


# In[14]:


credits.columns


# In[15]:


## Capturing unnamed columns also


# In[16]:


credits.duplicated().sum()


# In[17]:


credits.isnull().sum()


# In[18]:


credits=credits[['movie_id', 'title', 'cast', 'crew']]


# In[19]:


## Fixxing extra populated columns


# In[20]:


credits.shape


# In[21]:


credits.info()


# In[22]:


movies.nunique()


# In[23]:


credits.nunique()


# In[24]:


sns.countplot(movies['status'])


# In[25]:


movies=movies.merge(credits, on = 'title')


# In[26]:


movies.head()


# In[27]:


movies.shape


# In[28]:


movies.columns


# In[29]:


movies_df = movies[['id', 'keywords','overview','title','cast','crew','genres']]


# In[30]:


movies_df.dropna(inplace=True)


# In[31]:


movies_df.shape


# In[32]:


movies_df.iloc[0].genres


# In[33]:


import ast 


# In[34]:


# Abstract Syntax Tree. ast. Import(names) is a class defined in the ast module 
# that is used to express the import statement in python in the form of an 
# Abstract Syntax Tree. When the parse() method of ast is called on a Python source code 
# that contains the import keyword, the ast.


# In[35]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[36]:


# The ast. literal_eval method is one of the helper functions that helps traverse 
# an abstract syntax tree. This function evaluates an expression node or a string 
# consisting of a Python literal or container display.


# In[37]:


# genres column is a list of dictionary and key “name” holds the genres. 
# our aim is to convert the list of the dictionary into a list of genres


# In[38]:


movies_df['genres']=movies_df['genres'].apply(convert)


# In[39]:


movies_df['genres']


# In[40]:


movies_df['keywords']=movies_df['keywords'].apply(convert)


# In[41]:


movies_df['keywords']


# In[42]:


movies_df.cast[0]


# In[43]:


# Cast column contains lists of dictionary of having all cast names. 
# we want to take only the top 3 casts for our recommendations


# In[44]:


def convert3(obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count<3:
            L.append(i['name'])
        count+=1
    return L


# In[45]:


movies_df['cast']=movies_df['cast'].apply(convert3)


# In[46]:


movies_df['cast'].head()


# In[47]:


movies_df.overview[0]


# In[48]:


movies_df['overview']=movies_df['overview'].apply(lambda x : x.split())


# In[49]:


movies_df['overview']


# In[50]:


movies_df.drop('crew',inplace=True,axis=1)


# In[51]:


def collapse(L):
    L1=[]
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
    


# In[52]:


movies_df['cast']=movies_df['cast'].apply(collapse)
movies_df['genres']=movies_df['genres'].apply(collapse)
movies_df['keywords']=movies_df['keywords'].apply(collapse)


# In[53]:


movies_df['Tags']=movies_df['overview']+movies_df['cast']+movies_df['genres']+movies_df['keywords']


# In[54]:


movies_df.head()


# In[56]:


new_df=movies_df[['id','title','Tags']]


# In[57]:


new_df.head()


# In[59]:


new_df['Tags'] = new_df['Tags'].apply(lambda x :" ".join(x))


# In[60]:


new_df['Tags'].head()


# In[61]:


new_df.head()


# In[63]:


new_df['Tags'][0]


# In[64]:


# CountVectorizer is a great tool provided by the scikit-learn library in Python. 
# It is used to transform a given text into a vector on the basis of the frequency 
# (count) of each word that occurs in the entire text.
# Convert a collection of text documents to a matrix of token counts.


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')


# In[66]:


vectors = cv.fit_transform(new_df['Tags']).toarray()


# In[67]:


vectors.shape


# In[68]:


cv.get_feature_names()


# In[69]:


len(cv.get_feature_names())


# In[70]:


# our model should be capable of finding the similarity between movies based on their tags.
# Our Recommender model takes a movie title as input and predicts top-n most similar movies 
# based on the tags
# here we will use the concept of Cosine distance to calculate the similarity of tags
# sklearn provides a class for calculating pairwise cosine_similarity.


# In[71]:


# In Data Mining, similarity measure refers to distance with dimensions representing 
# features of the data object, in a dataset. If this distance is less, there will be a 
# high degree of similarity, but when the distance is large, there will be a low degree of similarity.
# Cosine similarity is a metric, helpful in determining, how similar the data objects are 
# irrespective of their size. We can measure the similarity between two sentences in 
# Python using Cosine Similarity. In cosine similarity, data objects in a dataset 
# are treated as a vector. The formula to find the cosine similarity between two vectors is –
#Cos(x, y) = x . y / ||x|| * ||y||
#where,
#x . y = product (dot) of the vectors ‘x’ and ‘y’.
#||x|| and ||y|| = length of the two vectors ‘x’ and ‘y’.
#||x|| * ||y|| = cross product of the two vectors ‘x’ and ‘y’.


# In[72]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors).shape


# In[73]:


similarity = cosine_similarity(vectors)


# In[74]:


similarity[0]


# In[75]:


# Enumerate is a useful function in Python as it returns both the indexes and values of Lists.


# In[76]:


sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]


# In[77]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[78]:


recommend('The Dark Knight Rises')


# In[79]:


recommend('Avengers: Age of Ultron')


# In[80]:


recommend('Iron Man 3')


# In[81]:


recommend('Spider-Man 3')


# In[82]:


recommend('Toy Story')


# In[83]:


# Pickling is a way to convert a python object (list, dict, etc.) into a character stream. 
# The idea is that this character stream contains all the information necessary to reconstruct 
# the object in another python script.
# The Python pickle module is another way to serialize and deserialize objects in Python. 
# It differs from the json module in that it serializes objects in a binary format, which means 
# the result is not human readable.


# In[84]:


import pickle 
pickle.dump(new_df, open('movies_df.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))


# In[85]:


new_df['title'].values


# In[86]:


new_df.to_dict()


# In[87]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# GUID: 2557756 
# https://github.com/emilylonie/AI_Arts_A_23-24

# # Week 1: Getting started with Python, Jupyter Notebook and Anaconda
# Exercises to familarise myself with Jupyter Notebook and it's relation to Python

# ### Task 1

# ### a)	Why you chose to join this course – for, motivation, vision, aspiration?
# I chose to join this course as I think learning some Python will be a useful skill. And I find the impact of AI on the Humanities interesting.

# ### b) Prior experience, if any, you have with AI and/or Python
# 
# My experience with AI is from other classes, I took Introduction to the Digital Humanities last year which touched on the use of AI within the humanities. I have also had a little previous experience with Python in previous studies.

# ###  c) What you expect to learn from the course 
# * I expect to pick up on some Python language and knowledge and be able to apply this to the humanities work.
# * I expect to learn current developments in AI and associated applications
# * I expect to learn an overview of AI history and development

# <div class="alert alert-block alert-success">
# <b>Success:</b> I have now completed my markdown cells and can start coding!
# </div>

# ### Task 2.1

# In[2]:


print ("Hello, world!") #The print function outputs the code in the ("")


# In[3]:


message = "Hello, world!" #The message variable pre-defines the string 'hello, world!'
print (message)


# In[4]:


message2 = "Hello, autumn!"
print (message2)


# In[5]:


print (message + " " + message2)  #Here, + joins strings together while " " acts as a space


# In[6]:


print (message2*3)  #*3 repeats the string 3 times


# In[7]:


print ((message2 + " ")*3)


# In[8]:


print (message[0]) #[0] prints the first element in the string


# In[9]:


print (message2[7:13]) #[7:13] prints the elements in the string between positions 7 and 13


# ### Task 2.2

# In[10]:


from IPython.display import *  #imports a python library which is capable of displaying youtube videos 
YouTubeVideo("AqAJLh9wuZ0")  #Video sourced from youtube


# ### Task 2.3

# In[ ]:


import webbrowser #importing the web broswer library
import requests #importing requests library

print("Shall we hunt down an old website?") #printing string
site = input("Type a website URL: ") #assigning a value to "site" variable
era = input("Type year, month, and date, e.g., 20150613: ") #assigning a value to "era" variable
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era) #assigning a value to "url" variable
response = requests.get(url) #assigning a value to "response" variable
data = response.json() #assigning a value to "data" variable
try: #the following block may fail if it is unable to open the web broswer or the data response doesn't exist
    old_site = data["archived_snapshots"]["closest"]["url"] #assigning a value to "old_site" variable
    print("Found this copy: ", old_site) #printing string output
    print("It should appear in your browser.") #printing string output
    webbrowser.open(old_site) #command to open old_site
except: #if the previous code fails, the next line of code will be computed
    print("Sorry, could not find the site.") #printing string output (failure message)


# # Week 2: Exploring Data in Multiple Ways

# ### Task 3.1

# In[ ]:


from IPython.display import Image  #Imports python function capable of displaying images
Image ("picture1.jpg")  #retrieves an image saved on the computer


# In[11]:


from IPython.display import Audio  #Imports python function capable of playing audio

Audio ("audio1.mid")  #retrieves a midi file saved on the computer


# In[ ]:


Audio ("audio2.ogg")
#This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license. 
#You are free: 
#•to share – to copy, distribute and transmit the work
#•to remix – to adapt the work
#Under the following conditions: 
#•attribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
#•share alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
#The original ogg file was found at the url: 
#https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# # Explanation
# Audio1.mid will not play as it is a MIDI (Musical Instrument Digital Interface) file. This holds information on how musical instruments should be played, and needs software to translate it. However, Audio2.ogg works as it is compatible with the Python library's software.

# ### Task 3.2

# In[12]:


from matplotlib import pyplot  #imports a python library
test_picture = pyplot.imread("picture1.jpg") 
print("Numpy array of the image is: ", test_picture)
pyplot.imshow(test_picture)

#this prints numbers representing the RGB values of each pixel


# In[13]:


test_picture_filtered = 2*test_picture/3 # / means the data will be converted to floats
pyplot.imshow(test_picture_filtered) #Reffered to as Image1 


# In[14]:


test_picture_filtered = 2*test_picture//3 # // means the data will stay as an integer
pyplot.imshow(test_picture_filtered) #Reffered to as Image2


# In[15]:


test_picture_filtered = 2*(test_picture//3) #the brackets means the multiplication will not proceed until the divison has finished i.e there will be no overflow
pyplot.imshow(test_picture_filtered) #Reffered to as Image3


# # Discussion:
# 
# Image1*: is displaying incorectly as 2*test_picture/3 is dividing the pixels/RGB data by 3, causing them to turn into decimals/floats. This means that every pixel which is out with the range specified will show a white pixel instead of a coloured one. 
# 
# Image2: Here the double slash (//) is instructing the program to keep the values as an integer and not convert them to a string. However, as there are no brackets keeping the division seperate from the multiplication, overflow is occuring. This causes the pixels to display incorectly.
# 
# Image3: As a double slash is used as well as brackets to keep the division and multiplication seperate, the image is displaying without any errors.
# 
# 
# 
# *See comments for image names

# ### Task 3.3

# In[17]:


from sklearn import datasets  #loads the datasets


# In[18]:


dir(datasets)  #Shows all of the datasets/functions available in the library


# For this task I'm going to work with the datasets 'load_wine' as I like wine and 'load_breast_cancer' as womens health is an interesting topic to me

# In[20]:


wine_data = datasets.load_wine()
breast_cancer_data = datasets.load_breast_cancer()
#loads 2 different datasets from the library


# In[21]:


wine_data.DESCR #description of the data in this file


# In[ ]:


breast_cancer_data.DESCR 


# In[22]:


print (wine_data.DESCR) #Makes the description easier to read!


# In[ ]:


print (breast_cancer_data.DESCR)


# In[ ]:


wine_data.feature_names #by observing the feature names I can conclude that this dataset is trying to analyse what components makes various wines taste different.


# In[ ]:


breast_cancer_data.feature_names #From the featured names I can guess that this dataset is observing the average attributes of the collected data, as well as the most severe instances and errors


# In[23]:


wine_data.target_names


# In[ ]:


wine_data.keys() #the objects/categories are shown


# In[24]:


images_data = datasets.load_sample_images()
print (images_data.DESCR)


# In[ ]:


images_data.keys()


# In[ ]:


images_data.images #shows the data related to the images


# In[ ]:


images_data.filenames #shows the file names of the images


# ### Task 3.4

# In[25]:


from sklearn import datasets 
import pandas

wine_data = datasets.load_wine()

wine_dataframe = pandas.DataFrame(data=wine_data['data'], columns = wine_data['feature_names'])


# In[26]:


wine_data.data #Shows the values in the wine dataset in its raw form


# In[27]:


wine_dataframe.head() #when loaded with pandas, the data is formatted... and readable!


# In[28]:


wine_dataframe.describe() #pandas also allows you to compute statistics about the data


# ### Discussion:
# wine_data.data displays all of the data available in the dataset. 
# dataframe.head displays the first few rows of this data in a column (using feature_names as headings 
# dataframe.describe takes the averages and trends from all 178 rows of original data and displays it in a table (i.e provides a summary of the data)

# In[ ]:





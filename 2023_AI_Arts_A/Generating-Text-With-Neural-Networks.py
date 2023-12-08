#!/usr/bin/env python
# coding: utf-8

# GUID: 2557756 
# https://github.com/emilylonie/AI_Arts_A_23-24

# # Generating Text with Neural Networks

# ## Introduction to the Code
# The code below is designed to train a model to generate text in the style of William Shakespeare's work. The code utalises Tensorflow which is an open-source library which can be used to build neural networks. Neural networks are artificial intelligence (AI) designed to mimic the way humans think. Therefore, they can be very useful for generating text in the style of a person. The model below was trained on a database of William Shakespeare, it includes various plays (including Romeo & Juliet and Hamlet). 

# ## What can the code be used for?
# There are many potentials for a text generator such as this, within the humanities. It could be used for educational purposes such as language analysis. The text generated could be used to understand the lingustic styles of that period, or even observe how AI differs from true text of this time. Furthermore, it could be used educationally within nmuseums showcasing his work. Text generators could be used to create Immersive exhibits in the style of Shakespeare. 
# 
# Additionally, it could be used as a source of inspiration and creativity. As Shakespeare is regarded as one of the best poets, and playwrites of all time, it makes sense to use his methods of thinking and creativity as inspiration. Even today, popular artists are still comnpared to Shakespeare, despite the centuries of lingustic evelution. For instance, Taylor Swift's recent songwriting has been compared to Shakespeare's with online quizzes to see if players can tell the two apart: https://www.buzzfeed.com/taylorswiftmidnights/evermore-edition-taylor-swift-or-shakespeare. 
# 
# This showcases, how his writing style is still relevant today. Meaning, people in the arts and humanities could use a text generator as a starting point for their own work. 

# ## Weaknesses in this model
# 
# While the data was trained on several of Shakespeare's plays, providing it with thousands of lines of script, it was missing some key literature. For instance, it fails to include sonnets or plays such as Macbeth and A Midsummer Night's Dream. This means, the model is limited in its knowledge of Shakespeare's writing style, and is less likely to produce text outwith the style of a script.
# 
# Furthermore, using this method of generating text may be difficult and impractical for those in the humanities. This is due partly due to the coding knowledge needed. However, mainly the amount of time it takes to train neural networks may not be suited to those in the humanities. Depending on the coputer set-up, it may take anywhere for an hour to 20 hours to train the neural networks. If the text generation is needed instantly, it may make more sense for humanities scholars to use an instantaneous method, such as a conversational AI chatbot. There are ways to reduce the time required to train the model, such as training it on less data, or reducing the number of epochs (An epoch is the number of times the model spends looking at the training data). Although, both of these methods would reduce the quality of results.
# 
# As you will see, the output of text generated is not of the highest quality, as it is not logical. This sugests a larger data set should be used, or more epochs should have been used. It also indicates this was not successful for use by humanities scholars and alternatives should be considered.

# # Getting the Data

# *Disclaimer: The code was understood through conversations with my peers*
# 
# The cell below is importing the tensorflow library and acessing the data on Shakespeare.

# In[14]:


import tensorflow as tf


# In[15]:


import tensorflow as tf 

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url) 
with open(filepath) as f:
    shakespeare_text = f.read()


# This will print the first 80 characters of the data

# In[16]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# # Preparing the Data

# The code below is making all of the training data text lowercase. This reduces the number of characters the model has to interpret, allowing it to learn from the data more efficiently. It compacts the text by looking for patterns in the text and replacing them with more efficient encodings. Part of this is replacing the characters in the text with 'tokens' each of which can represent one or more characters.
# 
# <img src="neural_network.jpeg" alt="Neural Network diagram">
# Image by Sabrina Jiang © Investopedia 2020

# In[17]:


text_encoder = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_encoder.adapt([shakespeare_text])
encoded_data = text_encoder([shakespeare_text])[0]


# This prints the format of the compacted data representation

# In[18]:


print(text_encoder([shakespeare_text]))


# This finds out how many distinct tokens there are, in total there are 39. The full dataset has 1115394 tokens in total.

# In[19]:


encoded_data -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
number_of_tokens = text_encoder.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded_data)  # total number of chars = 1,115,394


# In[20]:


print(number_of_tokens, dataset_size)


# This prepares the data into a format suitable for training/testing our model, and splits it into a training/validation/testing set.

# In[21]:


def to_dataset(input_data, length, shuffle=False, seed=None, batch_size=32):
    # Convert data into a format suitable for tensorflow
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    
    # Split data into smaller groups (of size 'length')
    dataset = dataset.window(length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window_dataset: 
                                 window_dataset.batch(length + 1))
    if shuffle:
        dataset = dataset.shuffle(100_000, seed=seed)
        
    # Batches groups of data together
    dataset = dataset.batch(batch_size)
    return dataset.map(lambda window: 
                         (window[:, :-1], window[:, 1:])).prefetch(1)


# The training set contains 1 million tokens, the validation set is the next 60,000 tokens, while the testing set is the remaining tokens.
# 
# To make training the code faster, the number of tokens can be changed to a smaller number (perhaps 100_000).

# In[22]:


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded_data[:1_000_000], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded_data[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded_data[1_060_000:], length=length)


# # Building and Training the Model

# Below is the code which trains the model. As mentioned before, the number of epochs are set to 10, however, this could be increased to potentially raise the accuracy of the model. Or it could be decreased to train the model faster, however, this may make the model less accurate.

# In[9]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=number_of_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(number_of_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# This adds layers to the model which converts between input characters and the encoded representation of text that the model uses.

# In[25]:


model.save("shakespeare_model.keras") 
#This saves the neural network we just trained


# In[3]:


model = tf.keras.models.load_model('shakespeare_model.keras') 
model.summary() 
#this loads it, that way if the notebook is restarted, we don't have to train it again!


# In[26]:


shakespeare_model = tf.keras.Sequential([
    text_encoder,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# Here we get to start generating some text!!!
# 
# The cell below, shows how the model is able to predict what the final letter will be in the phrase "To be or not to b".

# In[11]:


token_probabilities = shakespeare_model.predict(["To be or not to b"])[0, -1]
predicted_token = tf.argmax(token_probabilities)  # choose the most probable character ID
text_encoder.get_vocabulary()[predicted_token + 2]


# The following cells set up the generator to select characters based on probability. The model determines how likely the next character will be and then uses the "temperature" to determine how much the probabilities influence the choice of character. Therefore, the higher the "temperature" is, the more random the text generation will be. And the lower the "temperature", the more accurate it will be.
# 
# <img src="to_be_or_not.jpeg" alt="To be or not to be. That is the question.">
# Image source: https://www.azquotes.com/quote/267345

# In[12]:


log_probabilities = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probabilities, num_samples=8)  # draw 8 samples


# In[23]:


def next_char(text, temperature=1):
    token_probabilities = shakespeare_model.predict([text])[0, -1:]
    token_weight = tf.math.log(token_probabilities) / temperature
    predicted_token = tf.random.categorical(token_weight, num_samples=1)[0, 0]
    return text_encoder.get_vocabulary()[predicted_token + 2]


# This defines the function "extend_text". This provides a method to generate large blocks of text by repeatedly predicting the next character. By default the number of characters generated is 50 and the temperature is 1, but these can be changed when using the function.

# In[24]:


def extend_text(text, number_of_chars=50, temperature=1):
    for _ in range(number_of_chars):
        text += next_char(text, temperature)
    return text


# Sets the randomness at 42 

# In[15]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# Prints the text in quotations, followed by generated text. The temperature and number of characters has been modified.

# In[16]:


print(extend_text("To be or not to be", temperature=0.01, n_chars=30))


# The cell above has a low temperature (0.01), therefore, it should be relatively accurate, in theory. Despite this, our output has a spelling mistake.

# In[17]:


print(extend_text("To be or not to be", temperature=1))


# You can see in the cell above that the text is following the format of a script, due to "provost:"

# In[18]:


print(extend_text("To be or not to be", temperature=100))


# You can see from the cell above that when the temperature is set to 100, the results are gibberish.

# In[19]:


print(extend_text("To be or not to be", temperature=2))


# Even the temperature at 2 does not make sense.

# In[28]:


print(extend_text("Shall I compare thee to a", temperature=0.001))


# In[23]:


print(extend_text("Shall I compare thee to a", temperature=0.5))


# You can see from the two cells above that they are not quite conherant sentences. Although they may sound similar to Shakespeare. There is not much meaning to grasp from these sentences. These attempts are suggesting that the model is not successful enough at "understanding" english. 

# <img src="oh-romeo-romeo.jpeg" alt="O Romeo, Romeo, wherefore art thou Romeo?" width=50%>
# Image sourced from: https://makeameme.org/meme/oh-romeo-romeo-cd90fc5a05

# In[35]:


print(extend_text("O Romeo, Romeo, wherefore art thou Romeo?", temperature=1))


# The first sentence here was better... then it mentioned tables. 
# 
# <img src="table.jpg" alt="image of table" width=50%>
# Image source: ikea.com

# In[38]:


print(extend_text("O Romeo, Romeo, where", temperature=0.5))


# It seems to be slightly better at finishing half sentences.

# ## Lets test it for finishing lyrics!

# In[31]:


print(extend_text("I'd meet you where the spirit meets the bones", temperature=0.01, n_chars=50))


# "I'd meet you where the spirit meets the bones" is a Taylor Swift lyric. As discussed earlier, neural networks like this could have the potential to stimulate creativity and provide inspiration. But, this attempt has been unsuccessful.  
# 
# <img src="Taylor_Swift.png" alt="Evermore album cover">
# Image Source: Taylor Swift and  Republic Records

# In[44]:


print(extend_text("I'd meet you where the spirit meets", temperature=0.5))


# Shortening the text and adjusting the temperature did not help improving the output.

# ## Comparison to Hugging Chat (AI chatbot)

# In[46]:


print(extend_text("In love's embrace, hearts entwine", temperature=0.5))


# In[47]:


print(extend_text("In love's embrace, hearts entwine", temperature=0.75)) #An example with this temperature as this is the temperature Hugging Chat uses


# <img src="hugging-chat.png" alt="Input to Hugging Chat: Expand on this text, in the style of Shakespeare's writing: In love's embrace, hearts entwined. Output:Oh, how sweet it is to be ensnared within the tender clutches of love's gentle embrace. For when two hearts are entwined, as if by fate's design, their union brings forth a joy so pure and true, that all else fades into insignificance.">
# 

# This example highlights how an AI chatbot preforms significantly better at this task than our model. This suggests that it may be more benefitial for humanities schoolars to use an open-access, pre-trained text generator. As as we have seen, our model has many weaknesses.

# ## Final Overview
# 
# #### Evaluation of the result of the generative AI code:
# While the code generated incoherent text. The training data size needs to be taken into consideration, as well as the time spent training the model. Both of these could have been significantly increased to produce improved results. When this is considered, it is arguabley impressive that the text formed sentences which look like they could have been written by Shakespeare, at a first glance. This is because the model managed to copy Shakespeare's rhythm. Additionally, one could argue that the spelling mistakes and made up words is the model are mimicking Shakespeare's inventiveness with language. As he was known for creating new words which are derived from real words. In my personal opnion, the model is not sophisticated enough to purposefully do this.
# 
# #### How you might apply your code to your own data:
# Keeping with the theme of relating Shakespere's work to pop songs, I would be interested in seeing if the model could be trained on a database of pop music. This could be interesting, as perhaps in this case, the model would be able to effectively produce song lyrics in the style of shakespeare. Then it could finally be used as a method of stimulating creativity! In order to do this, I would have to gather a lot of song lyrics into a text file and prepare it the same way we prepared the shakespeare data (by turning them into Tokens!).
# 
# 
# #### Ethical concerns 
# There are intellectual property concerns involved with using someone else's material (in this case songs) to train a model. Additionally, if the AI was to produce a song which effectively mimicked their style, this could be reason for concerned as that is their creative style that they have harnessed and defined. There are less concerns with Shakespeare as his work was produced centuries ago (meaning there are no current copyright laws on his work). Despite this, there would still be reason for ethical concerns if text was effectively generated in his style and it was presented to the public as newly discoved work, written by Shakespeare himself. 

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/liadmagen/NLP-Course/blob/master/exercises_notebooks/01_LM_NLP_python_basics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Python Basics
# 
# In this exercise, we'll explore python NLP capabilities with the help of the package `nltk`.
# 
# By the end of the exercise, you will:
# * be introduced to the nltk package and its functionality
# * understand the basics of text analysis, and know how to approach this unstructured data.
# * Understand the terms 'n-gram' & 'collocation'

# We are going to use the package `NLTK` - 'Natural Language Toolkit' (https://www.nltk.org/).
# 
# NLTK is a great package for research and for learning. However, it isn't recommended for production use and for real-world applications, as it isn't fast enough and therefore doesn't scale.

# # Setup

# In[ ]:


import random

import nltk


# In[ ]:


# nltk.download('book')


# In[ ]:


from nltk.book import *


# # A Closer Look at Python: Texts as Lists of Words

# We will use the great book 'Moby Dick' by Herman Melville, as our learning experiment playground.
# 
# The book is already tokenized and stored as a list of these tokens, under the variable `text1`.
# 
# We start - as always - with looking at our data. 
# 
# Let's peek at the first 100 words:

# In[ ]:


# print(text1[:100])


# Pay attention that punctuations are also conisdered as a `token`.

# ## Exercise #1: Show the last 23 tokens in the book:

# In[ ]:
# print(text1[-23:])

### your turn: Write a code that shows the last sentence (23 tokens) of the book



### End


# ## Lists vs Sets
# 
# In python, an ordered set, with repetition, is defined as a List, and marked with sqaured brackets [].
# 
# An unordered set, where repetitions are discarded, is defined with regular brackets: ().
# 
# When converting the list into a set, we get the vocabulary of the corpus:

# In[ ]:


vocab = set(text1)

# We can't get the 'last 25 words', since there is no order... 
# But we can convert it into a list first, and even sort it
list(sorted(vocab))[-50:]


# ## Exercise #2: Vocabulary Length
# 
# How many words does our vocabulary have?

# In[ ]:
print("number of different word in the book : ", len(vocab))

### your turn: Write a code that prints the size of Moby Dick book's vocabulary



### End


# # Text Analysis: Frequency Distribution

# nltk is a library with many research tools for probabilistic information. 
# 
# For example, it includes a function, `FreqDist`, that return the probability of the occurance of a word in a text:
# 
# http://www.nltk.org/api/nltk.html?highlight=freqdist#module-nltk.probability

# In[ ]:


## write a code that calculate the frequency of words in text1 and prints the top 50 common ones.
## How many times do the words 'with', 'Moby', 'fish' and 'whale' appear in the book?
## hint - fdist is a smart dictionary that already has methods for these tasks, 
## such as .most_common() 

fDist = FreqDist(text1)
print(fDist)
print(dir(fDist))
print("most commun words are : ")
print(fDist.most_common(50))

print("nb of occurence of word 'with' :", fDist["with"])
print("nb of occurence of word 'Moby' :", fDist["Moby"])
print("nb of occurence of word 'fish' :", fDist["fish"])
print("nb of occurence of word 'whale' :", fDist["whale"])
# for having a visualisation of the distribution :
# fDist.plot()

### End


# Some of the common words are actually punctuations and 'stop-words'. They don't really help us with our analysis of the text, and therefore should be ignored.
# 
# Luckily, NLTK supplies a list of stop words, and python has the punctuation built in into the string package:

# In[ ]:


from nltk.corpus import stopwords

print(stopwords.words('english'))


# In[ ]:


import string

print(string.punctuation)


# In[ ]:





# In[ ]:


### Calculate the top 50 frequennt words, without stop words or punctuation.
### Hint: Working with sets, is much faster than with lists.
### like the mathematical sets, a python Set has an ability to intersect, detect subsets and even subtract:
### See more in here: https://docs.python.org/3.8/library/stdtypes.html#set


unwantedToken= set(string.punctuation) | set(stopwords.words('english')) | set(["--"])
cleanedText1 = [str.lower(w) for w in text1 if w not in unwantedToken]
newFDist = FreqDist(cleanedText1 )
print(newFDist)
print("new top 50 commun words")
print(newFDist.most_common(50))

###


# FreqDist can be used even further. Let's analyse the text by the word length.
# 
# Using python 'list-comprehension' (https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) we can easily get a list of all the words by their lengths:

# In[ ]:


# For convenience of reading, showing here only the first 30
print([len(w) for w in text1][:30])


# ## Exercise #3 (Advanced): length frequency

# In[ ]:


### Write a code to calculate the frequency of the length of words in `text`. 
lenFrequencyList = [len(w) for w in cleanedText1]
lenFDist = FreqDist(lenFrequencyList)
print("top 20 word length")
print(lenFDist.most_common()[:20])
sortedText1 = cleanedText1
sortedText1.sort(key=lambda item: (-len(item), item), reverse=True)
print("top 20 longuest word")
top20 = sortedText1[-20:]
print(top20)
print("number of time that the longuest words appear")


### How often do the 20 most lengthiest words appear in the text?
for w in top20:
  print(w + " : " + str(fDist[w]))
### Find out what those 20 words are (Sort them alphabetically if two have the same length).
print(top20)




### End


# # Text Analysis: n-grams and collocation

# As we learnt in class, a word is not always a single token. In the case of 'New York', 'ice cream', 'red wine', etc., a single word meaning is different than the combined one.
# 
# A **collocation** is a sequence of words that occur together unusually often.
# 
# An `n-gram` is a sequence of a size of 'n' of tokens (i.e. words):
# 
# * When n=1: it is called **unigram**
# * When n=2: it is called **bigram**
# * When n=3: it is called **trigram** ...
# * When n>3: it is just called an **n-gram** with the size of 4.
# 
# 
# NLTK has two functions: `bigrams` and `collocations`

# In[ ]:


list(bigrams([1,2,3,4,5]))


# In[ ]:


## Bigrams generates bi-grams from the text: every two words would be collected together.
list(bigrams(text1))[:20]


# In[ ]:


text1.collocations()


# # Python and NLP

# Python has many strong capabilities, built in, when it comes to string and text procesing, in combined with the list comprehension.
# 
# Here are some examples of filtering the word list:

# In[ ]:


# Get all the words that ends with 'ableness', sorted:
sorted(w for w in set(text1) if w.endswith('ableness'))


# In[ ]:


# Get all the words that contains 'orate', sorted:
sorted(term for term in set(text1) if 'orate' in term)


# In[ ]:


# Get all the words which their first letter is capitalized:
sorted(item for item in set(text1) if item.istitle())


# And there are more. if `wrd` is a string, then, for example:
# 
# * `wrd.islower()` will return true if the word is all lowercase
# * `wrd.isalpha()` will return true if all the character in the string are letters
# 
# and there are also: `wrd.startswith('str')`, `wrd.isdigit()`, `wr.isalnum()`
# and more.

# ## Exercise #4: Functions and substrings search

# In[ ]:


from typing import List

### Exercise: 

def detect_string(tokens: List[str], search_str: str, search_position: int = 0) -> List[str]:
  """Returns a sorted list of the vocabulary tokens which match the search conditions

  params:
    tokens: a document tokens list.
    search_str: a string to search in the token list 
    search_position: one of the following:
      0 - anywhere in the string
      1 - searches for the string at the beginning of the token
      2 - searches for the string at the end of the token
  """
  ### Fill in this function to returns the result of searching for the  
  ### given string "search_str" in the token vocabulary "tokens", according to 
  ### the position parameter, as explained in the docstring
  if search_position == 0:
    return sorted([w for w in set(tokens) if search_str in w])
  
  if search_position == 1:
    return sorted([w for w in set(tokens) if w.startswith(search_str) ])
    
  if search_position == 2:
    return sorted([w for w in set(tokens) if w.endswith(search_str) ])
    
  

### 

# In[ ]:


### Test:
assert detect_string(text1, 'tably', 2) == ['comfortably',
 'discreditably',
 'illimitably',
 'immutably',
 'indubitably',
 'inevitably',
 'inscrutably',
 'profitably',
 'unaccountably',
 'unwarrantably']


# In[ ]:


### Test:
assert detect_string(text1, 'argu', 1) == ['argue', 'argued', 'arguing', 'argument', 'arguments']


# In[ ]:


### Test:
assert detect_string(text1, 'arg', 2) == []


# In[ ]:


### Test
assert detect_string(text1, 'larg') == ['enlarge',
 'enlarged',
 'enlarges',
 'large',
 'largely',
 'largeness',
 'larger',
 'largest']


# In[ ]:





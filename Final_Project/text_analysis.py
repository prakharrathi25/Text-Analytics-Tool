# File containing all the analysis functions for the web app

# Standard Libraries
import os 
import re 
import string 
import numpy as np
from collections import Counter

# Text Processing Library 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud
from gensim import utils
import streamlit as st
import pprint
import gensim
import gensim.downloader as api
import warnings
import spacy
from spacy import displacy
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import tempfile
warnings.filterwarnings(action='ignore')


# Data Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns
import spacy_streamlit
from PIL import Image


# Constants 
STOPWORDS = stopwords.words('english')
STOPWORDS + ['said']


# Text cleaning function 
def clean_text(text):
    '''
        Function which returns a clean text 
    '''    
    # Lower case 
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d', '', text)
    
    # Replace \n and \t functions 
    text = re.sub(r'\n', '', text)
    text = text.strip()
    
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove Stopwords and Lemmatise the data
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text)
    
    return text

# Create a word cloud function 
def create_wordcloud(text, image_path = None):
    '''
    Pass a string to the function and output a word cloud
    
    ARGS 
    text: The text for wordcloud
    image_path (optional): The image mask with a white background (default None)
    
    '''
    
    st.write('Creating Word Cloud..')

    text = clean_text(text)
    
    if image_path == None:
        
        # Generate the word cloud
        wordcloud = WordCloud(width = 600, height = 600, 
                    background_color ='white', 
                    stopwords = STOPWORDS, 
                    min_font_size = 10).generate(text) 
    
    else:
        mask = np.array(Image.open(image_path))
        wordcloud = WordCloud(width = 600, height = 600, 
                    background_color ='white', 
                    stopwords = STOPWORDS,
                    mask=mask,
                    min_font_size = 5).generate(text) 
    
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation = 'nearest') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()  


# Function to plot the ngrams based on n and top k value
def plot_ngrams(text, n=2, topk=15):
    '''
    Function to plot the most commonly occuring n-grams in bar plots 
    
    ARGS
    	text: data to be enterred
    	n: n-gram parameters
    	topk: the top k phrases to be displayed

    '''

    st.write('Creating N-Gram Plot..')

    text = clean_text(text)
    tokens = text.split()
    
    # get the ngrams 
    ngram_phrases = ngrams(tokens, n)
    
    # Get the most common ones 
    most_common = Counter(ngram_phrases).most_common(topk)
    
    # Make word and count lists 
    words, counts = [], []
    for phrase, count in most_common:
        word = ' '.join(phrase)
        words.append(word)
        counts.append(count)
    
    # Plot the barplot 
    plt.figure(figsize=(10, 6))
    title = "Most Common " + str(n) + "-grams in the text"
    plt.title(title)
    ax = plt.bar(words, counts)
    plt.xlabel("n-grams found in the text")
    plt.ylabel("Ngram frequencies")
    plt.xticks(rotation=90)
    plt.show()


# Function to return POS tags of a sentence 
def pos_tagger(s):
    
    # Define the tag dictionary 
    output = ''
    
    # Remove punctuations
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(s))
    for tag in tagged_sentence:
        out = tag[0] + ' ---> ' + tag[1] + '<br>'
        output += out

    return output
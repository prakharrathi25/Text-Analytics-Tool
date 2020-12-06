# Important libraries 
import re
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Collect stopwords 
STOPWORDS = stopwords.words('english')

# Function to clean the model input text 
def clean_text_spam(text):
    
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    
    # Remove punctuation 
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove Stopwords 
    text = [word for word in text.split() if word.lower() not in STOPWORDS]
    words = ""
    
    # Stemming 
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    
    return [words]

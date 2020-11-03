# Import necessary libraries
import joblib
import re
import streamlit as st
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from io import StringIO
from PIL import  Image
from rake_nltk import Rake

# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Import the custom modules 
import spam_filter as sf
import text_analysis as nlp

# Describing the Web Application 

# Title of the application 
st.title('Text Analysis Tool\n', )
st.subheader("by Prakhar Rathi")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
["Home", "Email Spam Classifier", "Keyword Sentiment Analysis", 'Word Cloud', 'N-Gram Analysis', 'Part of Speech Analysis', 'Similarity Analysis'])

st.set_option('deprecation.showfileUploaderEncoding', False)

if option == 'Home':
	st.write(
			"""
				## Project Description
				This is a complete text analysis tool developed by Prakhar Rathi. It's built in with multiple features which can be accessed
				from the left side bar.
			"""
		)

# Word Cloud Feature
elif option == "Word Cloud":

	st.header("Generate Word Cloud")
	st.subheader("Generate a word cloud from text containing the most popular words in the text.")

	# Ask for text or text file
	st.header('Enter text or upload file')
	text = st.text_area('Type Something', height=400)
	   
	# Collect data from a text file
	# file_text = st.file_uploader("Choose a text file", accept_multiple_files=False)	
	
	# if file_text is not None:

	# 	# Read as string data 
	# 	text = file_text.read()
	# 	text = text.decode("latin-1")

	# 	# Check if the length was recognized correctly 
	# 	if len(text) == 0:
	# 		st.write("**Error**: Please upload the text file again")
	# 	else:
	# 		st.write(f"Identified {len(text)} characters")

	# Upload images 
	mask = st.file_uploader('Use Image Mask', type = ['jpg'])

	# Add a button feature
	if st.button("Generate Wordcloud"):

		# Generate word cloud 
		st.write(len(text))
		nlp.create_wordcloud(text, mask)
		st.pyplot()

# N-Gram Analysis Option 
elif option == "N-Gram Analysis":
	
	st.header("N-Gram Analysis")
	st.subheader("This section displays the most commonly occuring N-Grams in your Data")

	# Ask for text or text file
	st.header('Enter text below')
	text = st.text_area('Type Something', height=400)

	# Parameters
	n = st.sidebar.slider("N for the N-gram", min_value=1, max_value=8, step=1, value=2)
	topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

	# Add a button 
	if st.button("Generate N-Gram Plot"): 
		# Plot the ngrams
		nlp.plot_ngrams(text, n=n, topk=topk)
		st.pyplot()

# Spam Filtering Option
elif option == "Email Spam Classifier":

	st.header("Enter the email you want to send")

	# Add space for Subject 
	subject = st.text_input("Write the subject of the email", ' ')

	# Add space for email text 
	message = st.text_area("Add email Text Here", ' ')

	# Add button to check for spam 
	if st.button("Check"):

		# Create input 
		model_input = subject + ' ' + message
		
		# Process the data 
		model_input = sf.clean_text_spam(model_input)

		# Vectorize the inputs 
		vectorizer = joblib.load('Models/count_vectorizer_spam.sav')
		vec_inputs = vectorizer.transform(model_input)
		
		# Load the model
		spam_model = joblib.load('Models/spam_model.sav')

		# Make the prediction 
		if spam_model.predict(vec_inputs):
			st.write("This message is **Spam**")
		else:
			st.write("This message is **Not Spam**")
		
# POS Tagging Option 
elif option == "POS Tagging":
	st.header("Enter the statement that you want to analyze")

elif option == "Keyword Sentiment Analysis":

	st.header("Sentiment Analysis Tool")
	st.subheader("Enter the statement that you want to analyze")

	text_input = st.text_area("Enter sentence", height=50)

	# Model Selection 
	model_select = st.selectbox("Model Selection", ["Naive Bayes", "SVC", "Logistic Regression"])

	if st.button("Predict"):
		
		# Load the model 
		if model_select == "SVC":
			sentiment_model = joblib.load('Models/SVC_sentiment_model.sav')
		elif model_select == "Logistic Regression":
			sentiment_model = joblib.load('Models/LR_sentiment_model.sav')
		elif model_select == "Naive Bayes":
			sentiment_model = joblib.load('Models/NB_sentiment_model.sav')
		
		# Vectorize the inputs 
		vectorizer = joblib.load('Models/tfidf_vectorizer_sentiment_model.sav')
		vec_inputs = vectorizer.transform([text_input])

		# Keyword extraction 
		r = Rake(language='english')
		r.extract_keywords_from_text(text_input)
		
		# Get the important phrases
		phrases = r.get_ranked_phrases()

		# Make the prediction 
		if sentiment_model.predict(vec_inputs):
			st.write("This statemen is **Positve**")
		else:
			st.write("This statemen is **Negative**")

		# Display the important phrases
		st.write("These are the **keywords** causing the above sentiment:")
		for i, p in enumerate(phrases):
			st.write(i+1, p)
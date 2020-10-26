
# Import necessary libraries
import joblib
import re
import streamlit as st
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from PIL import  Image

# Import the custom modules 
import spam_filter as sf

# Describing the Web Application 

# Title of the application 
st.title('Text Analysis Tool\n', )
st.subheader("by Prakhar Rathi")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display, use_column_width = True)

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
['Home', "Email Spam Classifier",  'Word Cloud', 'N-Gram Analysis', 'Part of Speech Analysis', 'Similarity Analysis'])

st.set_option('deprecation.showfileUploaderEncoding', False)

# st.header('Enter text or upload file')
# text = st.text_area('Type Something', height = 400)
    
# file_text = st.file_uploader('Text File', encoding = 'ISO-8859-1')
    
# if file_text!=None:
#     text = file_text.read()

if option == 'Home':
	st.write(
			"""
				## Project Description
				This is a complete text analysis tool developed by Prakhar Rathi. It's built in with multiple features which can be accessed
				from the left side bar.
			"""
		)

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
		vectorizer = joblib.load('Models/count_vectorizer_sentiment.sav')
		vec_inputs = vectorizer.transform(model_input)
		
		# Load the model
		spam_model = joblib.load('Models/spam_model.sav')

		# Make the prediction 
		if spam_model.predict(vec_inputs):
			st.write("This message is **Spam**")
		else:
			st.write("This message is **Not Spam**")
		

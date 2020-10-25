
# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from gensim import utils
import pprint
import gensim
import warnings
import tempfile
from PIL import  Image
import urllib

# Describing the Web Application 

# Title of the application 
st.title('Text Analysis Tool\n', )
st.subheader("by Prakhar Rathi")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display, use_column_width = True)

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
['Home ', "Email Spam Classifier",  'Word Cloud', 'N-Gram Analysis', 'Part of Speech Analysis', 'Similarity Analysis'])

st.set_option('deprecation.showfileUploaderEncoding', False)

# st.header('Enter text or upload file')
# text = st.text_area('Type Something', height = 400)
    
# file_text = st.file_uploader('Text File', encoding = 'ISO-8859-1')
    
# if file_text!=None:
#     text = file_text.read()

if option == 'Home':
	st.write(
			"""
				Enter some markdown 
			"""
		)

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
		
	
	# Process the text 

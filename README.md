# Text Analytics Tool



This is an application that automates the process of text analysis with a user friendly GUI. :iphone: It has been implemented using Python and deployed with the Streamlit package. 

### Text Analytics :bar_chart:

Text analytics is the automated process of translating large volumes of unstructured text into quantitative data to uncover insights, trends, and patterns. Combined with data visualization tools, this technique enables companies to understand the story behind the numbers and make better decisions. It is an artificial intelligence (AI) technology that uses natural language processing (NLP) to transform the unstructured text in documents and databases into normalized, structured data suitable for analysis or to drive machine learning (ML) algorithms. \cite{b1}

### Need to automate text analytics :robot_face: 

Creating an analysis app has a lot of benefits. Anyone who wants to get started with data exploration does not have to write pipelines to visualise their data and begin modelling. An application like this reduces the time between Exploratory Data Analysis (EDA) and model building. I have built and integrated many tools for data visualisation and analysis. The user interface is an important aspect of my application. The front end application was built using Streamlit that makes testing of machine learning applications very easy. The user interface offers multiple tools to choose from using a left dropdown option. :nut_and_bolt: 


## Tools that can be accessed from the application

### Word Cloud Generator :cloud: 

A word cloud is a data visualization method that forms an integral aspect of text analytics. It is a text representation in which words are shown in varying sizes depending on how often they appear in our corpus. The words with higher frequency are given a bigger font and stand out from the words with lower frequencies. The pre-requisite to this task is a set of text cleaning and processing functions. 

I have integrated these functions as a part of my tool which run as soon as the text is added on the application. Another feature I have added is the masking feature where the output can defined in a particular shape based on the image that is provided to the generator. 

<p  style="text-align:center;">
  <img style="height:500px" src="https://i.imgur.com/ra8ittj.png" alt="word-cloud output"/>
</p>


### N-Gram Analysis

n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus. 

In this analysis, I am trying to identify the most commonly occurring n-grams. While word cloud focusses on singular words, this analysis can yield multiple phrases instead of just one. This is used to analyse writing style of a person to see how repetitive it is and what patterns occur in the writing a lot. I have also provided UI features on top of the analysis. Users can change the value of n and the count of top phrases which they want. 

<p style="text-align:center;">
  <img style="height:600px" src="https://i.imgur.com/40UdN8B.png" alt="n-gram"/>
</p>

### Spam Analysis

I have built a model which given an email input (subject + message) would output whether the email will be classified as spam or not.Models like Naive Bayes use a probabilistic approach towards solving the problem of spam classification and prove to be very effective. Another application of this can be when people are sending out emails so they can check whether their email sounds like a spam or a relevant email. 



## References

[1] Brimacombe, J. M. (2019, December 13). What is text mining, text analytics and natural language processing? Linguamatics. [Access.](https://www.linguamatics.com/what-text-mining-text-analytics-and-natural-language-processing)

[2] Metsis, V., Androutsopoulos, I. and Paliouras, G. (2006) Spam Filtering with Naive Bayes—which Naive Bayes? Third Conference on Email and Anti-Spam (CEAS), Mountain View, July 27-28 2006, 28-69.

[3] Jurafsky, D. (2000). Speech & language processing. Pearson Education India.

[4] Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. 1993. Building a large annotated corpus of English: the penn treebank. Comput. Linguist. 19, 2 (June 1993)

[5] Weischedel, Ralph, et al. OntoNotes Release 5.0 LDC2013T19. Web Download. Philadelphia: Linguistic Data Consortium, 2013.

[6] Mihalcea, Rada & Tarau, Paul. (2004). TextRank: Bringing Order into Text.
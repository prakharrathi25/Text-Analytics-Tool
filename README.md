# Text Analytics Tool



This is an application that automates the process of text analysis with a user friendly GUI. :iphone: It has been implemented using Python and deployed with the Streamlit package. Here's a look at the application page.

<p style="align:center">
    <img src="https://i.imgur.com/hoVozCc.jpg">
</p>



### Text Analytics :bar_chart:

Text analytics is the automated process of translating large volumes of unstructured text into quantitative data to uncover insights, trends, and patterns. Combined with data visualization tools, this technique enables companies to understand the story behind the numbers and make better decisions. It is an artificial intelligence (AI) technology that uses natural language processing (NLP) to transform the unstructured text in documents and databases into normalized, structured data suitable for analysis or to drive machine learning (ML) algorithms. [1]

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

#### Dataset 

The dataset that I used for this analysis is the publicly available dataset, first described in the 2006 conference paper "Spam Filtering with Naive Bayes?" by V. Metsis, I. Androutsopoulos, and G. Paliouras. [2]

The data can found in the `Datasets` folder in the project directory named as *`spam.txt`* which was then cleaned and converted to *`spam_data.csv`*.

#### Result

All the processing and model building can be found in the `Notebooks` folder under the name *`SpamClassifier.ipynb`*. 

The model performs extremely well and gives an accuracy of **98.7%**. This is a very promising result. The application output of the same can be seen here. 

<p style="text-align:center;">
  <img src="https://i.imgur.com/9qS6aLH.jpg" alt="spam classifier"/>
</p>


### Keyword Sentiment Analysis :key: 

This is a two-part task. The first part is sentiment extraction which given a sentence, identifies whether it is positive of negative sentence. The second phase is identifying which keyword in the sentence is causing that emotion. The keywords are extracted based on what the contribution of the word is to the final sentiment value of the statement.

#### Dataset 

The models were trained on a large corpus first to make them more robust and generalizable. The data was collected using a scraper that I built to scrape Google Play Store Reviews. The scraper collected roughly 12,500 reviews. The implementation of the scraper can be found in the Notebooks folder under the name of *`sentiment_analysis_data_collection.ipynb`*. 

#### Results 

I implemented 4 algorithms for sentiment analysis and the results are as follows. The implementation of the first three algorithms can be found in the `sentiment_analysis_lite.ipynb` notebook while BERT can be found in the `sentiment_analysis_BERT.ipynb` notebook. The results of the different algorithms can be seen below. 


| Algorithm           | F1-Score |
| ------------------- |:--------:|
| Logistic Regression |   0.82   |
| Naive Bayes         |   0.81   |
| SVC                 |   0.81   |
| BERT                |   0.84   |



The application output can be seen in the image below.

<p style="text-align:center">
  <img src="https://i.imgur.com/4oJTzRY.jpg" alt=""/>
</p>

### Text Summarization 

Given a large corpus, we need to summarize it into a summary which covers key points and conveys the exact meaning of the original document to the best of our ability. I have created an extractive text summary tool which  takes the same words, phrases, and sentences from the original summary, thereby selecting the most important sentences in the given text. There are different methods of estimating the most important sentences in a large text. The number of sentences is calculated using a compression ratio. 

Sentence Count = η * Total Sentence Count

where η is the compression ratio between 0 to 1.  I have used different forms of text-ranking algorithm which builds a graph related to the text. In a graph, each sentence is considered as vertex and each vertex is linked to the other vertex. These vertices cast a vote for another vertex. The importance of each vertex is defined by the higher number of votes. [6] The algorithms I used can be found in the `Text_Summarization.ipynb` notebook. 

### POS Tagging

In traditional grammar, part of speech (POS) is the category of words  that have similar grammatical properties and similar syntactic behaviour. Parts of speech are useful because they reveal a lot about a word and its neighbors. Knowing whether a word is a noun or a verb tells us about likely neighboring words. The activity of assigning POS tags to words in a corpus is known as POS Tagging. I intended to do the tagging based on the [Penn-Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) tagset [4]. It is a set of 45-tag set which has been used to label many corpora. 

The implementation can be found in the `text_analysis.ipynb` notebook. The result in the application can be seen below. The tags can be compared with the tagset above. 

![](https://i.imgur.com/MyxlJ5a.jpg)

### Named Entity Recogniton

Named entity recognition (NER) is a very important aspect of information recognition which is further used in knowledge graphs, chatbots and many other implementations. The task involves classifying text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values etc.

The library I use for this is SpaCy which performs named entity annotation. I have also added a selection box from where we can choose which named entities we want to display. SpaCy's Named Entity Recognition model has been trained on the OntoNotes5 corpus [5]. It was released through the OntoNotes5 project, a collaborative effort between BBN Technologies, the University of Colorado, the University of Pennsylvania and the University of Southern Californias Information Sciences Institute. Their goal was to annotate a large corpus comprising of different types of text (news, conversational telephone speech, weblogs etc.) with structural information and shallow semantics. 

Once the user inputs a sentence, it is tokenized, tagged and then add to the NER function which helps in selection and visualisation of different entities. The result can be seen below. 

![](https://i.imgur.com/TsxAKoh.jpg)


## Tech Stack 

1. Python 
2. nltk 
3. Streamlit
4. Scikit-learn
5. Hugging Face - Tokenizer, Transformer
6. PyTorch

## How to Run 

1. Clone the repository
2. To install all the libraries (preferably in a VM): `pip install -r requirements.txt`
3. To run the app: `streamlit run app.py`

## References

[1] Brimacombe, J. M. (2019, December 13). What is text mining, text analytics and natural language processing? Linguamatics. [Access.](https://www.linguamatics.com/what-text-mining-text-analytics-and-natural-language-processing)

[2] Metsis, V., Androutsopoulos, I. and Paliouras, G. (2006) Spam Filtering with Naive Bayes—which Naive Bayes? Third Conference on Email and Anti-Spam (CEAS), Mountain View, July 27-28 2006, 28-69.

[3] Jurafsky, D. (2000). Speech & language processing. Pearson Education India.

[4] Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. 1993. Building a large annotated corpus of English: the penn treebank. Comput. Linguist. 19, 2 (June 1993)

[5] Weischedel, Ralph, et al. OntoNotes Release 5.0 LDC2013T19. Web Download. Philadelphia: Linguistic Data Consortium, 2013.

[6] Mihalcea, Rada & Tarau, Paul. (2004). TextRank: Bringing Order into Text.

## Contact 

For any feedback or queries, please reach out to prakharrathi25@gmail.com.

Note: The project is only for education purposes, no plagiarism is intended.

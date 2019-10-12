### Sentiment-and-Stance-analysis-in-twitter-along-with-detailed-temporal-analysis

Detection of Sentiment and Stance analysis on dyanamic twitter data

## 1. Introduction

Public opinion in a traditional way is a costly and a time consuming process which can require contacting many people. Twitter being a public opinion platform, is the best source for collecting textual data to minimize the difficulties associated with modeling public behaviour. My motive is to predict the ​Stance​ of people opinion on ​Gun Control Law in United States Of America​ expressed on the social media platform Twitter in a ten day period from 19th september to 28th september. I want to capture the trend of public opinion and how it changed  over time. 

## 2. Why is it important ?

Understanding public opinion would help in bringing more reformed amendments in the gun laws. In order to achieve this the modern computation techniques of Machine Learning , Natural Language Processing and Deep learning over a sample of 13,000 tweets made by individuals in USA that contains one of predetermined relevant keywords. Tweets are downloaded using the Twitter streaming API and labelled using predetermined keywords. Data is prepared for analysis through data preprocessing and data cleaning steps. Topic modelling technique is used to get an overview of public opinion on gun laws in America and Temporal analysis of public opinion over the period of 10 days.  Features were extracted using NLP techniques to feed as input to machine learning classifiers for text classification into two stance: pro gun and anti gun. Furthermore deep learning algorithms would we applied for better accuracy of stance prediction. 

## 3. Methodology
 
The detailed methodology for analysing public opinion incorporates: 

  ### 3.1 Data collection

  Twitter platform is used to collect the tweets related to Gun Control law. For data collection Twitter Streaming API tweepy is used which provides twitter feed in a machine readable JSON format. The popular hashtags, using  trending api  on twitter related to anti-gun and pro-gun are found and a list of hashtags is prepared separately for anti gun hashtags and progun hashtags. These hashtags were used to download the tweets from the streaming api. Data was collected for a period of 10 days , from 19 sep to 28 Sep and consist with  13000 tweets which contain at least one or more gun  related hashtags.  Data was uploaded to MongoDb database which is suitable to  handle the json formatted tweet data. The use of MongoDb not only helps in storing the data is a structured format but also eliminates the duplicate entries and also simplifies different queries. Now,we extract the id , date and full_text field for each tweet and stored it in an excel file as initially we want to perform only text analysis. 
  
  ### 3.2 Data pre-processing
    
  After getting the dataset which contain tweets related to gun control law, the next step is to clean the data to provide the input for text classification model. Accuracy of feature extraction also greatly depends on the quality of text data. Following are the steps perform for data cleaning.

   #### 3.2.1 Removal of Punctuation marks and symbols
   
   #### 3.2.2 Tokenization and Removal of Stop Words 
 
  ### 3.3 Feature extraction
  
   #### 3.3.1 Sentiment Score
   
  “The sentiment of a piece of text is its ​positivity ​ or ​negativity ​ .”In order to calculate the sentiment of a piece of text, we split it into individual words.We have a database of words, each with a "score" to determine how positive or negative it is.The higher the score, the more positive the word and similarly opposite for negative words.Not every word in a piece of positive text will be positive, and not every word will be negative, but by feeding the number of identified words and their scores into our algorithm, we end up with a score for the sentiment of the text. Thus we classify the text as positive, negative and neutral based on these scores. For this we use text.blob library which gives ​polarity score​ of each text. The polarity score is a float value range:​[-1,1]​. 
  ➢ Negative sentiment -  score<0 
  ➢ Positive sentiment - score>0 
  ➢ Neutral sentiment - otherwise
  
   #### 3.3.2  POS Tagging 
  
  A POS tag (or part-of-speech tag) is a special label assigned to each token (word) in a text corpus to indicate the part of speech and often also other grammatical categories such as tense, number (plural/singular), case etc. POS tags are used in corpus searches and in text analysis tools and algorithms. POS tags are used from Spacy Library to classify the label of our tweet whether it is ‘for’ or ‘against’ the topic. 
  
   #### 3.3.3 TF-IDF
  
   ​TFIDF, short for ​term frequency–inverse document frequency​, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. ● TF(w) = (Number of times term w appears in a document) / (Total number of terms in the document) ● IDF(w) = log_e(Total number of documents / Number of documents with term w in it) Each word or term has its respective TF and IDF score. The product of the TF and IDF scores of a term is called the TF*IDF weight of that term.The higher the TF*IDF score (weight), the rarer the term and vice versa. 

### 3.4 Topic Modelling

Topic Modeling​ ​is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. 

### 3.5 Named Entity Recognizition

Named Entity Recognition​ is probably the first step towards information extraction that seeks to locate and classify named entities in text into predefined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. ,  A count of these entities were extracted from each tweet is extracted using the NLP library  ​Spacy. 

### 3.6 Temporal study

Tweets were collected over a time period of 10 days, starting 19 September,2019 to 28 September,2019.An analysis was done to see the trend of public opinion over this period of time. Public opinion is categorized into two categories namely: Pro and Anti. 
 
Users  which termed as Anti gun are speaking in favour of gun law amendment and the ones which are against any amendment in gun laws are termed as Pro gun. 
 
So, a time series analysis is done to see the trend in opinions in favour and against the gun laws. A bar chart plotting the tweet count of the  two categories is made with the help of pandas group by function and the charting and plotting library : Matplotlib and Seaborn.  

### 3.7 Classification model
 
After gathering all the features , Sentiment scores, POS, NER,TF IDF score,hashtags count ,we then split the data into training set and testing set.Machine learning  Classifiers used: SVM,Random Forest,KNN,Logistic . Different accuracies were obtained after incorporating different features. 

● With PoS and sentimental score using logistic regression 60.1 percent. 
● With POS and sentimental score using random forest classifier accuracy is 65.4 %. 
● With TF idf using logistic regression the accuracy is 95.9 percent. 
● With TF idf using random forest classifier accuracy is 95.1 %. 
 
### 3.8 Evaluation

Precision​ - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
Precision = TP/TP+FP 
Recall (Sensitivity) ​- Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. Recall = TP/TP+FN 
F1 score​ - F1 Score is the weighted average of Precision and Recall.  
F1 Score = 2*(Recall * Precision) / (Recall + Precision).

### 3.9 K-Cross Validation 

Cross-validation is a statistical method used to estimate the skill of machine learning models. Cross Validation is used to assess the predictive performance of the models and and to judge how they perform outside the sample to a new data set also known as test data. K-Fold CV is where a given data set is split into a ​K ​ number of sections/folds where each fold is used as a testing set at some point. 
 
### 3.10 Visualization 

Please refer Visualizations folder.

## 4 Results 

Graphs are in Visualization folder.

A peak of anti-gun feeling on the day of "shooting related incident" is observed which quickly falls to pre-event levels. More surprisingly the analysis shows a peak of pro-gun sentiment on the day of the shooting that is sustained at an elevated level for a number of days

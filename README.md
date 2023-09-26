### Determining Whether an Article is "Fake News" or Not

**Nikhil Sakhamuri**

#### Executive summary

#### Rationale
In this era of overwhelming amount of information from innumerable sources, it is very important to know whether or not the news one is reading is real or not. Fake news and headlines are having an increasing amount of influence on people mindsets and behaviors and thereby political outcomes in our country. A machine learning algorithm that is able to quickly and accurately discern between real and fake news articles is imperative to fixing this issue, and thwarting misinformation in our country and all over the world.

#### Research Question
I am trying to answer the question of which machine learning classifier most efficiently and accurately classify between real and fake news articles based on the article title. 

#### Data Sources
I will be using the database of real and fake articles at the below kaggle link-

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?datasetId=572515&sortBy=voteCount

#### Methodology
First, I will preprocess the text using both a stemmer and a lemmatizer. Then I will vectorize the text using both a CountVectorizer and a TfIdfVectorizer. Lastly, for the classification I will use Naive Bayes, Logistic Regression, Decision Tree, SVM, and a Random Forest. 

#### Results
I found that the SVM model fitted using a stemmer and count vectorizer had the highest accuracy score but had the second lowest recall score of all the classifiers. This is a little concerning, as recall in this context stands for how many fake articles were correctly classified as fake. This means that the SVM model incorrectly classified a relatively high number of fake articles as real. Overall, the random forest model performed the best. It had a slightly lower overall accuracy than the SVM but had a higher precision and the highest recall of all models. It also had by far the highest f1-score (weighted average of recall and precision) of all the models. However, the optimal random forest model used 1000 individual decision trees grown to purity and took a ridiculous 4306 sec (71min) to train on my computer. Factoring in training time, the Logistic Regression model seems like the best performer. It had only a slighty lower accuracy than the Random Forest and had the second highest f1-score of all the models, while only taking 25 sec to train. 

#### Next steps
In terms of next steps, it would be very interesting to see a Neural Network trained on this data. I believe a properly build NN could outperform all of these models. Additionally, it seems as though most of the headlines and articles supplied in this dataset are older. It would be useful to retrain the models on newer data in order to be implemented in the real world today. 

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information
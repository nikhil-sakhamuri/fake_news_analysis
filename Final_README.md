### Determining Whether an Article is "Fake News" or Not

**Nikhil Sakhamuri**

### Executive summary

**Project overview and goals:** The goal of this project is to identify more effective ways for detecting fake news online by identifying and evaluating the best machine learning model for detecting fake news based on article title. We will be training five different models to classify whether or not an article is 'fake'. Leveraging the trained data, the model will be able to predict the validity of future artcies. After identifying the best model, we will analyze it's decision-making process and the factors it highlights as most important in predicting whether or not an article is fake. This will give useful insight on the factors that differentiate real and fake articles. Lastly, I will recommend areas to research and possible ways to create a better model in further works.

**Findings:** The best model for detecting fake news articles was the **Random Forest** model, with an accuracy score of 0.9575, a precision score of 0.9726, and a recall score of 0.9449. Among all the models, the **SVM** model had the highest accuracy of .9635, while the **Random Forest** had the highest precision, recall, and f1-score (.9590). One drawback of the **Random Forest** model was the training time, as it took a staggering 4306 sec (71) to train and optimize. If training time was a limiting factor then I would suggest the **Logistic Regression** model as the best one as it had an accuracy score of 0.9526, precision score of 0.9256, and a recall score of 0.9074. The issue with the **SVM** model was even though it had a high accuracy, it had a very low recall score of 0.8580. This meant that **SVM** model incorrectly classified a high number of fake articles as real. To me, this seemed like a costly error and therefore lowered the efficacy of the model. 

Diving deeper into our **Random Forest** model, I found that the token 'video' was by far the most important feature in determining if an article was fake. This is due to the fact that it was highly prevalent in fake news titles and almost non-existant in real titles in our dataset as show by the EDA in the first notebook. This word definitely seems like it would serve as clickbait for fake articles in order to garner user interactions and views. The 2nd most important feature was the token 'say'. This also makes sense as many real articles include quotes or he says/she says titles to grab attention, juxtaposed with fake articles using videos. Unsurprisingly, the tokens 'trump', 'hillari', and 'obama' are also near the top. It makes sense that 'trump' is a valuable token but not the most valuable as I saw in the initial exploratory analysis that the token was extremely prevalant in both real and fake news titles, albeit more popular in fake ones.

**Next steos and recommendations:** Further work can be done to add new features to the dataset rather than just the vectorized tokens. Given the performance of our model, it is clear that simply using the Tfidf Vectorizer (or even a Count Vectorizer) is enough to build an effective predictive model. However, adding other feature before the text is preprocessed such as the proportion of capital letters in the title (I would assume that fake news titles have more caps), the length of the title, and the existence and counts of different types of punctuation could lead to a model with better predictive power. 

In addition to adding more features, it would be extremely interesting the see a Neural Network trained on this data. Obviously, much more work would need to go into designing the layers of such the Neural Network and it would require more computing time than any of the models trained in this analysis, but it would also probably outperform the models tested here. One possible negative of a Neural Network is that it is much more of a black-box model than the models tested here and the prediction process would lose some interpretability.

Lastly, the articles in this dataset on go up until 2018. In order for a fake news prediction model to be used by social media and news companies in real time, more recent date will need to be collected and trained on. However, this project serves to show that such a model is possible and extremely effective in such a classification task. 

#### Rationale
In this era of overwhelming amount of information from innumerable sources, it is very important to know whether or not the news one is reading is real or not. Fake news and headlines are having an increasing amount of influence on people mindsets and behaviors and thereby political outcomes in our country. Back in 2017, BBC News interviewed a panel of 50 experts and many named "the breakdown of trusted information sources" as one of the grand challenges we face in the 21st century. As social media bots become more sophisticated in generating and spreading fake news, algorthims assessing the validity of these news articles must also grow in power and speed. A machine learning algorithm that is able to quickly and accurately discern between real and fake news articles is imperative to fixing this issue, and thwarting misinformation in our country and all over the world.

#### Research Question
I am trying to answer the question of which machine learning classifier most efficiently and accurately classify between real and fake news articles based on the article title. 

#### Data Sources
I will be using the database of real and fake articles at the below kaggle link-

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?datasetId=572515&sortBy=voteCount

Articles are from between March 30th 2015 and February 18th 2018.

#### Methodology
For this task, I will be experimenting with 5 different models: Naive Bayes, Logistic Regression, Decision Tree, SVM, and a Random Forest. For each model, I will fit the model with preprocessed text using a stemmer and a lemmatizer. Then, we will fit a model using both a count vectorizer and a tfidf vectorizer. This will create 4 different feauturized representations of the data that I will fit each model to. Then I will optimize each model over its hyperparameters and record its various metrics on the test data.

#### Results

Below are the best scores of each model recorded:

**SVM:**
	Data preprocessing- Stemmer
	Vectorization- Count Vectorizer
	Training time- 558.2 sec
	Accuracy Score- .9634
	F1 score- 0.9110
	Precision score- 0.9711
	Recall score- 0.8580

**Random Forest:**
	Data preprocessing- Lemmatization
	Vectorization- TfIdf Vectorizer
	Training time- 4306.4 sec
	Accuracy Score- .9580
	F1 score- 0.9590
	Precision score- 0.9720
	Recall score- 0.9463

**Logistic Regression:**
	Data preprocessing- Lemmatization
	Vectorization- TfIdf Vectorizer
	Training time- 24.8 sec
	Accuracy Score- .9526
	F1 score- 0.9164
	Precision score- 0.9256
	Recall score- 0.9074

**Naive Bayes:**
	Data preprocessing- Lemmatization
	Vectorization- Count Vectorizer
	Training time- 19.5 sec
	Accuracy Score- .9400
	F1 score- 0.9126
	Precision score- 0.9000
	Recall score- 0.9257

**Decision Tree:**
	Data preprocessing- Lemmatization
	Vectorization- Count Vectorizer
	Training time- 148.9 sec
	Accuracy Score- .9070
	F1 score- 0.8606
	Precision score- 0.8992
	Recall score- 0.8252

#### Outline of project

- https://github.com/nikhil-sakhamuri/fake_news_analysis/blob/main/Fake_News_Analysis.ipynb
- https://github.com/nikhil-sakhamuri/fake_news_analysis/blob/main/Model_Evaluation.ipynb


##### Contact and Further Information

Nikhil Sakhamuri 
nikhil.sakhamuri3@berkeley.edu
https://www.linkedin.com/in/nikhil-sakhamuri-580255141/

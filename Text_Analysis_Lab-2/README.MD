punit2_lab2.ipynb  is the lab realted to sentiment analysis:

Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis allows businesses to identify customer sentiment toward products, brands or services in online conversations and feedback.

In the current lab, we will work with IMDB movie reviews and develop different machine learning models to predict a given review as positive or negative.

1. We will begin cleaning the reviews(removing stopwords, punctuation and digits).

2. We will then use two different approaches to create feature space.

3. In the 1st one, we will consider the presence/absence of the word(feature) into account. We will send 2 classification models, the 1st one will be support vector machine and the second one would be a random forest classifier. We will calculate the f1 score, test set accuracy and 10 most ifluential words for each of the models.

4. In the 2nd one, we will use TF-IDF vectorization and also include bigrams in the analysis. We will then build naive bayes and random forest classifier. We will again calculate the f1 score, test set accuracy and 10 most ifluential words for each of the models.

5. We also use cross-validation and grid search approach to tune the random forest model in the 2nd appraoch.

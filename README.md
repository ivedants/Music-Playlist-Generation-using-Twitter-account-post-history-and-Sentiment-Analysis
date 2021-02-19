# Music Playlist Generation using Twitter account post history and Sentiment Analysis

In this project, we aimed to develop a playlist generation capability inspired by a user’s or entity’s social media posting history. One way to accomplish this is to use sentiment analysis to compare the distribution of tweet and song lyric sentiment polarities to generate a list of recommended songs. To optimize the efficacy of this prediction, we implemented a variety of both machine and deep learning methods to predict sentiment polarity and evaluated the trained models for both data types. Ultimately, the Bidirectional LSTM (Long-Short Term Memory) for tweets and VADER (Valence Aware Dictionary for Sentiment Reasoning) model for lyrics were implemented when integrating the two polarity scores.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Introduction](#introduction)
- [Datasets](#datasets)
  - [Twitter Data](#twitter-data)
  - [Song Lyric Data](#song-lyric-data)
- [Approach](#approach)
  - [Tweets](#tweets)
  - [Music Lyrics](#music-lyrics)
- [Evaluation](#evaluation)
  - [Evaluation of Twitter Sentiment Analysis](#evaluation-of-twitter-sentiment-analysis)
  - [Evaluation of Song Lyric Sentiment Analysis](#evaluation-of-song-lyric-sentiment-analysis)
- [Implementation: Gluing it all together](#implementation-gluing-it-all-together)
- [Conclusion](#conclusion)
- [References](#references)
  

## Problem Statement

Given tweets from a person’s Twitter account, how can we estimate the general mood of a person to generate a music playlist for them? 
1. Predict tweets’ sentiment as positive, negative, or neutral.
2. Prepare a music lyrics database with sentiment analysis to generate sentiment label for each song.
3. Generate a playlist mapping the overall sentiment of a person’s tweet history with sentiment of different song.

## Introduction

Sentiment analysis serves as a vehicle for delivering a more personalized experience to the users by putting them at the center of the technology and services they use. In our project, we developed a playlist generation capability using Twitter post history. First, we predict tweets’ sentiment as positive, negative, or neutral by evaluating convolutional neural networks (CNN), Long-Short Term Memory (LSTM), and Naive Bayes as baseline. Then, we prepare a music lyrics database with sentiment analysis to generate sentiment labels for each song, using the VADER model, CNNs, and Logistic Regression. Finally, we generate a playlist, mapping the overall sentiment of a person’s tweet history with sentiment of different songs.

## Datasets 

In this project, we used two types of data to accomplish the playlist generation goals: social media (Twitter) data and music lyric data. In the following section, we have outlined the data sources used for acquiring this data.

### Twitter Data

We used SemEval-2017 Twitter dataset (Link: https://alt.qcri.org/semeval2017/task4/index.php?id=results). This dataset consisted of over 53,300 tweets for training and 11,900 tweets for testing. Each tweet has been labeled by a human as either having positive, negative or neutral sentiment. In order to save time and obtain better results, we obtained Stanford’s GloVe pre-trained Twitter embeddings. The model was trained on over 2 billion tweets and 27 billion tokens and the resulting word embedding has a vocabulary size of 1.2 million words. We removed stop words, punctuation, URLs and HTMLs from the tweets. We also converted the tweets to lower case, tokenized, and lematized them. We analyzed the sentiment for the tweet post history of the @therealdonaldtrump Twitter account.

### Song Lyric Data

For our ML and Deep Learning Models for Senti- ment Analysis of song lyrics, we used Billboard Hot 100 dataset from 8/2/1959 - 6/22/2020. For the NLTK VADER Sentiment Analyzer, we collected the data of 100 most popular songs of the British rock band, Coldplay by simply calling it from the Genius Lyrics API.

## Approach

### Tweets

Our initial intent was to follow the framework proposed by Cliche (Cliche, 2017). However, due to lack of computational power and access to large Twitter corpus, we used Stanford’s pre-trained GloVe Twitter embeddings. We also experimented with training our own embeddings. Because of the mentioned limitations, we also had to skip distant supervision (step 3 of Cliche’s framework) which consisted of training our model on a dataset of tweets that are labeled positive if they contain a happy emoticon and are labeled negative if they contain a sad emoticon.

Skipping distant supervision had many disadvantages for the performance of our model. As Tang et al. mention, traditional word embedding algorithms ”only use the contexts of words but ignore the sentiment of texts” (Tang et al., 2016). Therefore, further tuning of embeddings was necessary to make them more sentiment-aware. At the current stage, although our embeddings can capture semantic accurately due to Stanford’s large corpus of Twitter data, they do not capture sentiment polarity.

We trained both CNNs and bi-directional LSTMs on the training portion of the labeled SemEval-2017 Twitter corpus. CNNs are capable of learning features from data invariant of their locations. In our case, they can learn ”the most important n-grams in the embedding space” (Cliche, 2017).

Additionally we implemented a Long-Short Term Memory, a type of Recurrent Neural Network (RNN), which can store sequential information (order of words). LSTMs are special types of RNNs, which learn long-term dependencies(Nowak et al., 2017). For the model architecture, after an initial sequential layer, there is an embedding layer, embedding layer not using pre-trained embeddings and let the network learn the embedding table on it’s own. Then there is an LSTM layer with 16 units and the fully connected layer with softmax activation.

Furthermore, bi-directional LSTMs are capable of not only learning long term dependencies but also remembering post word information for reading a sentence in the two directions of forward and backward. Here is our modified framework which reflects the deviations from the original intended framework: 1) We experiment with both Stanford’s pre-trained GloVe Twitter embeddings; 2) We use the embeddings from the previous step to initial- ize our CNN and bi-directional LSTM models and we will train them on the training data; 3) We will compare the performance of these models to the classic LSTM; 4) We will use the higher performing model to predict the sentiment of tweets from a person’s twitter account which will be later used to generate a playlist for them.

### Music Lyrics

Following the framework of models proposed by researchers (C ̧ ano, 2018), experimentation in song lyric sentiment classification revolved around two simple but efficient models for the task: Optimized Logistic Regression and N-Gram CNN. With these frameworks, we constructed three different Logistic Regression Models, each with N-Gram ranges of unigrams, unigrams and bigrams, and bigrams. We compared them to an N-Gram CNN with multiple convolution and max pooling layers for capturing different N-Gram ranges of lyrics.

The best approach for the purpose of our project was using the NLTK Vader Sentiment Analyzer. As introduced by (Hutto and Gilbert, 2014), VADER stands for “Valence Aware Dictionary and sEntiment Reasoner”. It is a lexicon and rule-based analysis tool consisting of many labeled lexical features (such as words) so it served very efficiently for the song lyrics data. Not only does it tell us about the positively or negativity score but it also tells us about how positive or negative a sentiment is, via a compound score - a normalized, weighted composite score which serves as our most useful metric for a unidimensional measure of sentiment.

<p align="center">
  <img src="https://github.com/ivedants/Music-Playlist-Generation-using-Twitter-account-post-history-and-Sentiment-Analysis/blob/main/Song%20Lyric%20Data%20Sentiment%20Analysis%20Snap.jpg" />
</p>

<p align="center">
  <img src="https://github.com/ivedants/Music-Playlist-Generation-using-Twitter-account-post-history-and-Sentiment-Analysis/blob/main/Lyric%20Sentiment%20Analysis%20Snap%20-%201.jpg" />
</p>

<p align="center">
  <img src="https://github.com/ivedants/Music-Playlist-Generation-using-Twitter-account-post-history-and-Sentiment-Analysis/blob/main/Lyric%20Sentiment%20Analysis%20Snap%202.jpg" />
</p>

## Evaluation

### Evaluation of Twitter Sentiment Analysis

We evaluated the performance of our models by testing them on the test dataset of labeled SemEval-2017 Twitter corpus. As mentioned, we experimented with CNNs, bidirectional LSTMs, regular LSTMs and Naive Bayes models. The figure below summarizes the results of different models that we used. As for the baselines themselves, as anticipated the TF-IDF vectorization technique performed better than bag of words, most likely because it reflects importance of a word in a document, which can preserve the semantics of the statement.

The regular LSTM had a difficult time labelling the sentiments acurately, especially the negative sentiments. This is most likely attributed to the fact that there was no pretrained embeddings as input into the embedding layer forced learn the embedding table on it’s own. As can be observed, bi-directional LSTMs along with pre-trained GloVe embeddings had the best performance in comparison to the other models. Surprisingly, CNNs performed even worse than the baseline Naive Bayes model.

<p align="center">
  <img src="https://github.com/ivedants/Music-Playlist-Generation-using-Twitter-account-post-history-and-Sentiment-Analysis/blob/main/Twitter%20Sentiment%20Analysis%20Evaluation%20Table.jpg" />
</p>

### Evaluation of Song Lyric Sentiment Analysis

Using a random sample of 20000 songs from the Billboard Hot 100 dataset and after proper data processing and transformation, we trained and fitted the Logistic Regression models and the N-Gram CNN. Figure 2 shows the output metrics of the models, with the unigrams and bigrams LR model achieving the best results overall. The CNN’s sub-par performance is most likely due to training our own word embeddings for the model instead of using pretrained embeddings such as Word2Vec and GloVe.

Finally, since all the data for the NLTK Vader method was called from an API and the lyrics were analyzed for sentiment using VADER Sentiment Analyzer, it had no specific test dataset.

<p align="center">
  <img src="https://github.com/ivedants/Music-Playlist-Generation-using-Twitter-account-post-history-and-Sentiment-Analysis/blob/main/Lyric%20Sentiment%20Analysis%20Evaluation%20Table.jpg" />
</p>

## Implementation: Gluing it all together 

For the implementation of the playlist generator, we decided to leverage the trained Bidirectional LSTM model for tweets and VADER model for lyrics due to overall better performance among the techniques. Since the VADER model reported a continuous metric between -1 and 1 and the Bidirectional LSTM produces discrete values of 0,1,2 we need to reconcile them to make an adequate comparison. So we simply averaged the difference in positive and negative polarities over the total weighting the value considering neutrals. This value is compared to each song’s compound polarity in the queried corpus and those with the smallest differences between are recommended. 

As a case study, we collected and ran the end-to-end on Donald Trump’s tweet history on @therealdonaldtrump by first specifying we wanted a playlist of Coldplay songs based on general sentiment associated with account. After the polarity is passed to the song lyric VADER compound polarities, we got a playlist as in the figure below.  

<p align="center">
  <img src="https://github.com/ivedants/Music-Playlist-Generation-using-Twitter-account-post-history-and-Sentiment-Analysis/blob/main/Generated%20Playlist%20Image.jpg" />
</p>

## Conclusion

A takeaway of this work is that traditional methods can sometimes be as effective as deep learning models, as we discovered with the Naive Bayes for tweet sentiment analysis. Also, pre-training decisions, like distant supervision have an impact as well, as we saw better accuracy in the Bidirectional LSTM referenced in Cliche (2017) who included this in their model. Finally, pre-trained embeddings can save a lot of training time and the GloVe embed- dings proved to show better prediction accuracy.

In this case, we conducted sentiment analysis for playlist generation, but sentiment analysis can be effectively used on social media data along with other data such as movies, advertisements, etc. Such connections and mappings can help in making state-of-the-art recommender systems in the future.

## References

1. Mathieu Cliche. 2017. BB_twtr at SemEval-2017 task 4: Twitter sentiment analysis with CNNs and LSTMs. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 573–580, Vancouver, Canada. Association for Computational Linguistics.
2. C.J. Hutto and E.E. Gilbert. 2014. Vader: A parsimo- nious rule-based model for sentiment analysis of social media text. In Eighth International Conference on Weblogs and Social Media (ICWSM-14)., Ann Arbor, MI.
3. Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha, Qatar. Association for Computational Linguistics.
4. Jakub Nowak, Ahmet Taspinar, and Rafal Scherer. 2017. Lstm recurrent neural networks for short text and sentiment classification. pages 553–562.
5. D. Tang, F. Wei, B. Qin, N. Yang, T. Liu, and M. Zhou. 2016. Sentiment embeddings with applications to sentiment analysis. IEEE Transactions on Knowledge and Data Engineering, 28(2):496–509.
6. Erion C ̧ ano. 2018. Text-based sentiment analysis and music emotion recognition.

## Authors 

1. Emily Parrish (GitHub: https://github.com/emparrish417)
2. Shiva Omrani (GitHub: https://github.com/shivaomrani; Email: omrani1997@gmail.com)
3. Vedant Shrivastava (GitHub: https://github.com/ivedants; Email: ivedantshrivastava@gmail.com)
4. Marcelino Velasquez (GitHub: https://github.com/MarcelinoV; Email: marcelinomv98x@gmail.com)

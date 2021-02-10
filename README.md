# Identifying_Bee_Anatomy_Using_Machine_Learning_and_NLP
## Introduction 
1. Snake species identification by using natural language processing:
Used Weka for data extraction.
Data tokenization: using weka, separated words and formed vectors.
Stemming: bringing words to their natural form and removing -ing, -s, -es, etc. 
Removing symbols, articles and punctuations. 
  Feature extraction using TF_IDF:
	𝑁𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑 𝑇𝐹 = 𝑁𝑜.𝑜𝑓 𝑡𝑒𝑟𝑚 𝑡ℎ𝑎𝑡 𝑜𝑐𝑐𝑢𝑟𝑟𝑒𝑑 𝑖𝑛 𝑡ℎ𝑒 𝑡𝑒𝑥𝑡 / 𝑇𝑜𝑡𝑎𝑙 𝑛𝑜 𝑜𝑓 𝑤𝑜𝑟𝑑 𝑖𝑛 𝑡ℎ𝑒 𝑡𝑒xt.

𝐼𝐷𝐹 = log( 𝑇𝑜𝑡𝑎𝑙 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑡𝑒𝑥𝑡𝑠/ 𝑁𝑜.𝑜𝑓 𝑡𝑒𝑥𝑡 𝑖𝑛 𝑤ℎ𝑖𝑐ℎ 𝑠𝑒𝑙𝑒𝑐𝑡𝑒𝑑 𝑡𝑒𝑟𝑚 𝑖𝑠 𝑎𝑝𝑝𝑒𝑎𝑟𝑒𝑑 )
𝑇𝐹_𝐼𝐷𝐹 = 𝑁𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑 𝑇𝐹 ∗ 𝐼𝐷𝐹

## Approaches compared:
Then the K-NN, naive bayes, SVM, and decision tree models were trained. 
Decision tree 71.67% 
SVM 68.33%. 
Naïve Bayes 61.11% 
k-NN by 55.56% 

Possibly random forest could be used to classify the species with even higher accuracy. 

2. Identifying Search Keywords for Finding Relevant Social Media Posts:
The initial ranking, which is less accurate but very efficient, is to identify a large shortlist of likely keywords, or in other words, to remove those unlikely keywords. Re-ranking refines the ranking of the shortlisted words. 
Uses double ranking approach. 
Step 1: Providing keywords to be searched. 
Step 2: Finding all the keywords from the set from the dataset. 
Step 3: Using the initial ranking algo to rank the found keywords. And the unsuitable words should be removed from consideration. 
Step 4: Using the reranking algo, rerank all the keywords in the obtained new set, from the dataset to get the more refined set of found words. 
Step 5: from the new rank list, a subset is selected and checked if the dataset contains the keywords from the new set or not.
Step 6: If the results are satisfactory, end the process, otherwise repeat from step 2. 

3. Zhang H. (et. al), 2018,  MIDAS at SemEval-2019 Task 6: Identifying Offensive Posts and Targeted Offense from Twitter
 
The paper is divided into completing two subtasks which are
Classifying/predicting the tweets if they are Offensive(OFF) or Not offensive(NOT)
Differentiating between targeted Offense(TIN) and untargeted offence. 
They apply the following techniques in this paper:
Convolutional Neural network.
Bidirectional LSTM. 
 Bidirectional LSTM + GRU.
An ensemble of all above 3 architectures. 
They had set up their own dataset by collecting the tweets through python libraries and manually finding the tone of the text body. 
FOr preprocessing they used nltk and Keras. They completed preprocessing in the steps which are:
Tokenization.
Cleaning and normalization. 
Then they applied the above-mentioned techniques to obtain the following results. 


## Literature Review:

I. Armouty, B. and Tedmori, S., 2019, April. Automated Keyword Extraction using Support Vector Machine from Arabic News Documents. In 2019 IEEE Jordan International Joint Conference on Electrical Engineering and Information Technology (JEEIT) (pp. 342-346). IEEE.

In this paper, the authors tried to automate the process of extracting keywords from 844 Arabic News Documents. They used statistical feature extraction methods and a supervised learning model (Support Vector Machine classifier) to do so. To preprocess the data, the text was split into sentences and tokenized. Tokens with the same stem in each document were grouped together and then two statistical methods: Term Frequency - Inverse Document Frequency (TF-IDF) and First Occurrence were used as the prime methods of feature extraction using statistical methods. Using these techniques, the frequency of each word was compared to that in other documents. The dataset used by the authors consisted of imbalanced data points - with a very large number of non-keywords as compared to keywords in the documents. To balance the data, they made the number of ‘non-keywords’ closer to the number of ‘keywords’ using the Downsampling Method. .The data values were also normalized between 0 and 1, to make it comparable. Since the data used was not linearly dependent, Support Vector Machines couldn’t classify the data points without the use of a kernel. To overcome this, the authors used the Radial Basis Function (RBF) and found a better hyperplane for separating the two classes. SVM gave a precision of 0.77, recall of 0.58 and an F1 score of 0.65, which were better than that provided by Naive Bayes and Random Forest, in earlier studies.

II. C. Zhang, H. Wang, Y. Liu, D. Wu, Y. Liao, B. Wang, “Automatic Keyword Extraction from Documents Using Conditional Random Fields” in Journal of CIS 4:3(2008), pp.1169-1180, 2008.

In this paper, the authors proposed a sequence labelling method, called ‘Conditional Random Fields’ (CRF), to effectively extract keywords using most of the features present in 600 Chinese academic documents. These documents were divided into 10 datasets, with each document consisting of title, abstract, full-text, headings, subheading and references. The authors used a tool called SegTag to preprocess and compile the data. They defined CRF as a ‘new probabilistic model for segmenting and labelling sequence data.” It is an undirected graphical model that encodes a conditional probability distribution with a given set of features. Maximum entropy learning method was used to train CRF. For their dataset, the most probable label sequence was determined by: 
Y' =arg max P(X|Y) 
Where Y’ was determined using the Viterbi Algorithm. Because of its ability to relax the assumption of conditional independence of the observed data, CRF was also able to avoid the label-bias problem (the ability of a model to completely ignore the current observation when predicting the next label). Using the 10-fold cross-validation method, the authors were able to prove that the CRF model outperforms other machine learning models. An average of 7.83 annotated keywords were extracted from the documents. This model was then compared with other models and it was concluded that CRF (Precision: 0.66, Recall: 0.41, F1 score: 0.51) and SVM (Precision: 0.80, Recall: 0.33, F1 score: 0.46) significantly outperform the rest.

III. Mouratis, T. and Kotsiantis, S., 2009, November. Increasing the accuracy of discriminative of multinomial bayesian classifier in text classification. In 2009 Fourth International Conference on Computer Sciences and Convergence Information Technology (pp. 1246-1251). IEEE.

In this paper, the authors combined Discriminative Multinomial Bayesian Classifier with a feature selection technique that evaluates the worth of an attribute by computing the value of the chi-squared statistic with respect to the class. For data preprocessing, the texts were tokenized, stemmed and represented in the vector form. Then, the dimensionality of the dataset was reduced by removing irrelevant words (eg: a, an, the, etc.), to enhance the model’s computation power.  Since SVMs produce good precision but poor recall, the authors tried to incorporate a Bayesian network classifier, with Frequency Estimate (FE) as the parameter. Using the Chi-squared method for feature ranking, it was found that an alpha of 0.01 produced the best results. The outcome of the model was measured using 10-cross validation.

IV. Rose, S., Engel, D., Cramer, N. and Cowley, W., 2010. Automatic keyword extraction from individual documents. Text mining: applications and theory, 1, pp.1-20.

In this paper, the authors proposed a method called Rapid Automatic Keyword Extraction (RAKE) that uses a list of stopwords and phrase delimiters to detect the most relevant words or phrases in a text. First, the text is split into a list of words and the stopwords (eg: is, that, then) are removed. This gives a list of ‘content words’. Then, the algorithm creates a matrix of words and stores the number of times they co-occur. After this, the words are given a score, which is the degree of a word in the matrix, the degree of the word divided by its frequency, or simply the word frequency. If two keywords or keyphrases appear together in the same order more than twice, a new keyphrase is created regardless of how many stopwords the keyphrase contains in the original text. A keyword is chosen if its score belongs to the top T scores where by default, T is one-third of the content words in the document.

V. Mahata, D., Shah, R.R., Kuriakose, J., Zimmermann, R. and Talburt, J.R., 2018, April. Theme-weighted ranking of keywords from text documents using phrase embeddings. In 2018 IEEE conference on multimedia information processing and retrieval (MIPR) (pp. 184-189). IEEE.

In this paper, the authors used word-embedding algorithms to extract keywords from the arxiv dataset. To preprocess the data, the authors stemmed and tokenized the words present in the text. Stopwords, such as in, that, those; were also removed. To train this preprocessed data, two toolkits - Word2Vec and Fasttext were used. The Word2Vec algorithms include skip-gram and CBOW (continuous bag of words) models, using either hierarchical softmax or negative sampling. In the CBOW model, the distributed representations of surrounding words are combined to predict the word in the middle. While in the Skip-gram model, the distributed representation of the input word is used to predict the context. It was found that the models trained using Fasttext performs better than the models trained using Word2Vec on all the three tasks





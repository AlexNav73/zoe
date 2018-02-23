# zoe
Chatbot experiments

Here is a [retrieval-based model](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/) of a chatbot. 
The model is based on embedding vectors (word2vec) and some kind of heuristic. 
The model predicts a question from a predefined list by a user query. 
The idea is that this task is similar to search using similarity between a user query and a set of documents.
Main idea: calculate the average vector for all words in every sentence/document and use cosine similarity between vectors 
([via](https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt)).

Assumptions: 
- work with short sentences only
- user query can contain any (invalid) text data

Requirements:
- set of questions to map user queries on
- chat history to train/tune the model
- labeled data (pairs "user query - correct question") to train/tune model

Files:
- `zoe.py`: entry point
- `word2vec_model.py`: model based on gensim vectors
- `nl_processor.py`: input data parser/cleaner

Main steps:
- load cleaned/labeled data
- call `model.fit()` the model will store the list of questions and will find the best similarity threshold to detect a correct question by the given query
- call `model.predict()` to get the predicted question (if any) by the given query

Pros:
- recognises semantically close words/sentences
- fast and simple
- interpretable

Cons:
- works bad with acronyms, domain-specific terms, proper names (persons' names, locations etc.)
  - can be fixed by adding dictionaries and hooks (easy but not the best solution)
  - or by pretraining based on a domain-specific wiki data (requires lots of data and efforts)
- works bad with long sentences
- hard to improve the model results; would be easier to rewrite from scratch

TODOs:
- support other languages except English
- compare by speed/accuracy with other word2vec-based methods:
  - [gensim.models.KeyedVectors.wmdistance()](https://radimrehurek.com/gensim/models/keyedvectors.html)
  - [gensim.models.KeyedVectors.n_similarity()](https://radimrehurek.com/gensim/models/keyedvectors.html)
  - using [smooth inverse frequency](https://github.com/peter3125/sentence2vec)
  - weighted average w2v vectors (e.g. tf-idf) etc

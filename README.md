# zoe
Chatbot experiments

Here is a retrieval-based model of a chatbot. 
The model is based on embedding vectors (word2vec) and some kind of heuristic. 
The model predicts a question from a predefined list by a user query. 
The idea is that this task is similar to search using similarity between a user query and a set of documents.
Main idea: calculate the average vector for all words in every sentence/document and use cosine similarity between vectors 
([via](https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt)).

Assumptions: 
- work with short sentences only
- user query can contain any text data

Requirements:
- set of questions to map user queries on
- chat history to train/tune the model
- labeled data (pairs "user query - correct question") to train/tune model

Files:
- zoe: entry point
- word2vec model: model based on gensim vectors
- nl_processor: input data parser/cleaner

Main steps:
- load cleaned/labeled data.
- Call model.fit(); the model will store the list of questions and will find the best similarity threshold to detect a correct question by the given query.
- Call model.predict() to get the predicted question (if any) by the given query.

Pros:
- fast
- interpretable

Cons:
- works bad with acronyms, domain-specific terms, proper names (persons' names, locations etc.)
- works bad with long sentences

TODOs:
- support other languages except English

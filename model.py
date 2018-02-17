from gensim.models import KeyedVectors
import numpy as np
from scipy import spatial
from gensim.scripts.glove2word2vec import glove2word2vec

class Word2VecModel:
  """Calculates cosine similarity between vectors of sentences
    see
    https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt

    To use the model, download word embedding corpus first (e.g. glove from
    https://developer.syn.co.in/tutorial/bot/oscova/pretrained-vectors.html)
    and put to data folder.

    Usage:
      model = Word2VecModel()

      model.similarity('this is a sentence', 'this is also sentence')
      > 0.915479828613

      s, dist = model.most_similar('how are you', queries)
      > how are you, 1
  """

  num_features = 50
  glove_input_file = 'data/glove/glove.6B.50d.txt'
  word2vec_output_file = 'data/glove/glove.6B.50d.txt.word2vec'

  def __init__(self):
    # See how to convert from glove to word2vec:
    # https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    glove2word2vec(self.glove_input_file, self.word2vec_output_file)

    self.word_vectors = KeyedVectors.load_word2vec_format(self.word2vec_output_file, binary=False)
    #queen = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
    #for word, similarity in queen:
    #  print (word, similarity)

    self.index2word_set = set(self.word_vectors.index2word)
    self.sentence_vectors = {}

  def reset(self):
    self.sentence_vectors = {}

  def most_similar(self, sentence, sentences):
    if sentence not in self.sentence_vectors:
      self.sentence_vectors[sentence] = self.avg_vector(sentence)
    sentence_vector = self.sentence_vectors[sentence]

    max_cosine = 0
    most_similar_sentence = ''

    for s in sentences:
      if s not in self.sentence_vectors:
        self.sentence_vectors[s] = self.avg_vector(s)
      v = self.sentence_vectors[s]

      cosine = 1 - spatial.distance.cosine(sentence_vector, v)

      if max_cosine < cosine:
        max_cosine = cosine
        most_similar_sentence = s

    return most_similar_sentence, max_cosine

  def similarity(self, sentence1, sentence2):
    s1_v = self.avg_vector(sentence1)
    s2_v = self.avg_vector(sentence2)
    return 1 - spatial.distance.cosine(s1_v, s2_v)

  def avg_vector(self, sentence):
    '''Returns average vector for sentence'''
    words = sentence.split()
    feature_vec = np.zeros((self.num_features, ), dtype='float32')
    n_words = 0
    for word in words:
      if word in self.index2word_set:
        n_words += 1
        feature_vec = np.add(feature_vec, self.word_vectors[word])
    if (n_words > 0):
      feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


from gensim.models import KeyedVectors
import numpy as np
from scipy import spatial
from gensim.scripts.glove2word2vec import glove2word2vec

class Word2VecModel:
  """Calculates cosine similarity between vectors of sentences
    See https://goo.gl/2r1mTF
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


  def __init__(self, logging):
    self.logging = logging

    # See how to convert from glove to word2vec:
    # https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    glove2word2vec(self.glove_input_file, self.word2vec_output_file)

    # TODO: too slow; optimize loading e.g. using binary data.
    self.word_vectors = \
      KeyedVectors.load_word2vec_format(self.word2vec_output_file, binary=False)
    self.index2word_set = set(self.word_vectors.index2word)
    # Cashes sentence average vector
    self.sentence_to_vector_map = {}


  def reset(self):
    self.sentence_to_vector_map = {}


  def most_similar(self, sentence, sentences):
    if sentence not in self.sentence_to_vector_map:
      self.sentence_to_vector_map[sentence] = self.avg_vector(sentence)
    sentence_vector = self.sentence_to_vector_map[sentence]

    max_cosine = 0
    most_similar_sentence = ''

    for s in sentences:
      if s not in self.sentence_to_vector_map:
        self.sentence_to_vector_map[s] = self.avg_vector(s)
      v = self.sentence_to_vector_map[s]

      cosine = 1 - spatial.distance.cosine(sentence_vector, v)

      if max_cosine < cosine:
        max_cosine = cosine
        most_similar_sentence = s

    return most_similar_sentence, max_cosine


  def similarity(self, sentence1, sentence2):
    """Bad examples for the word2vec model. Fixed by removing stop words.
       # should be 'hello'
      print(model.similarity('hi there', 'where to so see holidays'))
      # should be 'become mentee'
      print(model.similarity('how to become mentee', 'become mentee'))
    """
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


class ChatModel(Word2VecModel):
  """Predicts user question by user input.
    Stores predefined list of questions during calling fit().
  """

  def __init__(self, logging):
    Word2VecModel.__init__(self, logging)
    # Similarity threshold used to detect if similar sentence found
    self.similarity = 0
    # Set of questions.
    self.questions = {}


  def reset(self):
    Word2VecModel.reset(self)
    self.similarity = 0
    self.questions = {}


  def fit(self, data, similarity_thresholds=[0.9]):
    """Trains using the given data trying to choose best similarity threshold.
    :param data: list of dictionaries in the form:
    {query:<>, parsed_query:<>, question:<>, correct:<1|0>}
    :param similarity_thresholds:
    :return:
    """

    rows_number = len(data)
    self.questions = {v['question'] for v in data}

    # A query can have several correct questions. Cache it in the map.
    query_to_correct_questions_map = {}
    for row in data:
      if not row['correct']:
        continue

      #query = row['query']
      query = row['parsed_query']
      question = row['question']
      if query not in query_to_correct_questions_map:
        query_to_correct_questions_map[query] = [question]
      else:
        query_to_correct_questions_map[query].append(question)


    # Cache query similarities in map.
    self.logging.debug("Calculate similarities")
    parsed_query_to_similarity_map = {}
    processed_rows_number = 0
    # Be careful: this metric shows better results without removing stop words
    # from user history.
    avg_similarity = 0
    for row in data:
      processed_rows_number += 1

      parsed_query = row['parsed_query']
      if parsed_query not in parsed_query_to_similarity_map:
        similar_question, similarity = \
          self.most_similar(parsed_query, self.questions)
        parsed_query_to_similarity_map[parsed_query] = \
          (similar_question, similarity)
      else:
        similarity = parsed_query_to_similarity_map[parsed_query][1]

      avg_similarity += similarity

      if processed_rows_number % 100 == 0:
        self.logging.debug(
            'processed: %5d of %d, avg similarity: %0.8f',
            processed_rows_number,
            rows_number,
            avg_similarity / processed_rows_number)

    self.logging.debug(
        'processed: %d of %d, avg similarity: %0.8f',
        processed_rows_number,
        rows_number,
        avg_similarity / processed_rows_number)

    # Find similarity level which gives best results
    self.logging.debug("Calculate accuracy")
    max_accuracy = 0
    max_precision = 0
    max_recall = 0
    best_similarity = 0
    for similarity_threshold in similarity_thresholds:
      tp = 0
      tn = 0
      fp = 0
      fn = 0
      for row in data:
        parsed_query = row['parsed_query']
        #query = row['query']
        query = parsed_query
        #question = v['question']
        #is_correct = v['correct']

        similar_question, similarity = \
          parsed_query_to_similarity_map[parsed_query]

        correct_question_found = False
        if query in query_to_correct_questions_map:
          if similar_question in query_to_correct_questions_map[query]:
            correct_question_found = True

        if similarity < similarity_threshold and not correct_question_found:
          tn += 1
        elif similarity >= similarity_threshold and correct_question_found:
          tp += 1
        elif similarity < similarity_threshold and correct_question_found:
          fn += 1
        else:
          #if similarity > similarity_threshold and not correct_question_found:
          fp += 1

      accuracy = (tp + tn) / rows_number
      precision = tp / (tp + fp) if (tp + fp) != 0 else 0
      recall = tp / (tp + fn) if (tp + fn) != 0 else 0
      self.logging.info(
          'precision: %0.4f, recall: %0.4f, accuracy: %0.4f, similarity: %0.4f',
          precision, recall, accuracy, similarity_threshold)
      if accuracy > max_accuracy:
        best_similarity = similarity_threshold
        max_accuracy = accuracy
        max_precision = precision
        max_recall = recall

    self.logging.info(
        'Best precision: %0.4f, '
        + 'recall: %0.4f, '
        + 'accuracy: %0.4f, '
        + 'similarity: %0.4f',
        max_precision, max_recall, max_accuracy, best_similarity)

    self.similarity = best_similarity
    return accuracy, best_similarity


  def predict_questions(self, questions_data):
    predicted_questions = []
    #rows_to_process = math.inf
    current_row = 0
    for k, v in questions_data.items():
      current_row += 1
      #if current_row > rows_to_process:
      #  break

      parsed_query = v['parsed_query']
      predicted_question, similarity = self.predict(parsed_query)
      predicted_questions.append({
        'similarity': similarity,
        'query': v['query'],
        'parsed_query': parsed_query,
        'predicted_question': predicted_question,
        'question': v['question'],
        #'answer': v['answer']
      })

      if current_row % 100 == 0:
        self.logging.debug(
            'Predicted: %5d of %d',
            current_row,
            len(questions_data))

    return predicted_questions

  def predict(self, sentence):
    similar_question, similarity = self.most_similar(sentence, self.questions)
    if similarity < self.similarity:
      return '', similarity # Not found.
    return similar_question, similarity
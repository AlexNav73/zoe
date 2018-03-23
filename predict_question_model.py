
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy import spatial

class AbstractModel:

  def reset(self):
    raise NotImplementedError('subclasses must override it')


  def fit(self, data, similarity_thresholds):
    raise NotImplementedError('subclasses must override it')


  def predict_questions(self, questions_data):
    raise NotImplementedError('subclasses must override it')


class PredictQuestionModel(AbstractModel):
  """Predicts user question by user input.
    Stores predefined list of questions during calling fit().
  """

  def __init__(self, metric, logging):
    self.logging = logging
    self.metric = metric
    # Similarity threshold used to detect if similar sentence found
    self.similarity = 0
    # Set of questions.
    self.questions = {}


  def reset(self):
    self.metric.reset(self)
    self.similarity = 0
    self.questions = {}


  def load_questions(self, questions):
    self.questions = set(questions)


  def fit(self, data, similarity_thresholds=[0.9]):
    """Trains using the given data trying to choose best similarity threshold.
    :param data: list of dictionaries in the form:
    {query:<>, parsed_query:<>, question:<>, correct:<1|0>}
    :param similarity_thresholds:
    :return:
    """

    rows_number = len(data)
    # A query can have several correct questions. Cache it in the map.
    query_to_correct_questions_map = {}
    for row in data:
      if not row['correct']:
        continue

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
          self.metric.most_similar(parsed_query, self.questions)
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
        query = row['parsed_query']

        similar_question, similarity = \
          parsed_query_to_similarity_map[query]

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
    current_row = 0

    for _, v in questions_data.items():
      current_row += 1

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
        self.logging.debug(f'Predicted: {current_row} of {len(questions_data)}')

    return predicted_questions

  def predict(self, sentence):
    similar_question, similarity = self.metric.most_similar(sentence, self.questions)
    if similarity < self.similarity:
      return '', similarity # Not found.
    return similar_question, similarity
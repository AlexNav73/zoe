import csv
import logging
import time

from word2vec_model import ChatModel
from nl_processor import NLProcessor


def cleaned_data_to_file(rows, filename, delimiter='\t'):
  logging.info("storing user history to %s", filename)

  with open(filename, "w") as history_output:
    #clean_history_output.write('\n'.join(history_rows))
    fieldnames = ['query', 'parsed_query', 'question', 'accuracy', 'date', 'answer']
    writer = csv.DictWriter(history_output, delimiter=delimiter, fieldnames=fieldnames)
    writer.writeheader()
    for key, value in rows.items():
      query = value['query']
      parsed_query = value['parsed_query']
      question = value['question']
      accuracy = value['accuracy']
      date = value['date']
      answer = value['answer']
      writer.writerow({
        'query': query,
        'parsed_query': parsed_query,
        'question': question,
        'accuracy': accuracy,
        'date': date,
        'answer': answer})


def load_cleaned_data(path, delimiter='\t'):
  """Expected csv format:
    query	parsed_query	question	accuracy	date	answer	correct
    where correct=1|0 - label that means that parsed_query is a correct question
    accuracy - oscova chatbot accuracy (from another model)
  :return: list of dictionaries
  """
  with open(path, 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=delimiter)
    cleaned_data = []
    for row in reader:
      cleaned_data.append(row)
  return cleaned_data


class ModelStub:
  """Used for testing only"""

  def __init__(self, logging):
    self.logging = logging


  def most_similar(self, sentence, sentences):
    return next(iter(sentences)), 1


  def fit(self, data, similarity_thresholds=[0.9]):
    logging.info(
        'accuracy: %0.4f, similarity: %0.4f', 1, similarity_thresholds[0])
    return 1, similarity_thresholds[0]


  def predict_questions(self, questions_data):
    predicted_questions = []
    predicted_questions.append({
      'similarity': 1,
      'query': 'query',
      'parsed_query': 'parsed_query',
      'predicted_question': 'predicted_question',
      'question': 'question',
      #'answer': v['answer']
    })
    return predicted_questions


  def predict(self, sentence):
    return sentence, 1


def main():
  start = time.clock()

  # Load data to train model
  cleaned_data_file = 'data/cleaned_history.csv'
  logging.debug("Loading cleaned data from %s", cleaned_data_file)
  #history_to_file(history_rows, output_file)
  cleaned_data = load_cleaned_data(cleaned_data_file)

  nl_processor = NLProcessor(logger)
  # nl_processor.corrected_spelling = 0
  # for row in cleaned_data:
  #   parsed_query = row['parsed_query']
  #   corrected_spelling = \
  #     nl_processor.correct_spelling(parsed_query, nl_processor.acronyms)
  #   if parsed_query != corrected_spelling:
  #     row['parsed_query'] = corrected_spelling

  logging.debug(">> time spent: %ds\n", time.clock() - start)

  # Build model
  logging.info("Creating model")
  model = ChatModel(logger)
  #model = ModelStub(logger)
  logging.debug("Training model")
  accuracy, similarity = model.fit(cleaned_data, [.96, .97, .98])

  logging.debug(">> time spent: %ds\n", time.clock() - start)

  # Load raw data
  history_file = 'data/history.csv'
  logging.debug("Cleaning data from %s", history_file)
  questions_data = nl_processor.parse(history_file)

  logging.debug(">> time spent: %ds\n", time.clock() - start)

  # Predict
  logging.info("Predicting questions")
  predicted_questions = model.predict_questions(questions_data)
  print("{:<8}; {:<40}; {:<40}; {:<40}; {:<40}".format(
      'SIMILARITY',
      'QUERY',
      'PARSED QUERY',
      'PREDICTED QUESTION',
      'QUESTION'
  ))
  for row in predicted_questions:
    #logging.debug(
    #"%d;\t%s;\t\t\t%s", row['accuracy'], row['query'], row['similar_question'])
    print("{:.8f}; {:<40}; {:<40}; {:<40}; {:<40}".format(
        float(row['similarity']),
        row['query'],
        row['parsed_query'],
        row['predicted_question'],
        row['question']))

  logging.debug(">> time spent: %ds\n", time.clock() - start)
  logging.info('done')


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  logger = logging.getLogger(__name__)
  main()

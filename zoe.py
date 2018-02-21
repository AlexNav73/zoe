import csv
import logging

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


  def fit(self, data, similarity_thresholds=[0.5]):
    logging.info(
        'accuracy: %0.4f, similarity: %0.4f', 1, similarity_thresholds[0])
    return 1, similarity_thresholds[0]


  def predict(self, sentence):
    return sentence, 1


def main():
  # Load data to train model
  cleaned_data_file = 'data/cleaned_history.csv'
  #history_to_file(history_rows, output_file)
  cleaned_data = load_cleaned_data(cleaned_data_file)

  # Build model
  logging.info("Creating model")
  model = ChatModel(logger)
  #model = ModelStub(logger)
  logging.debug("Training model")
  accuracy, similarity = model.fit(cleaned_data, [.96, .97, .98])

  # Load raw data
  nl_processor = NLProcessor(logger)
  history_file = 'data/history.csv'
  raw_data = nl_processor.parse(history_file)

  # Predict
  logging.info("Predicting questions")
  predicted_questions = []
  #rows_to_process = math.inf
  current_row = 0
  for k, v in raw_data.items():
    current_row += 1
    #if current_row > rows_to_process:
    #  break

    parsed_query = v['parsed_query']
    predicted_question, similarity = model.predict(parsed_query)
    predicted_questions.append({
      'similarity': similarity,
      'query': v['query'],
      'parsed_query': parsed_query,
      'predicted_question': predicted_question,
      'question': v['question'],
      #'answer': v['answer']
    })

    if current_row % 50 == 0:
      logging.debug(
        'predicted: %5d of %d',
        current_row,
        len(raw_data))


  print("{:<8}; {:<40}; {:<40}; {:<40}; {:<40}".format(
      'SIMILARITY',
      'QUERY',
      'PARSED QUERY',
      'PREDICTED QUESTION',
      'QUESTION'
  ))
  for row in predicted_questions:
    #self.logging.debug(
    #"%d;\t%s;\t\t\t%s", row['accuracy'], row['query'], row['similar_question'])
    print("{:.8f}; {:<40}; {:<40}; {:<40}; {:<40}".format(
      float(row['similarity']),
      row['query'],
      row['parsed_query'],
      row['predicted_question'],
      row['question']))

  logging.info('done')


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  logger = logging.getLogger(__name__)
  main()
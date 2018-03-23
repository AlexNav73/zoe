import sys
import csv
import logging
import time
import argparse

from word2vec_model import ChatModel
from nl_processor import NLProcessor

OUTPUT_FILE = "data/output.txt"
# HISTORY_FILE = 'data/history.csv'
HISTORY_FILE = 'data/321.csv'
QUESTIONS_KB = 'data/questions.txt'
CLEANED_DATA_FILE = 'data/cleaned_history.csv'
GLOVE_INPUT_FILE = 'data/glove/glove.6B.50d.txt'

def cleaned_data_to_file(rows, filename, delimiter='\t'):
  logging.info("storing user history to %s", filename)

  with open(filename, "w") as history_output:
    #clean_history_output.write('\n'.join(history_rows))
    fieldnames = ['query', 'parsed_query', 'question', 'accuracy', 'date', 'answer']
    writer = csv.DictWriter(history_output, delimiter=delimiter, fieldnames=fieldnames)
    writer.writeheader()
    for _, value in rows.items():
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
    query,parsed_query,question,accuracy,date,answer,correct
    where `correct`=1|0 - label that means that `parsed_query` is a correct question
    `accuracy` - Oscova chatbot accuracy (from another model)
  :return: list of dictionaries
  """
  with open(path, 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=delimiter)
    cleaned_data = []
    for row in reader:
      cleaned_data.append(row)
  return cleaned_data


def load_questions_from_file(path):
  return { q.strip() for q in open(path, 'r') }


def write_results_to_file(results, path):
  with open(path, "w", encoding="utf8") as output:
    output.write("{:<8}; {:<40}; {:<40}; {:<40}; {:<40}\n".format(
        'SIMILARITY',
        'QUERY',
        'PARSED QUERY',
        'PREDICTED QUESTION',
        'QUESTION'
    ))
    for row in results:
      output.write("{:.8f}; {:<40}; {:<40}; {:<40}; {:<40}\n".format(
          float(row['similarity']),
          row['query'],
          row['parsed_query'],
          row['predicted_question'],
          row['question']))

def main():
  start = time.clock()

  # Load data to train model
  logging.debug("Loading cleaned data from %s", CLEANED_DATA_FILE)
  cleaned_data = load_cleaned_data(CLEANED_DATA_FILE)

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
  model = ChatModel(logger, GLOVE_INPUT_FILE)
  logging.debug("Loading KB questions")
  questions = load_questions_from_file(QUESTIONS_KB)
  model.load_questions(questions)
  logging.debug("Training model")
  accuracy, similarity = model.fit(cleaned_data, [.95, .96, .97, .98])

  logging.info("Model has been trained successfully. accuracy: %d similarity: %d",
      accuracy,
      similarity)
  logging.debug(">> time spent: %ds\n", time.clock() - start)

  # Load raw data
  logging.debug("Cleaning data from %s", HISTORY_FILE)
  questions_data = nl_processor.parse(HISTORY_FILE)

  logging.debug(">> time spent: %ds\n", time.clock() - start)

  # Predict
  logging.info("Predicting questions")
  predicted_questions = model.predict_questions(questions_data)

  logging.info("Dump output to %s file", OUTPUT_FILE)
  write_results_to_file(predicted_questions, OUTPUT_FILE)

  logging.debug(">> time spent: %ds\n", time.clock() - start)
  logging.info('done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--embed', action="store_true", help="Use script as embedded process. All logging will be skipped")

  args = parser.parse_args()

  level = logging.DEBUG
  if args.embed:
    level = logging.CRITICAL

  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=level)
  logger = logging.getLogger(__name__)
  main()

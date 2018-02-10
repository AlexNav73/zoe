import nltk
import numpy
import csv
import string
import re
from datetime import datetime
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import logging

from model import Word2VecModel

# nltk.download('punkt')
# nltk.download('stopwords')

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def clean_history(path, delimiter=','):

  """Cleans up text corpus from csv file with given delimiter:
    - filters non-ascii symbols
    - strips and lowers every row
    - removes not word characters or spaces
    - removes rows containing less than 2 symbols
    # - removes rows containing only one word
    - removes rows containing only digits and whitespaces
    - removes rows containing only stop words
    - leaves only unique rows by user message and question
    Expected csv format:
    Id	UserMessage	EtalonQuestion	Accuracy	Etalon	Created	Answer
  :param path:      path to file
  :param delimiter  delimiter of csv file
  :return: cleaned rows
  """

  def parse_user_message(text):
    nonlocal ignored_stopwords
    nonlocal ignored_len
    nonlocal ignored_digits

    # strips and lowers every row
    parsed = text.strip().lower() if text is not None else ''

    # removes not word characters or spaces
    parsed = re.sub(r'[^\w\s]', '', parsed).strip()

    # removes rows containing less than 2 symbols
    if (len(parsed)) < 2:
      ignored_len += 1
      return ''

    # if token.isdigit():
    #  continue

    # token_words = word_tokenize(token)
    token_words = parsed.split()

    # removes rows containing only one word
    # if len(token_words) <= 1:
    #  ignored_words_number += 1
    #  continue

    # removes rows containing only digits and whitespaces
    if all(word.isdigit() for word in token_words):
      ignored_digits += 1
      return ''

    # removes rows containing only stop words
    if all(word in stop_words for word in token_words):
      ignored_stopwords += 1
      return ''

    if all(word in stop_words_ru for word in token_words):
      ignored_stopwords += 1
      return ''

    return parsed


  with open(path, 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=delimiter)

    parsed_rows = {}
    stop_words = stopwords.words('english')
    # print(*sorted(stop_words), sep='\n')
    stop_words_ru = stopwords.words('russian')
    rows_number = 0
    ignored_accuracy = 0
    ignored_len = 0
    # ignored_words_number = 0
    ignored_digits = 0
    ignored_stopwords = 0
    ignored_nonunique = 0
    ignored_non_ascii = 0

    for row in reader:
      rows_number += 1

      user_message = row['UserMessage']
      etalon_question = row['EtalonQuestion']
      answer = row['Answer']
      is_etalon = row['Etalon']
      accuracy = row['Accuracy']
      created = row['Created']
      created_date = datetime.strptime(created, '%Y-%m-%d %H:%M:%S.%f')

      # filters non-ascii symbols
      try:
        user_message.encode('ascii')
        etalon_question.encode('ascii')
        answer.encode('ascii')
      except UnicodeEncodeError:
        ignored_non_ascii += 1
        continue

      if accuracy is None or not is_number(accuracy):
        ignored_accuracy += 1
        continue

      # if float(accuracy) == 1:
      #  continue

      parsed_user_message = parse_user_message(user_message)
      if parsed_user_message == '':
        continue

      # leaves only unique rows by user message and question
      hash = '{}_{}'.format(parsed_user_message, etalon_question)
      if hash not in parsed_rows:
        parsed_rows[hash] = \
          {
            'query': parsed_user_message,
            'question': etalon_question,
            'accuracy': accuracy,
            'date': created_date,
            'answer': answer
           }
      elif parsed_rows[hash]['accuracy'] == accuracy:
        # if there is row with the same hash already, leave the variant with the latest date
        updated_created_date = parsed_rows[hash]['date']
        if updated_created_date < created_date:
          updated_created_date = created_date
        # print('{} : {}\t{}, time: {}, {}'.format(hash, parsed_rows[hash][2], accuracy, parsed_rows[hash][3], created))
        parsed_rows[hash] = \
          {
            'query': parsed_user_message,
            'question': etalon_question,
            'accuracy': accuracy,
            'date': updated_created_date,
            'answer': answer
          }
      else:
        ignored_nonunique += 1

        # TODO: Handling of domain specific words, phrases, and acronyms
        # TODO: Locating and correcting common typos and misspellings

    logging.debug('{} of {} lines parsed\n'.format(len(parsed_rows), rows_number))

    return parsed_rows


def main():
  chatHistoryFile = 'data/history.csv'

  logging.info("cleaning chat history")
  history_rows = clean_history(chatHistoryFile, delimiter='\t')
  rows_number = len(history_rows)
  questions = [v['question'] for k, v in history_rows.items()]

  #with open("data/cleaned_history.txt", "w") as clean_history_output:
  #  history_rows.write('\n'.join(uniqueRows))

  # build model
  logging.info("creating word2vec model")
  model = Word2VecModel()

  # TODO: train model

  # validate model

  # hi_s, hi_dist = model.most_similar('hi', queries)
  # hello_s, hello_dist = model.most_similar('hello', queries)
  # how_are_you_s, how_are_you_dist = model.most_similar('how are you', queries)

  # Calculate % of questions that model finds
  found_questions_number = 0
  processed_rows_number = 0
  total_accuracy = 0
  for k, v in history_rows.items():
    query = v['query']
    question = v['question']
    similar_question, distance = model.most_similar(query, questions)
    total_accuracy += distance
    if similar_question == question:
      found_questions_number += 1
    processed_rows_number += 1
    if processed_rows_number % 10 == 0:
      logging.debug(
          'processed: %5d of %i, found questions: %i, accuracy: %0.8f',
          processed_rows_number,
          rows_number,
          found_questions_number,
          total_accuracy / processed_rows_number)
  logging.info(
      'processed: %i, found questions: %i, accuracy: %0.8f',
      rows_number,
      found_questions_number,
      total_accuracy / rows_number)
  logging.info('done')


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  main()
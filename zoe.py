import csv
import re
from datetime import datetime
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

def filter_stop_words(sentence, stop_words):
  words = sentence.split()
  filtered_words = [word for word in words if word not in stop_words]
  return ' '.join(filtered_words)

def clean_user_history(path, delimiter=','):

  """Cleans up text corpus from csv file with given delimiter:
    - filters non-ascii symbols
    - strips and lowers every row
    - removes not word characters or spaces
    - removes rows containing less than 2 symbols
    #- removes rows containing only one word
    - removes rows containing only digits and whitespaces
    - removes rows containing only stop words
    - removes stop words
    - leaves only unique rows by user message and question
    Expected csv format:
    Id	UserMessage	EtalonQuestion	Accuracy	Etalon	Created	Answer
  :param path:      path to file
  :param delimiter  delimiter of csv file
  :return: cleaned rows
  """

  def parse_query(text):
    nonlocal ignored_stopwords
    nonlocal ignored_len
    nonlocal ignored_words_number
    nonlocal ignored_digits

    # strips and lowers every row
    parsed = text.strip().lower() if text is not None else ''

    # removes not word characters or spaces
    parsed = re.sub(r'[^\w\s]', '', parsed).strip()

    # removes rows containing less than 2 symbols
    if (len(parsed)) < 2:
      ignored_len += 1
      return ''

    # token_words = word_tokenize(token)
    token_words = parsed.split()

    # removes rows containing only one word
    # if len(token_words) <= 1:
    #  ignored_words_number += 1
    #  return ''

    # removes rows containing only digits and whitespaces
    if all(word.isdigit() for word in token_words):
      ignored_digits += 1
      return ''

    # removes rows containing only stop words
    if all(word in stop_words for word in token_words):
      ignored_stopwords += 1
      return ''

    parsed = filter_stop_words(parsed, stop_words).strip()
    if len(parsed) == 0:
      ignored_stopwords += 1
      return

    if all(word in stop_words_ru for word in token_words):
      ignored_stopwords += 1
      return ''

    parsed = filter_stop_words(parsed, stop_words_ru).strip()
    if len(parsed) == 0:
      ignored_stopwords += 1
      return

    return parsed


  with open(path, 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=delimiter)

    parsed_rows = {}
    stop_words = stopwords.words('english')
    #print(*sorted(stop_words), sep='\n')
    stop_words_ru = stopwords.words('russian')
    #print(*sorted(stop_words_ru), sep='\n')
    rows_number = 0
    ignored_accuracy = 0
    ignored_len = 0
    ignored_words_number = 0
    ignored_digits = 0
    ignored_stopwords = 0
    ignored_nonunique = 0
    ignored_non_ascii = 0

    for row in reader:
      rows_number += 1

      query = row['UserMessage']
      etalon_question = row['EtalonQuestion']
      answer = row['Answer']
      is_etalon = row['Etalon']
      accuracy = row['Accuracy']
      created = row['Created']
      created_date = datetime.strptime(created, '%Y-%m-%d %H:%M:%S.%f')

      # filters non-ascii symbols
      try:
        query.encode('ascii')
        etalon_question.encode('ascii')
        answer.encode('ascii')
      except UnicodeEncodeError:
        ignored_non_ascii += 1
        continue

      if accuracy is None or not is_number(accuracy):
        ignored_accuracy += 1
        continue

      if float(accuracy) < .8:
        ignored_accuracy += 1
        continue

      parsed_query = parse_query(query)
      if parsed_query == '':
        continue

      # leaves only unique rows by user message and question
      hash = '{}_{}'.format(query, etalon_question)
      if hash not in parsed_rows:
        parsed_rows[hash] = \
          {
            'query': query,
            'parsed_query': parsed_query,
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
            'query': query,
            'parsed_query': parsed_query,
            'question': etalon_question,
            'accuracy': accuracy,
            'date': updated_created_date,
            'answer': answer
          }
        ignored_nonunique += 1
      else:
        ignored_nonunique += 1

        # TODO: Handling of domain specific words, phrases, and acronyms
        # TODO: Locating and correcting common typos and misspellings

    logging.debug('%d of %d lines parsed', len(parsed_rows), rows_number)
    logging.debug('\n\
        ignored_stopwords=%d\n\
        ignored_len=%d\n\
        ignored_words_number=%d\n\
        ignored_digits=%d\n\
        ignored_non_ascii=%d\n\
        ignored_accuracy=%d\n\
        ignored_nonunique=%d',
        ignored_stopwords,
        ignored_len,
        ignored_words_number,
        ignored_digits,
        ignored_non_ascii,
        ignored_accuracy,
        ignored_nonunique
    )

    return parsed_rows

def filterRowsByAccuracy(rows, model, accuracy=0.5):
  # Calculate % of questions that model finds
  found_questions_number = 0
  processed_rows_number = 0

  # Be careful: this metric shows better results without removing stop words from user history.
  avg_accuracy = 0
  rows_number = len(rows)
  filtered_rows = []
  questions = [v['question'] for k, v in rows.items()]
  logging.info('Filtering by accuracy %0.8f', accuracy)

  for k, v in rows.items():
    parsed_query = v['parsed_query']
    query = v['query']
    question = v['question']

    similar_question, distance = model.most_similar(parsed_query, questions)

    processed_rows_number += 1

    if distance < accuracy:
      continue

    filtered_rows.append({
      'parsed_query': parsed_query,
      'query': query,
      'similar_question': similar_question,
      'accuracy': distance,
      'question': question,
      #'answer': v['answer']
    })

    avg_accuracy += distance
    if similar_question == question:
      found_questions_number += 1

    if processed_rows_number % 20 == 0:
      logging.debug(
          'processed: %5d of %d, found questions: %d, avg accuracy: %0.8f',
          processed_rows_number,
          rows_number,
          found_questions_number,
          avg_accuracy / processed_rows_number)

  logging.debug(
      'processed: %d of %d, found questions: %d, avg accuracy: %0.8f',
      processed_rows_number,
      rows_number,
      found_questions_number,
      avg_accuracy / processed_rows_number)

  return filtered_rows


def history_to_file(rows, filename, delimiter='\t'):
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

class ModelStub:
  """Used for testing only"""

  def most_similar(self, sentence, sentences):
    return sentences[0], 1


def main():
  chatHistoryFile = 'data/history.csv'

  logging.info("cleaning chat history")
  history_rows = clean_user_history(chatHistoryFile, delimiter='\t')
  history_to_file(history_rows, 'data/cleaned_history.csv')

  # build model
  logging.info("creating word2vec model")

  # TODO: train model

  # validate model

  # Bad examples for the word2vec model. Fixed by removing stop words.
  #print(model.similarity('hi there', 'where to so see holidays')) # should be 'hello'
  #print(model.similarity('how to become mentee', 'become mentee')) # should be 'become mentee'

  model = Word2VecModel()
  #model = ModelStub()
  filtered_rows = filterRowsByAccuracy(history_rows, model, 0.8)

  for row in filtered_rows:
    #logging.debug("%d;\t%s;\t\t\t%s", row['accuracy'], row['query'], row['similar_question'])
    print("{:.8f}; {:<40}; {:<40}; {:<40}; {:<40}".format(
        float(row['accuracy']),
        row['query'],
        row['parsed_query'],
        row['similar_question'],
        row['question']))

  logging.info('done')

  # TODO: find accuracy level which gives better results


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  main()
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tree import Tree
import re
from datetime import datetime

# nltk.download('punkt')
# nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def filter_stop_words(phrase, stop_words):
  words = phrase.split()
  filtered_words = [word for word in words if word not in stop_words]
  return ' '.join(filtered_words)


# TODO: too slow; use detect_persons_names() instead
def detect_names_using_chunking(phrase):
  """See http://www.nltk.org/book/ch07.html
    Does not recognise lowercase.
    Usage:
    remove_names('Summer School Coordinator Katrina Langdon')
  """
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(phrase))):
    if type(chunk) == Tree:
      if (chunk.label() == 'GPE' or chunk.label() == 'PERSON') \
          and chunk[0][1] == 'NNP':
        #if chunk.label() == 'PERSON' and chunk[0][1] == 'NNP':
        #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
        return True

  return False


def detect_persons_names(phrase, names):
  """Doesn't understand some non-english names like Sergiy"""
  words = phrase.split()
  for word in words:
    if word in names:
      #self.logging.debug('%s -- %s', word, phrase)
      return True
  return False


class NLProcessor:

  def __init__(self, logging):
    self.logging = logging

  # TODO: parse non-english data (now non-ascii is just ignored).
  def parse(self, path, delimiter='\t'):
    """Cleans up text corpus from csv file with given delimiter:
      - filters non-ascii symbols
      - strips and lowers every row
      - removes not word characters or spaces
      - removes rows containing less than 2 symbols
      - removes rows containing only one word
      - removes rows containing only digits and whitespaces
      - removes rows containing only stop words
      - removes stop words
      - removes persons' names
      - leaves only unique rows by user message and question
      Expected csv format:
      Id	UserMessage	EtalonQuestion	Accuracy	Etalon	Created	Answer
    :param path:      path to file
    :param delimiter  delimiter of csv file
    :return: dictionary of dictionaries in the form:
    {query:<>, parsed_query:<>, question:<>, accuracy:<>, date:<>, answer:<>}
    where dictionary keys are combinations of parsed_query_question
    """

    def remove_stop_words(phrase):
      nonlocal ignored_stopwords

      # token_words = phrase.split()

      # removes rows containing only stop words
      # if all(word in stop_words for word in token_words):
      #   ignored_stopwords += 1
      #   return ''

      parsed_phrase = filter_stop_words(phrase, stop_words).strip()
      if len(parsed_phrase) == 0:
        ignored_stopwords += 1
        return ''

      # if all(word in stop_words_ru for word in token_words):
      #   ignored_stopwords += 1
      #   return ''

      parsed_phrase = filter_stop_words(parsed_phrase, stop_words_ru).strip()
      if len(parsed_phrase) == 0:
        ignored_stopwords += 1
        return ''

      return parsed_phrase


    def parse_query(phrase):
      nonlocal ignored_stopwords
      nonlocal ignored_len
      nonlocal ignored_words_number
      nonlocal ignored_digits
      nonlocal ignored_names

      # strips and lowers every row
      parsed_phrase = phrase.strip().lower() if phrase is not None else ''

      # removes not word characters or spaces
      parsed_phrase = re.sub(r'[^\w\s]', '', parsed_phrase).strip()

      # removes rows containing less than 2 symbols
      if (len(parsed_phrase)) < 2:
        ignored_len += 1
        return ''

      # token_words = word_tokenize(token)
      token_words = parsed_phrase.split()

      # removes rows containing only digits and whitespaces
      if all(word.isdigit() for word in token_words):
        ignored_digits += 1
        return ''

      parsed_phrase = remove_stop_words(parsed_phrase)

      # removes rows containing only one word
      token_words = parsed_phrase.split()
      if len(token_words) <= 1:
        ignored_words_number += 1
        return ''

      if detect_names_using_chunking(phrase):
        ignored_names += 1
        return ''

      if detect_persons_names(parsed_phrase, persons_names):
        ignored_names += 1
        return ''

      return parsed_phrase


    self.logging.info("Cleaning user history from %s", path)

    with open(path, 'r') as csvfile:
      reader = csv.DictReader(csvfile, delimiter=delimiter)

      query_question_to_row_map = {}
      stop_words = stopwords.words('english')
      #print(*sorted(stop_words), sep='\n')
      stop_words_ru = stopwords.words('russian')
      #print(*sorted(stop_words_ru), sep='\n')
      # downloaded from http://www.outpost9.com/files/WordLists.html
      # crl-names - filters names like How, Holiday:(
      # Given-Names - filters names like Meeting, Car:(

      # Downloaded form https://goo.gl/EwBq8s
      names_corpus = 'data/census.gov.names.txt'
      persons_names = \
        [ # TODO: find better dataset.
          line.rstrip('\n').lower() for line in open(names_corpus)
          if not line.startswith('#')
        ]
      # persons_names2 = []
      # for word in persons_names:
      #   parsed = word.split()
      #   persons_names2.append(parsed[2])
      persons_names = [word.split()[2] for word in persons_names]
      rows_number = 0
      ignored_accuracy = 0
      ignored_len = 0
      ignored_words_number = 0
      ignored_digits = 0
      ignored_stopwords = 0
      ignored_nonunique = 0
      ignored_non_ascii = 0
      ignored_names = 0

      for row in reader:
        rows_number += 1

        query = row['UserMessage']
        etalon_question = row['EtalonQuestion']
        answer = row['Answer']
        #is_etalon = row['Etalon']
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

        #if float(accuracy) < .8:
        #  ignored_accuracy += 1
        #  continue

        parsed_query = parse_query(query)
        if parsed_query == '':
          continue

        # leaves only unique rows by user message and question
        # Uncomment this if you need to see unfiltered data.
        # hash = '{}_{}'.format(query, etalon_question)
        hash = '{}_{}'.format(parsed_query, etalon_question)
        if hash not in query_question_to_row_map:
          query_question_to_row_map[hash] = \
            {
              'query': query,
              'parsed_query': parsed_query,
              'question': etalon_question,
              'accuracy': accuracy,
              'date': created_date,
              'answer': answer
            }
        elif query_question_to_row_map[hash]['accuracy'] == accuracy:
          # if there is row with the same hash already, leave the variant with
          # the latest date
          updated_created_date = query_question_to_row_map[hash]['date']
          if updated_created_date < created_date:
            updated_created_date = created_date
          # print('{} : {}\t{}, time: {}, {}'.format(
          # hash, parsed_rows[hash][2], accuracy, parsed_rows[hash][3], created))
          query_question_to_row_map[hash] = \
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

          # TODO: Handle domain specific words, phrases, and acronyms
          # TODO: Locate and correct common typos and misspellings

      self.logging.debug(
          '%d of %d lines parsed', len(query_question_to_row_map), rows_number)
      self.logging.debug('\n\
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

      return query_question_to_row_map

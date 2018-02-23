from autocorrect import spell
import csv
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tree import Tree
import re

# TODO: download only once
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')


class NLProcessor:

  def __init__(self, logging):
    self.logging = logging
    self.stop_words = stopwords.words('english')
    #print(*sorted(self.stop_words), sep='\n')
    self.stop_words_ru = stopwords.words('russian')
    #print(*sorted(self.stop_words_ru), sep='\n')
    # TODO: add other languages
    self.persons_names = self.load_persons_names()
    self.acronyms = self.load_acronyms()


  # TODO: parse non-english data (now non-ascii is just ignored).
  def parse(self, path, delimiter='\t'):
    """Cleans up text corpus from csv file with given delimiter:
      - filters non-ascii symbols
      - strips and lowers every row
      - removes not word characters or spaces
      - removes rows containing less than 2 symbols
      - removes rows containing only one word
      - removes rows containing only digits and whitespaces
      - corrects words spelling
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

    with open(path, 'r') as csvfile:
      reader = csv.DictReader(csvfile, delimiter=delimiter)

      query_question_to_row_map = {}

      # Used for statistics
      self.ignored_accuracy = 0
      self.ignored_len = 0
      self.ignored_words_number = 0
      self.ignored_digits = 0
      self.ignored_stopwords = 0
      self.ignored_nonunique = 0
      self.ignored_non_ascii = 0
      self.ignored_names = 0
      self.corrected_spelling = 0
      rows_number = 0
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
        if not self.is_ascii(query):
          continue
        if not self.is_ascii(etalon_question):
          continue
        if not self.is_ascii(answer):
          continue

        # if accuracy is None or not self.is_number(accuracy):
        #   self.ignored_accuracy += 1
        #   continue

        #if float(accuracy) < .8:
        #  self.ignored_accuracy += 1
        #  continue

        parsed_query = self.parse_query(query)
        if parsed_query == '':
          continue

        # leaves only unique rows by user message and question
        # Uncomment this if you need to see unfiltered data.
        # hash = '{}_{}'.format(query, etalon_question)
        hash = '{}_{}'.format(parsed_query, etalon_question)
        if hash not in query_question_to_row_map:
          query_question_to_row_map[hash] = {
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
          query_question_to_row_map[hash] = {
              'query': query,
              'parsed_query': parsed_query,
              'question': etalon_question,
              'accuracy': accuracy,
              'date': updated_created_date,
              'answer': answer
            }
          self.ignored_nonunique += 1
        else:
          self.ignored_nonunique += 1

      self.logging.debug(
          '%d of %d lines parsed', len(query_question_to_row_map), rows_number)
      self.logging.debug('\n\
          ignored_stopwords=%d\n\
          ignored_len=%d\n\
          ignored_words_number=%d\n\
          ignored_digits=%d\n\
          ignored_non_ascii=%d\n\
          ignored_accuracy=%d\n\
          ignored_nonunique=%d\n\
          corrected_spelling=%d',
                         self.ignored_stopwords,
                         self.ignored_len,
                         self.ignored_words_number,
                         self.ignored_digits,
                         self.ignored_non_ascii,
                         self.ignored_accuracy,
                         self.ignored_nonunique,
                         self.corrected_spelling
                         )

      return query_question_to_row_map


  def is_number(self, s):
    try:
      float(s)
      return True
    except ValueError:
      return False


  def is_ascii(self, phrase):
    try:
      phrase.encode('ascii')
      return True
    except UnicodeEncodeError:
      self.ignored_non_ascii += 1
      return False


  def filter_stop_words(self, phrase, stop_words):
    words = phrase.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words).strip()


  def detect_names_using_chunking(self, phrase):
    """See http://www.nltk.org/book/ch07.html
      Does not recognise lowercase.
      Usage:
      detect_names_using_chunking('Summer School Coordinator Katrina Langdon')
    """
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(phrase))):
      if type(chunk) == Tree:
        if (chunk.label() == 'GPE' or chunk.label() == 'PERSON') \
            and chunk[0][1] == 'NNP':
          #if chunk.label() == 'PERSON' and chunk[0][1] == 'NNP':
          #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
          return True

    return False


  def detect_persons_names(self, phrase, names):
    """Doesn't understand some non-english names like Sergiy"""
    words = phrase.split()
    for word in words:
      if word in names:
        #self.logging.debug('%s -> %s', word, phrase)
        return True
    return False


  def correct_spelling(self, phrase, exceptions={}):
    words = phrase.split()
    filtered_words = []
    for word in words:
      if word not in exceptions:
        if not self.is_number(word):
          word = spell(word)
      filtered_words.append(word)
    corrected_spelling = ' '.join(filtered_words).lower()
    if phrase != corrected_spelling:
      self.logging.debug("%s -> %s", phrase, corrected_spelling)
      self.corrected_spelling += 1
    return corrected_spelling


  def load_persons_names(self):
    # downloaded from http://www.outpost9.com/files/WordLists.html
    # crl-names - filters names like How, Holiday:(
    # Given-Names - filters names like Meeting, Car:(

    # Downloaded form https://goo.gl/EwBq8s
    # TODO: find better dataset.
    names_corpus = 'data/census.gov.names.txt'
    persons_names = [
      line.rstrip('\n').lower() for line in open(names_corpus)
      if not line.startswith('#')
    ]
    return [word.split()[2] for word in persons_names]


  def load_acronyms(self):
    return [line.rstrip('\n').lower() for line in open('data/acronyms.txt')]


  def remove_stop_words(self, phrase):
    # token_words = phrase.split()

    # removes rows containing only stop words
    # if all(word in self.stop_words for word in token_words):
    #   self.ignored_stopwords += 1
    #   return ''

    parsed_phrase = self.filter_stop_words(phrase, self.stop_words)
    if len(parsed_phrase) == 0:
      self.ignored_stopwords += 1
      return ''

    # if all(word in self.stop_words_ru for word in token_words):
    #   self.ignored_stopwords += 1
    #   return ''

    parsed_phrase = self.filter_stop_words(parsed_phrase, self.stop_words_ru)
    if len(parsed_phrase) == 0:
      self.ignored_stopwords += 1
      return ''

    return parsed_phrase


  def parse_query(self, phrase):
    # strips and lowers every row
    parsed_phrase = phrase.strip().lower() if phrase is not None else ''

    # removes not word characters or spaces
    parsed_phrase = re.sub(r'[^\w\s]', '', parsed_phrase).strip()

    # removes rows containing less than 2 symbols
    if (len(parsed_phrase)) < 2:
      self.ignored_len += 1
      return ''

    # token_words = word_tokenize(token)
    token_words = parsed_phrase.split()

    # removes rows containing only numbers and whitespaces
    if all(self.is_number(word) for word in token_words):
      self.ignored_digits += 1
      return ''

    # Correct word spelling. Too slow!
    # Improves accuracy from 0.9278 to 9297
    # TODO: add tech terms and names to exceptions.
    #parsed_phrase = self.correct_spelling(parsed_phrase, self.acronyms)

    parsed_phrase = self.remove_stop_words(parsed_phrase)

    # removes rows containing only one word
    token_words = parsed_phrase.split()
    if len(token_words) <= 1:
      self.ignored_words_number += 1
      return ''

    # TODO: too slow
    # if self.detect_names_using_chunking(phrase):
    #   self.ignored_names += 1
    #   return ''

    if self.detect_persons_names(parsed_phrase, self.persons_names):
      self.ignored_names += 1
      return ''

    return parsed_phrase


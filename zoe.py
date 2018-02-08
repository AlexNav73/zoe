import nltk
import numpy
import csv
import string
import re
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')
path = 'data/History.rpt'

with open(path, 'r') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')

  uniqueRows = []
  stop_words = stopwords.words('english')
  stop_words_ru = stopwords.words('russian')
  row_len = 0
  for row in reader:
    row_len += 1
    token = row['Message'].strip().lower() if row['Message'] is not None else ''

    # replaces not  word characters or spaces with the empty string
    token = re.sub(r'[^\w\s]', '', token)

    if (len(token)) < 2:
      continue

    # if token.isdigit():
    #  continue

    # token_words = word_tokenize(token)
    token_words = token.split()
    if len(token_words) <= 1:
      continue

    if all(word.isdigit() for word in token_words):
      continue

    if all(word in stop_words for word in token_words):
      continue

    if all(word in stop_words_ru for word in token_words):
      continue

    if token not in uniqueRows:
      uniqueRows.append(token)

    # todo: Handling of domain specific words, phrases, and acronyms
    # todo: Locating and correcting common typos and misspellings

uniqueRows = sorted(uniqueRows)

print(*uniqueRows, sep="\n")
print('{} of {} lines filtered'.format(len(uniqueRows), row_len))

with open("data/cleaned_history.txt", "w") as clean_history_output:
  clean_history_output.write('\n'.join(uniqueRows))
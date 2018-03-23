
import sys

from abc import ABC, abstractmethod
from enum import Enum

class ChannelBase(ABC):

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def write(self, data):
        raise NotImplementedError


class StdioChannel(ChannelBase):
    
    def read(self):
        return input()

    def write(self, data):
        # sys.stdout.write(data)
        # sys.stdout.flush()
        print(data)


class Command(Enum):
    START = 0
    SIMILARITY = 1
    EXIT = 2


class Service:

    def __init__(self, channel, nlp, model):
        self.channel = channel
        self.nlp = nlp
        self.model = model

    def start(self):
        command = Command.START
        while command != Command.EXIT:
            data = self.channel.read()
            # command = data.command
            # question = self.nlp.parse_query(data.message)
            question = self.nlp.parse_query(data)

            if question == '':
                self.channel.write({ 'message': "Invalid input" })
                continue

            similar_question, similarity = self.model.predict(question)
            self.channel.write({ 'message': similar_question, 'similarity': similarity })


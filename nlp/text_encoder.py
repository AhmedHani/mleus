# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

from flair.data import Sentence
from flair.embeddings import WordEmbeddings, BertEmbeddings, ELMoEmbeddings


class TextEncoder(object):

    def __init__(self, vocab2indexes=None, char2indexes=None, modelname='char_one_hot'):
        if isinstance(modelname, str):
            if modelname == 'char_one_hot':
                self.model = _CharOneHotLoader(char2indexes)
            elif modelname == 'char_index':
                self.model = _CharIndexLoader(char2indexes)
            elif modelname == 'word_one_hot':
                self.model = _WordOneHotLoader(vocab2indexes)
            elif modelname == 'char_embedding':
                self.model = _CharEmbeddingLoader(modelname)
            elif modelname == 'word_index':
                self.model = _WordIndexLoader(vocab2indexes)
            elif modelname in ['bert', 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased']:
                self.model = _BERTEmbedding(modelname)
            elif modelname in ['elmo', 'elmo-small', 'elmo-medium', 'elmo-original']:
                self.model = _ELMoEmbedding(modelname)
            else:
                self.model = _WordEmbeddingLoader(modelname)
        else:
            import inspect

            if isinstance(modelname, object) or inspect.isclass(modelname):
                self.model = modelname

    def encode(self, text):
        return self.model.encode(text)

    def encoding_size(self):
        return self.model.encoding_size()


class _WordEmbeddingLoader(object):

    def __init__(self, model_name):
        self.__load_embeddings(model_name)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_vectors = []

                sentence = Sentence(sentence)

                self.__model.embed(sentence)

                for token in sentence:
                    sentence_vectors.append(token.embedding.tolist())

                text_matrix.append(sentence_vectors)

            return text_matrix
        else:
            sentence_matrix = []

            sentence = Sentence(text)
            self.__model.embed(sentence)

            for token in sentence:
                sentence_matrix.append(token.embedding.tolist())

            return sentence_matrix

    def encoding_size(self):
        return self.__embedding_size

    def __load_embeddings(self, model_name):
        self.__model = WordEmbeddings(model_name)

        self.__embedding_size = self.__model.embedding_length

    def __len__(self):
        return self.__embedding_size


class _WordIndexLoader(object):
    def __init__(self, word2indexes):
        self.word2indexes = word2indexes
        self.max_index = max(self.word2indexes.values())

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_matrix = []

                for word in sentence.split():
                    sentence_matrix.append(self[word])

                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for word in text.split():
                sentence_matrix.append(self[word])

            return sentence_matrix

    def __getitem__(self, item):
        if item == '<sos>':
            return 0

        if item == '<eos>':
            return 1

        if item == '<pad>':
            return 2

        if item in self.word2indexes:
            return self.word2indexes[item]
        else:
            return 3

    def encoding_size(self):
        return max(self.word2indexes.values()) + 4


class _WordOneHotLoader(object):

    def __init__(self, word2indexes):
        self.word2indexes = word2indexes
        self.word_size = len(word2indexes)
        self.temp_vec = self.__one_hot_vec(self.word_size)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_matrix = []

                for word in sentence.split():
                    sentence_matrix.append(self[word])

                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for word in text.split():
                sentence_matrix.append(self[word])

            return sentence_matrix

    @staticmethod
    def __one_hot_vec(word_size):
        return [0.0] * word_size

    def __getitem__(self, item):
        import copy as cp

        if item in self.word2indexes:
            vec = cp.deepcopy(self.temp_vec)
            vec[self.word2indexes[item]] = 1.0

            return vec
        else:
            return self.temp_vec

    def __len__(self):
        return self.word_size


class _CharOneHotLoader(object):

    def __init__(self, char2indexes):
        self.char2indexes = char2indexes
        self.char_size = len(char2indexes)
        self.temp_vec = self.__one_hot_vec(self.char_size)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_matrix = []

                for char in list(sentence):
                    sentence_matrix.append(self[char])

                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for char in list(text):
                sentence_matrix.append(self[char])

            return sentence_matrix

    @staticmethod
    def __one_hot_vec(char_size):
        return [0.0] * char_size

    def __getitem__(self, item):
        import copy as cp

        if item in self.char2indexes:
            vec = cp.deepcopy(self.temp_vec)
            vec[self.char2indexes[item]] = 1.0

            return vec
        else:
            return self.temp_vec

    def __len__(self):
        return self.char_size


class _CharIndexLoader(object):

    def __init__(self, char2indexes):
        self.char2indexes = char2indexes
        self.char_size = len(char2indexes)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_matrix = []

                for char in list(sentence):
                    sentence_matrix.append(self[char])

                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for char in list(text):
                sentence_matrix.append(self[char])

            return sentence_matrix

    def __getitem__(self, item):
        if item in self.char2indexes:
            return self.char2indexes[item]
        else:
            return 0

    def __len__(self):
        return self.char_size


class _CharEmbeddingLoader(object):

    def __init__(self, *args):
        pass


class _BERTEmbedding(object):

    def __init__(self, model_name):
        self.__load_embeddings(model_name)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_vectors = []

                sentence = Sentence(sentence)

                self.__model.embed(sentence)

                for token in sentence:
                    sentence_vectors.append(token.embedding.tolist())

                text_matrix.append(sentence_vectors)

            return text_matrix
        else:
            sentence_matrix = []

            sentence = Sentence(text)
            self.__model.embed(sentence)

            for token in sentence:
                sentence_matrix.append(token.embedding.tolist())

            return sentence_matrix

    def encoding_size(self):
        return self.__embedding_size

    def __load_embeddings(self, model_name):
        if model_name == 'bert':
            self.__model = BertEmbeddings()
        else:
            self.__model = BertEmbeddings(model_name)

        self.__embedding_size = self.__model.embedding_length

    def __len__(self):
        return self.__embedding_size


class _ELMoEmbedding(object):

    def __init__(self, model_name):
        self.__load_embeddings(model_name)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_vectors = []

                sentence = Sentence(sentence)

                self.__model.embed(sentence)

                for token in sentence:
                    sentence_vectors.append(token.embedding.tolist())

                text_matrix.append(sentence_vectors)

            return text_matrix
        else:
            sentence_matrix = []

            sentence = Sentence(text)
            self.__model.embed(sentence)

            for token in sentence:
                sentence_matrix.append(token.embedding.tolist())

            return sentence_matrix

    def encoding_size(self):
        return self.__embedding_size

    def __load_embeddings(self, model_name):
        if model_name == 'elmo':
            self.__model = ELMoEmbeddings()
        else:
            model_name = model_name.replace('elmo-', '')

            self.__model = ELMoEmbeddings(model_name)

        self.__embedding_size = self.__model.embedding_length

    def __len__(self):
        return self.__embedding_size

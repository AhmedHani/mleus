# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The mleus Project, https://github.com/AhmedHani/mleus"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

import os
import re
import sys
import spacy
import codecs
import string
import copy as cp
from functools import reduce
from collections import Counter
from types import SimpleNamespace


class TextAnalyzer:

    def __init__(self, data, outpath):
        self.data = data

        self.out = self.__set_output_location(outpath)

    def analyze(self, n_instances=True,
                avg_n_words=True,
                avg_n_chars=True,
                n_unique_words=True,
                n_unique_chars=True,
                words_freqs=True,
                chars_freqs=True,
                words2index=True,
                chars2index=True):
        results = {}
        average_words, average_chars = 0, 0
        words, chars = {}, {}

        for sentence in self.data:
            sentence_tokens = sentence.split()
            chars_tokens = list(sentence)

            average_words += len(sentence_tokens)
            average_chars += len(chars_tokens)

            for word in sentence_tokens:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

            for char in chars_tokens:
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1

        if n_instances:
            self.out.write('number of instances: {}\n'.format(len(self.data)))
            results['n_instances'] = len(self.data)

        if avg_n_words:
            self.out.write('average number of words: {}\n'.format(average_words // len(self.data)))
            results['avg_n_words'] = average_words // len(self.data)

        if avg_n_chars:
            self.out.write('average number of chars: {}\n'.format(average_chars // len(self.data)))
            results['avg_n_chars'] = average_chars // len(self.data)

        if n_unique_words:
            self.out.write('number of unique words: {}\n'.format(len(words)))
            results['n_unique_words'] = len(words)

        if n_unique_chars:
            self.out.write('number of unique chars: {}\n'.format(len(chars)))
            results['n_unique_chars'] = len(chars)

        if words_freqs:
            words = Counter(words).most_common(len(words))

            freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in words])
            self.out.write('words frequencies: \n{}\n\n'.format(freqs_format))

            results['words_freqs'] = {item[0]: item[1] for item in words}

        if chars_freqs:
            chars = Counter(chars).most_common(len(chars))

            freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in chars])
            self.out.write('chars frequencies: \n{}\n\n'.format(freqs_format))

            results['chars_freqs'] = {item[0]: item[1] for item in chars}

        if words2index:
            results['words2index'] = {word: i for i, word in enumerate(words.keys(), start=4)}
            results['index2words'] = {i: word for i, word in enumerate(words.keys(), start=4)}

        if chars2index:
            results['chars2index'] = {char: i for i, char in enumerate(chars.keys(), start=4)}
            results['index2chars'] = {i: char for i, char in enumerate(chars.keys(), start=4)}

        return SimpleNamespace(**results)

    def n_instances(self):
        self.out.write('number of instances: {}\n'.format(len(self.data)))

        return len(self.data)

    def avg_n_words(self):
        average_words = sum([len(sentence.split()) for sentence in self.data]) // len(self.data)

        self.out.write('average number of words: {}\n'.format(average_words))

        return average_words

    def avg_n_chars(self):
        average_chars = sum([len(list(sentence)) for sentence in self.data]) // len(self.data)

        self.out.write('average number of words: {}\n'.format(average_chars))

        return average_chars

    def n_unique_words(self):
        words = set()

        for sentence in self.data:
            for word in sentence.split():
                words.add(word)

        self.out.write('number of unique words: {}\n'.format(len(words)))

        return len(words)

    def n_unique_chars(self):
        chars = set()

        for sentence in self.data:
            for char in list(sentence):
                chars.add(char)

        self.out.write('number of unique chars: {}\n'.format(len(chars)))

        return len(chars)

    def words_freqs(self):
        words = {}

        for sentence in self.data:
            for word in sentence.split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        words = Counter(words).most_common(len(words))

        freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in words])
        self.out.write('words frequencies: \n{}\n\n'.format(freqs_format))

        return {item[0]: item[1] for item in words}

    def chars_freqs(self):
        chars = {}

        for sentence in self.data:
            for char in list(sentence):
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1

        chars = Counter(chars).most_common(len(chars))

        freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in chars])
        self.out.write('chars frequencies: \n{}\n\n'.format(freqs_format))

        return {item[0]: item[1] for item in chars}

    @staticmethod
    def __set_output_location(outpath):
        return codecs.open(outpath, 'w', encoding='utf-8')


class LinguisticAnalyzer(object):

    def __init__(self):
        self.__nlp_instance = spacy.load('en_core_web_sm')

    def get_nouns(self, text):
        if isinstance(text, list):
            return [' '.join([noun.text for noun in self.__nlp_instance(sentence).noun_chunks]) for sentence in text]

        return ' '.join([noun.text for noun in self.__nlp_instance(text).noun_chunks])

    def get_lemma(self, text):
        if isinstance(text, list):
            return [' '.join([token.lemma_ for token in self.__nlp_instance(sentence)]) for sentence in text]

        return ' '.join([token.lemma_ for token in self.__nlp_instance(text)])

    def get_named_entities(self, text):
        if isinstance(text, list):
            return [[(token.text, token.label_) for token in self.__nlp_instance(sentence).ents] for sentence in text]

        return [(token.text, token.label_) for token in self.__nlp_instance(text).ents]

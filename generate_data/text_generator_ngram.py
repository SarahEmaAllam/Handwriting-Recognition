# Modified from this tutorial on N-grams: https://towardsdatascience.com/text-generation-using-n-gram-model-8d12d9802aa0
import string
import random
import time
from typing import List
import pandas as pd
import os

PATH = os.getcwd()


def tokenize(text: str) -> List[str]:
    """
    :param text: Takes input sentence
    :return: tokenized sentence
    """
    for punct in string.punctuation.replace("-", ''):
        text = text.replace(punct, ' ' + punct + ' ')
    t = text.split()
    return t


def get_ngrams(n: int, tokens: list) -> list:
    """
    :param n: n-gram size
    :param tokens: tokenized sentence
    :return: list of ngrams

    ngrams of tuple form: ((previous wordS!), target word)
    """
    # tokens.append('<END>')
    tokens = (n - 1) * ['<START>'] + tokens
    l = []
    for i in range(n - 1, len(tokens)):
        tu = tuple()
        for p in reversed(range(n - 1)):
            tu = tu + (tokens[i - p - 1],)
        l.append((tu, tokens[i]))
    return l


class NgramModel(object):

    def __init__(self, n):
        self.n = n

        # dictionary that keeps list of candidate words given context
        self.context = {}

        # keeps track of how many times ngram has appeared in the text before
        self.ngram_counter = {}

    def update(self, sentence: str, posterior_prob) -> None:
        """
        Updates Language Model
        :param sentence: input text
        """
        n = self.n
        ngrams = get_ngrams(n, tokenize(sentence))
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            # TODO: UPDATE THE N-GRAM TO INCLUDE POSTERIOR PROBABILITY BY FREQUENCY OF WORD
            # posterior_prob_extr = posterior_prob[posterior_prob.iloc[:,0] == ngram[0].replace(" ", "_")]
            # print("posterior prob : ", posterior_prob_extr)
            # self.ngram_counter[ngram] *= posterior_prob_extr
            # print(f"UPDATED ngram_counter of {ngram} is {self.ngram_counter[ngram]}")
            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

    def prob(self, context, token):
        """
        Calculates probability of a candidate token to be generated given a context
        :return: conditional probability
        """
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result

    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        :param context:
        :return:
        """
        r = random.random()
        map_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            map_to_probs[token] = self.prob(context, token)

        summ = 0
        # weight the summ by the posterior probability of the class appearing
        for token in sorted(map_to_probs):
            summ += map_to_probs[token]
            if summ > r:
                return token

    def generate_text(self, token_count: int):
        """
        :param token_count: number of words to be produced
        :return: generated text
        """
        n = self.n
        context_queue = (n - 1) * ['<START>']
        result = []
        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '.':
                    context_queue = (n - 1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)


def create_ngram_model(n, path):
    m = NgramModel(n)

    df = pd.read_excel(path)
    posterior_prob = df[["Names", "Probabilities"]]
    text = df["Names"]
    text = text.str.replace('_', ' ')
    text = text.str.replace('Tsadi', 'Tsadi-medial')
    text = text.str.replace('Tasdi-final', 'Tsadi-final')
    text = text.values.tolist()

    for sentence in text:
        # add back the fullstop
        if sentence:
            sentence += '.'
            m.update(sentence, posterior_prob)
    return m, posterior_prob


def generator(TEXT_LENGTH, NGRAM_SIZE):
    start = time.time()
    # m, posterior_prob = create_ngram_model(ngram_size, os.path.join(PATH, 'generate_data','ngrams_frequencies_withNames_prob.xlsx'))
    m, posterior_prob = create_ngram_model(NGRAM_SIZE, os.path.join(PATH,
                                                                    'ngrams_frequencies_withNames_prob.xlsx'))

    print(f'Language Model creating time: {time.time() - start}')
    # random.seed(7)
    print(f'{"=" * 50}\nGenerated text:')
    text = m.generate_text(TEXT_LENGTH)
    print("TEXT")
    print(text)
    print(f'{"=" * 50}')
    return text

# call generator
# generator(1000, 4)

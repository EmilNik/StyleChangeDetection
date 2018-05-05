import numpy as np
import pyphen
import math
import csv
import dill
from nltk import sent_tokenize, word_tokenize, Text, pos_tag, ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from collections import Counter, OrderedDict


with open('../data/most_common_pos_tag_trigrams.csv', 'r') as f:
    MOST_COMMON_POS_TAG_TRIGRAMS = []
    reader = csv.reader(f)
    for line in reader:
        MOST_COMMON_POS_TAG_TRIGRAMS.append(tuple(line))


with open('../data/vectorizer_500.pk', 'rb') as f:
    VECTORIZER = dill.load(f)


def _load_dale_chall_words():
    return set([word.rstrip('\n') for word in open('../data/dale_chall_words.txt')])


# Feature extraction
class StylometryExtractor:
    DALE_CHALL_WORDS = _load_dale_chall_words()
    TOKENIZER = RegexpTokenizer(r"\w+'\w+|\w+")
    SPECIAL_CHAR = '@<:@'

    def __init__(self, text):
        self.raw_text = text
        self.raw_text_length = len(text)
        self.number_of_letters = len([x for x in self.raw_text if x.isalpha() or x.isdigit()])
        self.words = StylometryExtractor.TOKENIZER.tokenize(self.raw_text)
        self.tokens = word_tokenize(self.raw_text)
        self.number_of_words = len(self.words)
        self.number_of_tokens = len(self.tokens)
#         self.text = Text(word_tokenize(self.raw_text))
        self.words_frequency = FreqDist(Text(self.words))
        self.tokens_frequency = FreqDist(Text(self.tokens))
        self.sentences = sent_tokenize(self.raw_text)
        self.number_of_sentences = len(self.sentences)
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.raw_text.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]
        self.all_trigrams = self._all_trigrams()
        self.ngram_string = self._to_ngram_string()
        self.features = self._to_dict()
        self.feature_names = list(self.features.keys())

    def _to_ngram_string(self):
        cleared_text = ' '.join([word for word in self.words if word not in stopwords.words('english')])
        return StylometryExtractor.SPECIAL_CHAR.join(
            ''.join(ngram) for ngram in ngrams(cleared_text, 4) if ' ' not in ngram and '\n' not in ngram)

    def term_per_thousand(self, term):
        return self.words_frequency[term] * 1000 / self.words_frequency.N()

    def char_per_thousand(self, char):
        return self.raw_text.count(char) / self.raw_text_length * 1000

    def chars_per_thousand(self, chars):
        return sum([self.char_per_thousand(char) for char in chars])

    def syllables_per_thousand(self):
        return self.get_number_syllables() / self.raw_text_length * 1000

    def get_number_syllables(self):
        dic = pyphen.Pyphen(lang='en')
        return sum([len(dic.inserted(word).split("-")) for word in self.words])

    def get_number_pollisyllable_words(self):
        dic = pyphen.Pyphen(lang='en')
        return len([word for word in self.words if len(dic.inserted(word).split("-")) >= 3])

    def get_words_longer_than_X(self, x):
        return len([word for word in self.words if len(word) >= x])

    def mean_of_syllables_per_word(self):
        return self.get_number_syllables() / self.number_of_words

    def num_of_words_with_more_than_three_syllables_per_thousand(self):
        return self.get_number_pollisyllable_words() / self.number_of_words * 1000

    def get_flesch_reading_ease(self):
        # http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        """
        90.0- 100.0 - sily understood by an average 11-year-old student
        60.0 - 70.0 - easily understood by 13- to 15-year-old students
        0.00 - 30.0 -  best understood by university graduates
        """
        return 206.835 - 1.015 * self.number_of_words / self.number_of_sentences - 84.6 * self.get_number_syllables() / self.number_of_words

    def flesch_kincaid_grade_level(self):
        # http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        """
            It is more or less the number of years of education generally required to understand this text.
            The lowest grade level score in theory is -3.40.
        """
        return 0.39 * self.number_of_words / self.number_of_sentences + 11.8 * self.get_number_syllables() / self.number_of_words - 15.59

    def get_coleman_liau_index(self):
        # http://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        """
             It approximates the U.S. grade level thought necessary to comprehend the text.
        """
        return 5.89 * self.number_of_letters / self.number_of_words - 29.6 * self.number_of_sentences / self.number_of_words - 15.8

    def get_gunning_fog_index(self):
        # http://en.wikipedia.org/wiki/Gunning_fog_index
        """
        The index estimates the years of formal education needed to understand the text on a first reading
        """
        return 0.4 * (self.number_of_words / self.number_of_sentences + 100.0 * self.get_number_pollisyllable_words() / self.number_of_words)

    def get_smog_index(self):
        # http://en.wikipedia.org/wiki/SMOG
        """
            Simple Measure of Gobbledygook (SMOG) is a simplification of Gunning Fog, also estimating the years of formal education needed
            to understand a text
        """
        return 1.043 * math.sqrt(self.get_number_pollisyllable_words() * 30.0 / self.number_of_sentences) + 3.1291

    def get_ari_index(self):
        # http://en.wikipedia.org/wiki/Automated_Readability_Index
        """
            It produces an approximate representation of the US grade level needed to comprehend the text.
        """
        return 4.71 * self.number_of_letters / self.number_of_words + 0.5 * self.number_of_words / self.number_of_sentences - 21.43

    def get_lix_index(self):
        # http://en.wikipedia.org/wiki/LIX
        # http://www.readabilityformulas.com/the-LIX-readability-formula.php
        """
            Value interpretation:
            Very Easy      - 20, 25
            Easy           - 30, 35
            Medium         - 40. 45
            Difficult      - 50, 55
            Very Difficult - 60+
        """
        long_words = self.get_words_longer_than_X(6)
        number_of_periods = self.number_of_sentences + self.tokens_frequency[':'] + self.tokens_frequency[';']
        return self.number_of_words / number_of_periods + 100.0 * long_words / self.number_of_words

    def number_of_dale_chall_difficult_words(self):
        return len([word for word in self.words if word not in StylometryExtractor.DALE_CHALL_WORDS])

    def get_dale_chall_score(self):
        # http://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
        """
            4.9 or lower    ---  easily understood by an average 4th-grade student or lower
            5.0–5.9         ---  easily understood by an average 5th or 6th-grade student
            6.0–6.9         ---  easily understood by an average 7th or 8th-grade student
            7.0–7.9         ---  easily understood by an average 9th or 10th-grade student
            8.0–8.9         ---  easily understood by an average 11th or 12th-grade student
            9.0–9.9         ---  easily understood by an average 13th to 15th-grade (college) student
            10.0 or higher  ---  easily understood by an average college graduate
        """
        return 15.79 * self.number_of_dale_chall_difficult_words() / self.number_of_words + 0.0496 * self.number_of_words / self.number_of_sentences

    def get_dale_chall_known_fraction(self):
        """
            Computes the fraction of easy words in the text, i.e., the fraction of words that could be found in the
            dale chall list of 3.000 easy words.
        """
        return 1.0 - self.number_of_dale_chall_difficult_words() / self.number_of_words

    def mean_sentence_len(self):
        return np.mean(self.sentence_word_length)

    def std_sentence_len(self):
        return np.std(self.sentence_word_length)

    def mean_paragraph_len(self):
        return np.mean(self.paragraph_word_length)

    def std_paragraph_len(self):
        return np.std(self.paragraph_word_length)

    def mean_word_len(self):
        word_chars = [len(word) for word in self.words]
        return sum(word_chars) / len(word_chars)

    def unique_words_ratio(self):
        return len(set(self.words)) / self.number_of_words * 100

#     def get_byte_ngrams(self, number_of_bytes):
    @classmethod
    def to_pos_tags(cls, sentence):
        tokens = StylometryExtractor.TOKENIZER.tokenize(sentence)
        pos_tags = list(map(lambda x: x[1], pos_tag(tokens)))
        return ['__START__'] + pos_tags + ['__END__']

    @classmethod
    def pos_tag_trigrams(cls, sentence):
        pos_tags = StylometryExtractor.to_pos_tags(sentence)
        return [(x, y, z) for x, y, z in zip(pos_tags, pos_tags[1:], pos_tags[2:])]

    def _all_trigrams(self):
        return Counter(trigram
            for sentence in self.sentences
            for trigram in StylometryExtractor.pos_tag_trigrams(sentence)
        )

    def pos_tag_percents(self):
        number_of_trigrams = sum(self.all_trigrams.values())
        return {
            '_'.join(trigram): self.all_trigrams[trigram] / number_of_trigrams
            for trigram in MOST_COMMON_POS_TAG_TRIGRAMS
        }

    def char_ngrams_tf_idf(self):
        return dict(zip(
            VECTORIZER.get_feature_names(),
            VECTORIZER.transform([self.ngram_string]).toarray()[0]
        ))

    def to_dict(self):
        return self.features

    def to_vector(self):
        return list(self.features.values())

    def _to_dict(self):
        features = {
            'Lexical diversity' : self.unique_words_ratio(),
            'Mean Word Length' : self.mean_word_len(),
            'Mean Sentence Length' : self.mean_sentence_len(),
            'STDEV Sentence Length' : self.std_sentence_len(),
            'Mean paragraph Length' : self.mean_paragraph_len(),
            'Number of letters' : self.number_of_letters,
            'Flesch Reading Ease' : self.get_flesch_reading_ease(),
            'Flesch Kincaid Grade' : self.flesch_kincaid_grade_level(),
            'Coleman Liau Index' : self.get_coleman_liau_index(),
            'Gunning Fog Index' : self.get_gunning_fog_index(),
            'Smog Index' : self.get_smog_index(),
            'Ari Index' : self.get_ari_index(),
            'Lix Index' : self.get_lix_index(),
            'Dale Chall Score' : self.get_dale_chall_score(),
            'Dale Chall Known Fraction' : self.get_dale_chall_known_fraction(),
            'Punctuation' : self.chars_per_thousand(['.', ',', '!', ';', '?']),
            'Special characters' : self.chars_per_thousand(['%', '#', ')', '(', '@', '$', '^','&', '>', '<', '*', '_', '-','=', '-', '+', '/','\\', '\'', '"', '`']),
            'Commas' : self.term_per_thousand(','),
            'Semicolons' : self.term_per_thousand(';'),
            'Quotations' : self.term_per_thousand('\"'),
            'Exclamations' : self.term_per_thousand('!'),
            'Colons' : self.term_per_thousand(':'),
            'Hyphens' : self.term_per_thousand('-'),
            'Double Hyphens' : self.term_per_thousand('--'),
            'A' : self.char_per_thousand('a'),
            'B' : self.char_per_thousand('b'),
            'C' : self.char_per_thousand('c'),
            'D' : self.char_per_thousand('d'),
            'E' : self.char_per_thousand('e'),
            'F' : self.char_per_thousand('f'),
            'G' : self.char_per_thousand('g'),
            'H' : self.char_per_thousand('h'),
            'I' : self.char_per_thousand('i'),
            'J' : self.char_per_thousand('j'),
            'K' : self.char_per_thousand('k'),
            'L' : self.char_per_thousand('l'),
            'M' : self.char_per_thousand('m'),
            'N' : self.char_per_thousand('n'),
            'O' : self.char_per_thousand('o'),
            'P' : self.char_per_thousand('p'),
            'Q' : self.char_per_thousand('q'),
            'R' : self.char_per_thousand('r'),
            'S' : self.char_per_thousand('s'),
            'T' : self.char_per_thousand('t'),
            'U' : self.char_per_thousand('u'),
            'V' : self.char_per_thousand('v'),
            'W' : self.char_per_thousand('w'),
            'X' : self.char_per_thousand('x'),
            'Y' : self.char_per_thousand('y'),
            'Z' : self.char_per_thousand('z'),
            'Numbers' : self.chars_per_thousand(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']),
            'Syllables' : self.syllables_per_thousand(),
            'Mean syllables per word' : self.mean_of_syllables_per_word(),
            'Words with >= 3 syllables' : self.num_of_words_with_more_than_three_syllables_per_thousand(),
            }

        for stopword in stopwords.words('english'):
            features[stopword] = self.term_per_thousand(stopword)

        features.update(self.pos_tag_percents())
        features.update(self.char_ngrams_tf_idf())

        return OrderedDict(sorted(features.items(), key=lambda t: t[0]))

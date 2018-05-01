from .stylometry_extractor import StylometryExtractor
from nltk import ngrams


class TextChunk(StylometryExtractor):
    SPECIAL_CHAR = '@<:@'
    
    def __init__(self, text, special_char = SPECIAL_CHAR):
        super().__init__(text)
        self.special_char = special_char
        self.ngram_string = self._doc_to_ngram_string(special_char)

    def _doc_to_ngram_string(self):
        return self.special_char.join(''.join(ngram) for ngram in ngrams(self.text.lower(), 4) if ' ' not in ngram and '\n' not in ngram)
    
    def squared_difference_with(self, other):
        vector = self.to_dict()
        other_vector = other.to_dict()
        features = vector.keys()
        return {feature: (vector[feature] - other_vector[feature]) ** 2 for feature in features}

    def absolute_difference_with(self, other):
        vector = self.to_dict()
        other_vector = other.to_dict()
        features = vector.keys()
        return {feature: abs(vector[feature] - other_vector[feature]) for feature in features}
        
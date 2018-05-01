from .stylometry_extractor import StylometryExtractor


class TextChunk(StylometryExtractor):
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

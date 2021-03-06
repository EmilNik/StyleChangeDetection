from nltk import sent_tokenize, word_tokenize, pos_tag

class Document:
    MARKER = '\033[91m\u2588\033[0m'

    def __init__(self, text, changes=None, positions=None):
        self.text = text
        self.has_changes = changes
        self.positions = positions # character positions
        self._sentences = None
        self._sent_positions = None
        self._words = None
        self._word_positions = None
        self._sent_counts_before_style_change = None
        self._word_counts_before_style_change = None

    @property
    def sentences(self):
        if self._sentences is None:
            self._sentences, self._sent_positions = self._split_sentences()
        return self._sentences

    @property
    def sent_positions(self):
        if self._sent_positions is None:
            self._sentences, self._sent_positions = self._split_sentences()
        return self._sent_positions

    @property
    def words(self):
        if self._words is None:
            self._words, self._word_positions = self._split_words()
        return self._words

    @property
    def word_positions(self):
        if self._word_positions is None:
            self._words, self._word_positions = self._split_words()
        return self._word_positions

    @property
    def sent_counts_before_style_change(self):
        if self._sent_counts_before_style_change is None:
            self._sent_counts_before_style_change = []
            if not self.has_changes:
                return self._sent_counts_before_style_change
            last = 0
            for sent_pos in self.sent_positions:
                self._sent_counts_before_style_change.append(sent_pos - last)
                last = sent_pos
        return self._sent_counts_before_style_change

    @property
    def word_counts_before_style_change(self):
        if self._word_counts_before_style_change is None:
            self._word_counts_before_style_change = []
            if not self.has_changes:
                return self._word_counts_before_style_change
            last = 0
            for word_pos in self.word_positions:
                self._word_counts_before_style_change.append(sent_pos - last)
                last = word_pos
        return self._word_counts_before_style_change

    def _split_sentences(self):
        # split and map character positions to sentence positions
        sent_positions = []

        if not self.has_changes:
            #sentences = [pos_tag(word_tokenize(s)) for s in sent_tokenize(self.text)]
            sentences = sent_tokenize(self.text)
        else:
            sentences = []
            parts = [self.text[i:j] for i,j in zip([None]+self.positions, self.positions+[None])]
            for part in parts:
                #sentences += [pos_tag(word_tokenize(s)) for s in sent_tokenize(part)]
                sentences += sent_tokenize(part)
                sent_positions.append(len(sentences))
            sent_positions.pop()

        return sentences, sent_positions

    def _split_words(self):
        # split and map character positions to word positions
        word_positions = []

        if not self.has_changes:
            words = word_tokenize(self.text)
        else:
            words = []
            parts = [self.text[i:j] for i,j in zip([None]+self.positions, self.positions+[None])]
            for part in parts:
                words += word_tokenize(part)
                word_positions.append(len(words))
            word_positions.pop()

        return words, word_positions

    def __str__(self):
        marked_document = self.text
        for index, position in enumerate(self.positions):
            position += index * len(Document.MARKER)
            marked_document = marked_document[:position] + Document.MARKER + marked_document[position:]
        return marked_document

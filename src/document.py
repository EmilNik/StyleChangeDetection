from nltk import sent_tokenize, word_tokenize, pos_tag

class Document:
    MARKER = '\033[91m\u2588\033[0m'
    SPECIAL_CHAR = '@<:@'

    def __init__(self, text, changes=None, positions=None):
        self.text = text
        self.has_changes = changes
        self.positions = positions # character positions
        self.marked_document = self._mark_document()
        self.sentences, self.sent_positions = self._split_sentences()
        self.sent_counts_before_style_change = self._sent_counts_before_style_change()
        self.ngram_string = self._doc_to_ngram_string(SPECIAL_CHAR)

    def _mark_document(self):
        # construct __str__'s representation
        marked_document = self.text
        for index, position in enumerate(self.positions):
            position += index * len(Document.MARKER)
            marked_document = marked_document[:position] + Document.MARKER + marked_document[position:]
        return marked_document

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

    def _sent_counts_before_style_change(self):
        if not self.has_changes:
            return []
        last = 0
        res = []
        for sent_pos in self.sent_positions:
            res.append(sent_pos - last)
            last = sent_pos
        return res
    
    def _doc_to_ngram_string(self, special_char = '$'):
        return special_char.join(''.join(ngram) for ngram in ngrams(self.text.lower(), 4))

    def __str__(self):
        return self.marked_document

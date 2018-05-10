import os
import itertools
import argparse
import json
import numpy as np
import sklearn
import logging

log_format_string = '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=log_format_string, level=logging.DEBUG)

from stylometry_extractor import StylometryExtractor
from models import Models


def load_texts(path):
    logging.debug('Loading txt files from EVAL_DIR...')
    texts = []
    for i in itertools.count(start=1):
        try:
            text = open(os.path.join(path, 'problem-' + str(i) + '.txt'), 'r').read()
            texts.append(text)
        except FileNotFoundError:
            break
    logging.debug('Loaded %d txt files.' % (i-1))
    return texts

def load_scalers():
    logging.debug('Loading pickled scalers...')
    standard_scaler = sklearn.externals.joblib.load('../data/standard_scaler.pk')
    minmax_scaler = sklearn.externals.joblib.load('../data/minmax_scaler.pk')
    logging.debug('Done.')
    return dict(standard=standard_scaler, minmax=minmax_scaler)

def split_texts(texts):
    """Return list of tuples of text chunks"""
    # TODO better way?
    return list(map(lambda t: (t[:len(t)//3], t[len(t)//3:len(t)//3*2], t[len(t)//3*2:]), texts))

def to_vectors(splitted_texts, normalizer):
    """Converts list of tuples of text chunks to feature vectors (normalized)"""
    logging.debug('Extracting stylometry for each chunk in splitted texts...')
    all_chunks = []
    for text_chunks in splitted_texts:
        all_chunks.extend(list(map(lambda chunk: StylometryExtractor(chunk).to_vector(), text_chunks)))

    normed = normalizer( np.array(all_chunks) )

    res = []
    j = 0
    for i in range(len(splitted_texts)):
        cur_tuple_len = len(splitted_texts[i])
        res.append(tuple(normed[j:j+cur_tuple_len]))
        j += cur_tuple_len

    logging.debug('Done.')
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--eval-dir", required=True, help="List of problem instances, i.e., [filename].txt files.")
    parser.add_argument("-o", "--output-dir", required=True, help="List of truth files, one for each txt file in our --eval-dir.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        logging.warn('[%s] already exists! Will possibly override files in it!' % args.output_dir)
    else:
        logging.debug('Creating [%s].' % args.output_dir)
        os.makedirs(args.output_dir)


    texts = load_texts(args.eval_dir)
    scalers = load_scalers()
    models = Models()

    splitted_texts = split_texts(texts)

    # NOTE using standard scaler
    splitted_texts_vectors = to_vectors(splitted_texts, scalers['standard'].transform)

    logging.debug('Making predictions...')
    for i, chunks_vectors_tuple in enumerate(splitted_texts_vectors, 1):
        chunks_vectors_pairs = itertools.combinations(chunks_vectors_tuple, 2)
        confidences = [0, 0] # (for false, for true)
        for a, b in chunks_vectors_pairs:
            pred_for_false, pred_for_true = models.classify_proba(a, b)
            # TODO if very confident about style change - break or something?
            confidences[0] += pred_for_false
            confidences[1] += pred_for_true

        y_pred = bool(np.argmax(confidences))
        with open(os.path.join(args.output_dir, 'problem-%d.truth' % i), 'w') as f:
            json.dump({ 'changes': y_pred }, f)

    logging.debug('Done.')

if __name__ == "__main__":
    main()

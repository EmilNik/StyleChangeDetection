import os
import logging
import itertools
import argparse
import numpy as np
import sklearn

from stylometry_extractor import StylometryExtractor


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
    standard_scaler = sklearn.externals.joblib.load('../data/models/standard_scaler.pk')
    min_max_scaler = sklearn.externals.joblib.load('../data/models/min_max_scaler.pk')
    logging.debug('Done.')
    return dict(standard=standard_scaler, min_max=min_max_scaler)

def load_models():
    logging.debug('Loading pickled models...')
    random_forest = sklearn.externals.joblib.load('../data/models/random_forest.pk')
    # TODO other models
    logging.debug('Done.')
    return dict(random_forest=random_forest)

def split_texts(texts):
    """Return list of tuples of text chunks"""
    # TODO better way
    return list(map(lambda t: (t[:len(t)//2], t[len(t)//2:]), texts))

def to_vectors(splitted_texts, normalizer):
    """Converts list of tuples of text chunks to feature vectors (normalized)"""
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
    return res
    ## very ugly and probably kinda slow TODO optimize!
    #return [tuple( map(lambda chunk: normalizer(np.array(StylometryExtractor(chunk).to_vector())[np.newaxis,...]), t) ) for t in splitted_texts]

def classify(chunks_vectors_pairs, models):
    for model_name, clf in models.items():
        for A, B in chunks_vectors_pairs:
            y_pred = clf.predict(abs(A - B).reshape(1, -1))
            # TODO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--eval-dir", required=True, help="List of problem instances, i.e., [filename].txt files.")
    parser.add_argument("-o", "--output-dir", required=True, help="List of truth files, one for each txt file in our --eval-dir.")
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # uncomment me later
    #if os.path.exists(args.output_dir):
    #    raise Exception('%s already exists!' % args.output_dir)

    #os.makedirs(args.output_dir)


    texts = load_texts(args.eval_dir)
    scalers = load_scalers()
    models = load_models()

    splitted_texts = split_texts(texts)

    # NOTE using standard scaler
    splitted_texts_vectors = to_vectors(splitted_texts, scalers['standard'].transform)

    for i, chunks_vectors_tuple in enumerate(splitted_texts_vectors, 1):
        chunks_vectors_pairs = itertools.combinations(chunks_vectors_tuple, 2)
        y_pred = classify(chunks_vectors_pairs, models)
        # TODO output y_pred in a json file in OUTPUT_DIR

if __name__ == "__main__":
    main()

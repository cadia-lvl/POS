#! /usr/bin/env python3
"""
    ABLTagger: Augmented BiDirectional LSTM Tagger

    Script for doing 10-fold validation

    Copyright (C) 2019 Örvar Kárason and Steinþór Steingrímsson.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
import argparse
import sys
import logging

import numpy as np
import pos.data as data
from pos.data import Embeddings
import dynet_config
dynet_config.set(mem=23000, random_seed=42)
__license__ = "Apache 2.0"

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # HYPERPARAMETERS
    parser.add_argument('--optimization', '-o', help="Which optimization algorithm",
                        choices=['SimpleSGD', 'MomentumSGD', 'CyclicalSGD', 'Adam', 'RMSProp'], default='SimpleSGD')
    parser.add_argument('--learning_rate', '-lr',
                        help="Learning rate", type=float, default=0.13)
    parser.add_argument('--learning_rate_decay', '-lrd',
                        help="Learning rate decay", type=float, default=0.05)
    parser.add_argument('--learning_rate_max', '-l_max',
                        help="Learning rate max for Cyclical SGD", type=float, default=0.1)
    parser.add_argument('--learning_rate_min', '-l_min',
                        help="Learning rate min for Cyclical SGD", type=float, default=0.01)
    parser.add_argument('--dropout', '-d',
                        help="Dropout rate", type=float, default=0.0)
    # EXTERNAL DATA
    parser.add_argument('--data_folder', '-data',
                        help="Folder containing training data", default='./data/format/')
    parser.add_argument('--out_folder', '-out-data',
                        help="Folder containing results", default='./models')
    parser.add_argument('--use_morphlex', '-morphlex',
                        help="File with morphological lexicon embeddings in ./extra folder. Example file: ./extra/dmii.or", default='./data/format/dmii.vectors')
    parser.add_argument('--load_characters', '-load_chars',
                        help="File to load characters from", default='./extra/characters_training.txt')
    parser.add_argument('--load_coarse_tagset', '-load_coarse',
                        help="Load embeddings file for coarse grained tagset", default='./extra/word_class_vectors.txt')
    # TRAIN AND EVALUATE
    parser.add_argument(
        '--corpus', '-c', help="Name of training corpus", default='otb')
    parser.add_argument('--dataset_fold', '-fold',
                        help="select which dataset to use (1-10)", type=int, default=1)
    parser.add_argument('--epochs_coarse_grained', '-ecg',
                        help="How many epochs for coarse grained training? (12 is default)", type=int, default=12)
    parser.add_argument('--epochs_fine_grained', '-efg',
                        help="How many epochs for fine grained training? (20 is default)", type=int, default=20)
    parser.add_argument(
        '--noise', '-n', help="Noise in embeddings", type=float, default=0.1)
    # DEBUG
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    try:
        args = parser.parse_args()
    except ValueError:
        sys.exit(0)
    args = parser.parse_args()

    chars = data.read_vocab(args.load_characters)

    out_folder = args.out_folder

    # Read train and test data
    train_tokens, train_tags, _ = data.read_tsv(
        f'{args.data_folder}IFD-{args.dataset_fold:02}TM.tsv')
    test_tokens, test_tags, _ = data.read_tsv(
        f'{args.data_folder}IFD-{args.dataset_fold:02}PM.tsv')
    # Prepare the coarse tags
    train_tags_coarse = data.coarsify(train_tags)
    test_tags_coarse = data.coarsify(test_tags)

    # Read the coarse tagset as defined in the old days
    c_t_map, coarse_embedding = data.read_embedding(
        args.load_coarse_tagset)

    # Define the vocabularies and mappings
    c_map = data.VocabMap(chars, special_tokens=[
        (data.EOS, 0),  # We start from 0, and do not use default id
        (data.SOS, 1)  # We start from 0, and do not use default id
    ])
    w_map = data.VocabMap(data.get_vocab(train_tokens))
    t_map = data.VocabMap(data.get_vocab(train_tags))
    log.info(f'Character vocab={len(c_map)}')
    log.info(f'Word vocab={len(w_map)}')
    log.info(f'Tag vocab={len(t_map)}')
    log.info(f'Coarse tag vocab={len(c_t_map)}')
    t_freq = data.get_tok_freq(train_tokens)
    # We filter the morphlex embeddings based on the training and test set for quicker training. This should not be done in production
    filter_on = data.get_vocab(train_tokens)
    filter_on.update(data.get_vocab(test_tokens))
    # The morphlex embeddings are similar to the tokens, no EOS or SOS needed
    if args.debug:
        log.info('Running in debug mode, shortening training and testing data')
        train_tokens = train_tokens[:30]
        train_tags = train_tags[:30]
        train_tags_coarse = train_tags_coarse[:30]
        test_tokens = test_tokens[:30]
        test_tags = test_tags[:30]
        test_tags_coarse = test_tags_coarse[:30]
        args.use_morphlex = args.use_morphlex  # + '_short'
    m_map, embedding = data.read_embedding(
        args.use_morphlex, filter_on=filter_on)

    morphlex_embeddings = Embeddings(m_map, embedding)
    log.info("Creating coarse tagger")
    from ABLTagger import ABLTagger
    tagger_coarse = ABLTagger(vocab_chars=c_map,
                              vocab_words=w_map,
                              vocab_tags=c_t_map,  # coarse tags
                              word_freq=t_freq,
                              morphlex_embeddings=morphlex_embeddings,
                              coarse_features_embeddings=None,
                              hyperparams=args)
    log.info(
        f"Number of params {sum(np.prod(p.shape()) for p in tagger_coarse.model.parameters_list())}")
    log.info(
        f"Number of params {sum(np.prod(p.shape()) for p in tagger_coarse.model.lookup_parameters_list())}")
    log.info(
        f"Number of params {sum(np.prod(p.shape()) for l in tagger_coarse.fwdRNN.get_parameters()for p in l )}")
    log.info(
        f"Number of params {sum(np.prod(p.shape()) for l in tagger_coarse.bwdRNN.get_parameters()for p in l )}")
    log.info(
        f"Number of params {sum(np.prod(p.shape()) for l in tagger_coarse.cFwdRNN.get_parameters()for p in l )}")
    log.info(
        f"Number of params {sum(np.prod(p.shape()) for l in tagger_coarse.cBwdRNN.get_parameters()for p in l )}")
    log.info("Starting training and evaluating")

    x_y = list(zip(train_tokens, train_tags_coarse))
    x_y_test = list(zip(test_tokens, test_tags_coarse))
    tagger_coarse.train_and_evaluate_tagger(x_y=x_y,
                                            x_y_test=x_y_test,
                                            total_epochs=args.epochs_coarse_grained,
                                            out_dir=out_folder,
                                            evaluate=True)
    train_coarse_tagged_tags: data.List[data.SentTags] = [
        tagger_coarse.tag_sent(sent) for sent in train_tokens]
    test_coarse_tagged_tags: data.List[data.SentTags] = [
        tagger_coarse.tag_sent(sent) for sent in test_tokens]

    coarse_embeddings = Embeddings(c_t_map, coarse_embedding)

    tagger_fine = ABLTagger(vocab_chars=c_map,
                            vocab_words=w_map,
                            vocab_tags=t_map,  # fine tags
                            word_freq=t_freq,
                            morphlex_embeddings=morphlex_embeddings,
                            coarse_features_embeddings=coarse_embeddings,
                            hyperparams=args)
    x = list(zip(train_tokens, train_coarse_tagged_tags))
    x_y = list(zip(x, train_tags))  # type: ignore
    x_test = list(zip(test_tokens, test_coarse_tagged_tags))
    x_y_test = list(zip(x_test, test_tags))  # type: ignore
    tagger_fine.train_and_evaluate_tagger(x_y=x_y,
                                          x_y_test=x_y_test,
                                          total_epochs=args.epochs_fine_grained,
                                          out_dir=out_folder,
                                          evaluate=True)

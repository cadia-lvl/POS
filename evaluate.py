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
__license__ = "Apache 2.0"

import sys
import os
import argparse

import dynet_config
dynet_config.set(mem=32000, random_seed=42)

from ABLTagger import ABLTagger
from data import Embeddings, Vocab, Corpus
import data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # HYPERPARAMETERS
    parser.add_argument('--optimization', '-o', help="Which optimization algorithm",
                        choices=['SimpleSGD', 'MomentumSGD', 'CyclicalSGD', 'Adam', 'RMSProp'], default='SimpleSGD')
    parser.add_argument('--learning_rate', '-lr', help="Learning rate", type=float, default=0.13)
    parser.add_argument('--learning_rate_decay', '-lrd', help="Learning rate decay", type=float, default=0.05)
    parser.add_argument('--learning_rate_max', '-l_max', help="Learning rate max for Cyclical SGD", type=float, default=0.1)
    parser.add_argument('--learning_rate_min', '-l_min', help="Learning rate min for Cyclical SGD", type=float, default=0.01)
    parser.add_argument('--dropout', '-d', help="Dropout rate", type=float, default=0.0)
    # EXTERNAL DATA
    parser.add_argument('--data_folder', '-data', help="Folder containing training data", default='./data/format/')
    parser.add_argument('--out_folder', '-out-data', help="Folder containing results", default='./models')
    parser.add_argument('--use_morphlex', '-morphlex', help="File with morphological lexicon embeddings in ./extra folder. Example file: ./extra/dmii.or", default='./extra/dmii.vectors')
    parser.add_argument('--load_characters', '-load_chars', help="File to load characters from", default='./extra/characters_training.txt')
    parser.add_argument('--load_coarse_tagset', '-load_coarse', help="Load embeddings file for coarse grained tagset", default='./extra/word_class_vectors.txt')
    # TRAIN AND EVALUATE
    parser.add_argument('--corpus', '-c', help="Name of training corpus", default='otb')
    parser.add_argument('--dataset_fold', '-fold', help="select which dataset to use (1-10)", type=int, default=1)
    parser.add_argument('--epochs_coarse_grained', '-ecg', help="How many epochs for coarse grained training? (12 is default)", type=int, default=12)
    parser.add_argument('--epochs_fine_grained', '-efg', help="How many epochs for fine grained training? (20 is default)", type=int, default=20)
    parser.add_argument('--noise', '-n', help="Noise in embeddings", type=float, default=0.1)
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    try:
        args = parser.parse_args()
    except ValueError:
        sys.exit(0)
    args = parser.parse_args()

    char_vocab: Vocab = set()
    with open(args.load_characters) as f:
        for line in f:
            line = line.strip()
            for char in line:
                char_vocab.add(char)

    out_folder = f'{args.out_folder}/IFD-{args.dataset_fold}'
    os.mkdir(out_folder)

    morphlex_embeddings = Embeddings(args.use_morphlex)

    # First train or load a coarse tagger without evaluation - then tag and finally run the fine tagger
    training_corpus: Corpus = data.read_tsv(f'{args.data_folder}IFD-{args.dataset_fold:02}TM.tsv')
    test_corpus: Corpus = data.read_tsv(f'{args.data_folder}IFD-{args.dataset_fold:02}PM.tsv')

    train_tokens, train_tags = data.tsv_to_pairs(training_corpus)
    test_tokens, test_tags = data.tsv_to_pairs(test_corpus)
    coarse_train_tags = data.coarsify(train_tags)
    coarse_testing_tags = data.coarsify(test_tags)

    tokens = data.get_vocab(train_tokens)

    char_vocap_map = data.VocabMap(char_vocab)
    token_vocab_map = data.VocabMap(data.get_vocab(train_tokens))
    tag_vocab_map = data.VocabMap(data.get_vocab(train_tags))
    coarse_tag_vocab_map = data.VocabMap(data.get_vocab(coarse_train_tags))
    token_freqs = data.get_tok_freq(train_tokens)

    print("Creating coarse tagger")
    tagger_coarse = ABLTagger(vocab_chars=char_vocap_map,
                              vocab_words=token_vocab_map,
                              vocab_tags=coarse_tag_vocab_map,
                              word_freq=token_freqs,
                              morphlex_embeddings=morphlex_embeddings,
                              coarse_features_embeddings=None,
                              hyperparams=args)
    print("Starting training and evaluating")
    tagger_coarse.train_and_evaluate_tagger(training_data=(train_tokens, coarse_train_tags),
                                            test_data=(test_tokens, coarse_testing_tags),
                                            total_epochs=args.epochs_coarse_grained,
                                            out_dir=out_folder,
                                            evaluate=False)
    train_coarse_tagged_tags = [tagger_coarse.tag_sent(sent) for sent in train_tokens]
    test_coarse_tagged_tags = [tagger_coarse.tag_sent(sent) for sent in test_tokens]

    coarse_embeddings = Embeddings(args.load_coarse_tagset)

    tagger_fine = ABLTagger(vocab_chars=char_vocap_map,
                            vocab_words=token_vocab_map,
                            vocab_tags=tag_vocab_map,
                            word_freq=token_freqs,
                            morphlex_embeddings=morphlex_embeddings,
                            coarse_features_embeddings=coarse_embeddings,
                            hyperparams=args)
    tagger_fine.train_and_evaluate_tagger(training_data=((train_tokens, train_coarse_tagged_tags), train_tags),
                                          test_data=((test_tokens, test_coarse_tagged_tags), test_tags),
                                          total_epochs=args.epochs_fine_grained,
                                          out_dir=out_folder,
                                          evaluate=True)

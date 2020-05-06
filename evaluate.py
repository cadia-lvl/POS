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

import random
import sys
import os
from itertools import cycle
from datetime import datetime
from time import time
import argparse
import csv
from typing import List, Tuple

from ABLTagger import ABLTagger, Embeddings, Vocab, Utils

spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')


def update_progress_notice(i, epoch, start_time, epoch_start_time, avg_loss, total_epochs, evaluation = None, morphlex_flag = False):
    now_time = time()
    if morphlex_flag:
        print(" ",
            next(spinner),
            "{:>2}/{}".format(epoch, total_epochs),
            ("  {:>4}/{:<5}".format(int(now_time - start_time), str(int(now_time - epoch_start_time)) + 's') if i % 100 == 0 or evaluation else ""),
            ("  AVG LOSS: {:.3}".format(avg_loss) if i % 1000 == 0 or evaluation else ""),
            ("  EVAL: tags {:.3%} sent {:.3%} tr. only {:.3%} morphlex only {:.3%} both {:.3%} knw {:.3%} unk {:.3%}".format(*evaluation) if evaluation else ""),
            end='\r'
        )
    else:
        print(" ",
            next(spinner),
            "{:>2}/{}".format(epoch, total_epochs),
            ("  {:>4}/{:<5}".format(int(now_time - start_time), str(int(now_time - epoch_start_time)) + 's') if i % 100 == 0 or evaluation else ""),
            ("  AVG LOSS: {:.3}".format(avg_loss) if i % 1000 == 0 or evaluation else ""),
            ("  EVAL: tags {:.3%} sent {:.3%} knw {:.3%} unk {:.3%}".format(*evaluation) if evaluation else ""),
            end='\r'
        )


def evaluate_tagging(tagger, test_data, morphlex_flag=False, coarse_flag=False):
    eval_out = ''
    if morphlex_flag:
        good = total = good_sent = total_sent = unk_good = morphlex_good = morphlex_total = train_good = train_total = both_good = both_total = unk_total = 0.0
    else:
        good = total = good_sent = total_sent = unk_good = unk_total = 0.0
    for sent in test_data:
        if coarse_flag:
            words, golds, coarse_tags = map(list, zip(*sent))
            tags = [t for _, t in tagger.tag_sent(words, coarse_tags)]
        else:
            words, golds = map(list, zip(*sent))
            tags = [t for _, t in tagger.tag_sent(words)]
        if tags == golds: good_sent += 1
        total_sent += 1
        for go, gu, w in zip(golds, tags, words):
            eval_out += w + '\t' + go + '\t' + gu + '\n'
            total += 1
            if morphlex_flag:
                if go == gu:
                    good += 1
                    if w in tagger.morphlex.keys():
                        if tagger.word_frequency[w] == 0: morphlex_good += 1
                        else: both_good += 1
                    else:
                        if tagger.word_frequency[w] == 0: unk_good += 1
                        else: train_good += 1
                if w in tagger.morphlex.keys():
                    if tagger.word_frequency[w] == 0: morphlex_total += 1
                    else: both_total += 1
                else:
                    if tagger.word_frequency[w] == 0: unk_total += 1
                    else: train_total += 1
            else:
                if go == gu:
                    good += 1
                    if tagger.word_frequency[w] == 0:
                        unk_good += 1
                if tagger.word_frequency[w] == 0:
                    unk_total += 1

    filename = FILE_PREFIX + 'PM.txt__tagger_out'
    f = open(filename, "w")
    f.write(eval_out)
    f.close()

    if morphlex_flag:
        eval_text = str(total) + "|" + str(train_total) + "|" + str(morphlex_total) + "|" + str(both_total) + "|" + str(unk_total)
        return good/total, good_sent/total_sent, train_good/train_total, morphlex_good/morphlex_total, both_good/both_total, (good-unk_good)/(total-unk_total), unk_good/unk_total, eval_text
    return good/total, good_sent/total_sent, (good-unk_good)/(total-unk_total), unk_good/unk_total


def write_results_to_file(epoch, evaluation, loss, learning_rate, morphlex_flag, total_epochs, num_words):
    if not os.path.exists('./evaluate/'):
        os.makedirs('./evaluate')

    file_name = args.corpus + '_' + str(format(args.dataset_fold, '02')) + '_' + args.optimization + '_' + str(args.learning_rate)

    word_acc, sent_acc, known_acc, unknown_acc = evaluation

    with open('./evaluate/' + file_name, mode='a+') as results_file:
        results_writer = csv.writer(results_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([datetime.fromtimestamp(time()).strftime("%d. %B %Y %I:%M:%S"),
                                 str(args.dataset_fold),
                                 str(epoch),
                                 str(word_acc),
                                 str(sent_acc),
                                 str(known_acc),
                                 str(unknown_acc),
                                 str(loss),
                                 str(args.noise),
                                 str(args.dropout),
                                 (str(args.learning_rate) if args.optimization in ['MomentumSGD', 'SimpleSGD'] else ""),
                                 (str(args.learning_rate_max) if args.optimization == 'CyclicalSGD' else ""),
                                 (str(args.learning_rate_min) if args.optimization == 'CyclicalSGD' else ""),
                                 str(learning_rate),
                                 args.optimization,
                                 ("X" if epoch == total_epochs else ""),
                                 str(args.learning_rate_decay),
                                 str(morphlex_flag), str(num_words)])


def tag_testset(test_data, tagger, coarse_flag=False):
    filename = FILE_PREFIX + 'PM.txt__tagger_output'
    with open(filename, "w") as f:
        for sent in test_data:
            print(sent)
            if coarse_flag:
                words, _, coarse_tags = map(list, zip(*sent))
                f.write("\n".join([x[0] + "\t" + x[1] for x in tagger.tag_sent(words, coarse_tags)]) + '\n')
                f.write("\n")
            else:
                words, _ = map(list, zip(*sent))
                f.write("\n".join([x[0] + "\t" + x[1] for x in tagger.tag_sent(words)]) + '\n')
                f.write("\n")


def tag_coarse(input, tagger):
    input_file = open(input, 'rt')
    input_text = input_file.readlines()
    input_file.close()
    output_file = input + '__temp'

    with open(output_file, "w") as f:
        tokens = []
        for line in input_text:
            line = line.strip()
            # We read a blank line - sentence has been read.
            if not line:
                # WTF?
                #if tokens[0][0].isupper() and not tokens[0] in tagger.vw.w2i:
                #    tokens[0] = tokens[0][0] + tokens[0][1:]
                for word, tag in tagger.tag_sent(tokens):
                    f.write(f'{word}\t{tag}\n')
                f.write("\n")
                tokens = []
            else:
                token, _ = i.split()
                tokens.append(token)
    if len(tokens) != 0:
        print("Gotcha!")
        for word, tag in tagger.tag_sent(tokens):
            f.write(f'{word}\t{tag}\n')
        f.write("\n")


def create_tagged_test_file(test_file, coarse_tagged_file, output_file):
    test_open = open(test_file, 'r')
    test_lines = test_open.readlines()
    ctf_open = open(coarse_tagged_file, 'r')
    ctf_lines = ctf_open.readlines()

    file_length = len(test_lines)

    ctr = 0
    output = ''

    while ctr < file_length:
        try:
            tag = ctf_lines[ctr].split()[1].strip()
            output += test_lines[ctr].strip() + '\t' + tag + '\n'
        except:
            output += test_lines[ctr]
        ctr += 1

    fixed_file = open(output_file, 'w')
    fixed_file.write(output)
    fixed_file.close()


def train_and_evaluate_tagger(tagger, training_data, test_data, total_epochs, evaluate=True, morphlex=None, coarse=None):
    '''
    Train the tagger, report progress to console and write to file.
    '''

    if coarse is not None:
        coarse_flag = True
    else:
        coarse_flag = False

    if args.dropout:
        tagger.fwdRNN.set_dropouts(args.dropout, 0)
        tagger.bwdRNN.set_dropouts(args.dropout, 0)
        tagger.cFwdRNN.set_dropouts(args.dropout, 0)
        tagger.cBwdRNN.set_dropouts(args.dropout, 0)

    start_time = time()
    for ITER in range(total_epochs):
        cum_loss = num_tagged = 0
        epoch_start_time = time()
        random.shuffle(training_data)
        for i, sent in enumerate(training_data, 1):
            # Training
            loss_exp = tagger.sent_loss(sent)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(sent)
            loss_exp.backward()
            tagger.update_trainer()

            if i % 10 == 0:
                if morphlex is not None:
                    update_progress_notice(i, ITER + 1, start_time, epoch_start_time, cum_loss / num_tagged, total_epochs, None, True)
                else:
                    update_progress_notice(i, ITER + 1, start_time, epoch_start_time, cum_loss / num_tagged, total_epochs)

        # Evaluate
        if evaluate:
            if morphlex is not None:
                evaluation = evaluate_tagging(tagger, test_data, True, coarse_flag)
                update_progress_notice(i, ITER + 1, start_time, epoch_start_time, cum_loss / num_tagged, total_epochs, evaluation, True)
                # prófa að setja fjölda orða inn líka
                print(len(test_data))
                write_results_to_file(ITER + 1, evaluation, cum_loss / num_tagged, tagger.trainer.learning_rate, True, total_epochs, len(test_data))
            else:
                evaluation = evaluate_tagging(tagger, test_data, False, coarse_flag)
                update_progress_notice(i, ITER + 1, start_time, epoch_start_time, cum_loss / num_tagged, total_epochs, evaluation)
                # prófa að setja fjölda orða inn líka
                print(len(test_data))
                write_results_to_file(ITER + 1, evaluation, cum_loss / num_tagged, tagger.trainer.learning_rate, False, total_epochs, len(test_data))

        # decay
        if args.learning_rate_decay:
            tagger.trainer.learning_rate = tagger.trainer.learning_rate * (1-args.learning_rate_decay)

    # Show hyperparameters used when we are done
    print("\nHP opt={} epochs={} emb_noise={} ".format(args.optimization, total_epochs, args.noise))
    #if evaluate:
    #    tag_testset(test_data, tagger, coarse_flag)


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
    except:
        sys.exit(0)
    args = parser.parse_args()

    char_list = []
    for charline in open(args.load_characters):
        characters = charline.strip().split('\t')
        for c in characters:
            char_list.append(c)
    VocabCharacters = Vocab.from_corpus([char_list])

    morphlex_embeddings = Embeddings(args.use_morphlex)

    training_file_ending_coarse = "TM.coarse"
    test_file_ending_coarse = "PM.coarse"
    training_file_ending_fine = "TM.fine"
    test_file_ending_fine = "PM.fine"

    TaggedToken = Tuple[str, str]
    DbblTaggedToken = Tuple[str, str, str]
    TaggedSentence = List[TaggedToken]
    Corpus = List[TaggedSentence]

    # First train or load a coarse tagger without evaluation - then tag and finally run the fine tagger
    FILE_PREFIX = f"{args.data_folder}IFD-{args.dataset_fold:02}"

    train_file_coarse = FILE_PREFIX + training_file_ending_coarse
    test_file_coarse = FILE_PREFIX + test_file_ending_coarse
    train_coarse: Corpus = list(Utils.read(train_file_coarse))
    test_coarse: Corpus = list(Utils.read(test_file_coarse))

    train_file_fine = FILE_PREFIX + training_file_ending_fine
    test_file_fine = FILE_PREFIX + test_file_ending_fine
    train_fine: Corpus = list(Utils.read(train_file_fine))

    words, word_frequency, tags_coarse = Utils.create_vocabularies(train_file_coarse)
    words, word_frequency, tags_fine = Utils.create_vocabularies(train_file_fine)

    WORDS_FILE = f"{FILE_PREFIX}_WORDS{training_file_ending_coarse}"
    with open(WORDS_FILE, "w") as word_freq_file:
        for i in words:
            word_freq_file.write(i.strip() + '\t' + str(word_frequency[i]) + '\n')
    TAGS_COARSE_FILE = f"{FILE_PREFIX}_TAGS_COARSE{training_file_ending_coarse}"
    with open(TAGS_COARSE_FILE, "w") as tag_file_coarse:
        for i in tags_coarse:
            tag_file_coarse.write(i.strip() + '\t')
    TAGS_FINE_FILE = f"{FILE_PREFIX}_TAGS_FINE{training_file_ending_coarse}"
    with open(TAGS_FINE_FILE, "w") as tag_file_fine:
        for i in tags_fine:
            tag_file_fine.write(i.strip() + '\t')

    VocabWords, WordFrequency = Utils.build_word_dict(list(Utils.read(WORDS_FILE)))
    VocabTagsCoarse = Utils.build_vocab_tags(TAGS_COARSE_FILE)
    VocabTagsFine = Utils.build_vocab_tags(TAGS_FINE_FILE)
    tagger_coarse = ABLTagger(VocabCharacters, VocabWords, VocabTagsCoarse, WordFrequency, morphlex_embeddings, None, args)
    print("Training pre-tagger... this will take hours!")
    train_and_evaluate_tagger(tagger_coarse, train_coarse, test_coarse, args.epochs_coarse_grained, False, args.use_morphlex)
    tag_coarse(test_file_coarse, tagger_coarse)

    temp_test_file = test_file_coarse + '__for_fine_grained'
    create_tagged_test_file(test_file_fine, test_file_coarse + '__temp', temp_test_file)
    coarse_embeddings = Embeddings(args.load_coarse_tagset)
    test_fine = list(Utils.read(temp_test_file, 3))

    tagger_fine = ABLTagger(VocabCharacters, VocabWords, VocabTagsFine, WordFrequency, morphlex_embeddings, coarse_embeddings, args)
    train_and_evaluate_tagger(tagger_fine, train_fine, test_fine, args.epochs_fine_grained, True, args.use_morphlex, True)

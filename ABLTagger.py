#!/usr/bin/env python
"""
    ABLTagger: Augmented BiDirectional LSTM Tagger

    Main module

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
import datetime
import gc

import numpy as np
import dynet as dy
import csv

import pos.data as data

random.seed(42)


# Layer Dimensions for Combined emb. model
class CombinedDims:
    def __init__(self):
        self.hidden = 32  # map each timestep from bi_main_lstm to this
        self.hidden_input = 0  # This is computed during creation
        self.word_input = 0  # This is computed during creation
        # characters-as-word representation size after char_lstm, char_bilstm = 2 * char_lstm
        self.char_output = 64
        # input representation size after main_lstm, bi_main_lstm = 2 * main_lstm
        self.word_output = 64
        self.word_lookup = 128  # word embedding size
        self.char_lookup = 20  # character embedding size
        self.morphlex_lookup = 65  # morphological lexicon embedding size - read from file
        self.word_class_lookup = 14  # coarse tag embeddings size - read from file


class ABLTagger():
    START_OF_WORD = data.SOS
    END_OF_WORD = data.EOS

    def __init__(self,
                 vocab_chars: data.VocabMap,
                 vocab_words: data.VocabMap,
                 vocab_tags: data.VocabMap,
                 word_freq: data.Counter,
                 morphlex_embeddings: data.Embeddings = None,
                 coarse_features_embeddings: data.Embeddings = None,
                 hyperparams=None):
        self.hp = hyperparams
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(
            self.model, learning_rate=self.hp.learning_rate)
        self.word_frequency: data.Counter = word_freq

        self.vw: data.VocabMap = vocab_words
        self.vt: data.VocabMap = vocab_tags
        self.vc: data.VocabMap = vocab_chars

        self.morphlex_flag = False
        # If flag is set True then the model expects the coarse tag to be present in input (x)
        self.coarse_features_flag = False
        if coarse_features_embeddings is not None:
            self.coarse_features_flag = True
            self.coarse_features_embeddings = coarse_features_embeddings.embeddings
            self.coarse_features = coarse_features_embeddings.vocab
        if morphlex_embeddings is not None:
            self.morphlex_flag = True
            self.morphlex_embeddings = morphlex_embeddings.embeddings
            self.morphlex = morphlex_embeddings.vocab

        self.dim = CombinedDims()
        self.create_network()

    def create_network(self):
        assert len(
            self.vw), "Need to build the vocabulary (build_vocab) before creating the network."

        self.dim.word_input = self.dim.word_lookup + self.dim.char_output * 2
        if self.morphlex_flag:
            self.dim.word_input += self.dim.morphlex_lookup

        if self.coarse_features_flag:
            self.dim.word_input += self.dim.word_class_lookup
        self.dim.hidden_input = self.dim.word_output * 2  # * 2 because of bidirectional
        self.WORDS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.vw), self.dim.word_lookup))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.vc), self.dim.char_lookup))

        if self.morphlex_flag:
            self.MORPHLEX_LOOKUP = self.model.add_lookup_parameters(
                (len(self.morphlex_embeddings), self.dim.morphlex_lookup))
            self.MORPHLEX_LOOKUP.init_from_array(self.morphlex_embeddings)
        if self.coarse_features_flag:
            # TODO: Is this + 1 correct? We now read '0' from the file.
            self.WORD_CLASS_LOOKUP = self.model.add_lookup_parameters(
                (self.coarse_features_embeddings.shape[0], self.coarse_features_embeddings.shape[1]))
            self.WORD_CLASS_LOOKUP.init_from_array(
                self.coarse_features_embeddings)

        # MLP on top of biLSTM outputs, word/char out -> hidden -> num tags
        self.pH = self.model.add_parameters(
            (self.dim.hidden, self.dim.hidden_input))  # hidden-dim, hidden-input-dim
        self.pO = self.model.add_parameters(
            (len(self.vt), self.dim.hidden))  # vocab-size, hidden-dim

        # word-level LSTMs
        # layers, input-dim, output-dim
        self.fwdRNN = dy.LSTMBuilder(
            1, self.dim.word_input, self.dim.word_output, self.model)
        self.bwdRNN = dy.LSTMBuilder(
            1, self.dim.word_input, self.dim.word_output, self.model)

        # char-level LSTMs
        self.cFwdRNN = dy.LSTMBuilder(
            1, self.dim.char_lookup, self.dim.char_output, self.model)
        self.cBwdRNN = dy.LSTMBuilder(
            1, self.dim.char_lookup, self.dim.char_output, self.model)
        self.fwdRNN.set_dropout(self.hp.dropout)
        self.bwdRNN.set_dropout(self.hp.dropout)
        self.cFwdRNN.set_dropout(self.hp.dropout)
        self.cBwdRNN.set_dropout(self.hp.dropout)

    def char_rep(self, w, cf_init, cb_init):
        char_ids = [self.vc.w2i[self.START_OF_WORD]] + [self.vc.w2i[c]
                                                        if c in self.vc.w2i else -1 for c in w] + [self.vc.w2i[self.END_OF_WORD]]
        char_embs = [self.CHARS_LOOKUP[cid]
                     if cid != - 1 else dy.zeros(self.dim.char_lookup) for cid in char_ids]
        # unknown char is treated as -1 idx which is mapped to zeroes().
        # PAD is treated like any other char
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([fw_exps[-1], bw_exps[-1]])

    def word_rep(self, w):
        # unknown word is treated as zeroes().
        if self.word_frequency[w] == 0:
            return dy.zeros(self.dim.word_lookup)
        w_index = self.vw.w2i[w]
        return self.WORDS_LOOKUP[w_index]

    def morphlex_rep(self, w):
        if w not in self.morphlex.keys():
            return dy.zeros(self.dim.morphlex_lookup)
        else:
            return self.MORPHLEX_LOOKUP[self.morphlex[w]]

    def coarse_features_rep(self, t):
        if t not in self.coarse_features.keys():
            return dy.zeros(self.dim.word_class_lookup)
        else:
            return self.WORD_CLASS_LOOKUP[self.coarse_features[t]]

    def word_and_char_rep(self, w, cf_init, cb_init, t='0'):
        wembs = self.word_rep(w)
        cembs = self.char_rep(w, cf_init, cb_init)
        if self.coarse_features_flag:
            coarse_features = self.coarse_features_rep(t)
        if self.morphlex_flag:
            morphlex = self.morphlex_rep(w)

        if self.coarse_features_flag and self.morphlex_flag:
            return dy.concatenate([wembs, cembs, morphlex, coarse_features])
        elif self.coarse_features_flag:
            return dy.concatenate([wembs, cembs, coarse_features])
        elif self.morphlex_flag:
            return dy.concatenate([wembs, cembs, morphlex])
        else:
            return dy.concatenate([wembs, cembs])

    def build_tagging_graph(self, x: data.Union[data.SentTokens, data.Tuple[data.SentTokens, data.SentTags]]):
        """
        Graph is built per input. Each input should be a complete sentence
        x is the input. Will be treated as (token, coarse_tag) if model should use coarse tag feature.
        Otherwise (token) only.
        """
        # Clear the graph
        dy.renew_cg()
        # Shotgun debugging - running into memory issues. Let's force a gc
        gc.collect()  # 27,2% mem (using 33GB for DyNet, 126GB total)
        # Initialize the LSTMs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()

        cf_init = self.cFwdRNN.initial_state()
        cb_init = self.cBwdRNN.initial_state()
        # if self.coarse_features_flag:
        if type(x) == data.Tuple:
            try:
                wembs = [self.word_and_char_rep(
                    w, cf_init, cb_init, t) for w, t in zip(*x)]
            except ValueError:
                print("Value error in tagging_graph", x)
                raise
        else:
            wembs = [self.word_and_char_rep(w, cf_init, cb_init) for w in x]

        if self.hp.noise > 0:
            wembs = [dy.noise(we, self.hp.noise) for we in wembs]

        # Feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))

        # biLSTM states
        bi_exps = [dy.concatenate([f, b])
                   for f, b in zip(fw_exps, reversed(bw_exps))]

        # Feed each biLSTM state to an MLP
        return [self.pO * (dy.tanh(self.pH * bi_exp)) for bi_exp in bi_exps]

    def tag_sent(self, x) -> data.SentTags:
        """x is the input. Will be treated as (token, coarse_tag) if model should use coarse tag feature.
        Otherwise (token) only."""
        vecs = self.build_tagging_graph(x)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.vt.i2w[tag])
        return tuple(tags)

    def train_and_evaluate_tagger(self,
                                  x_y,
                                  x_y_test,
                                  total_epochs: int,
                                  out_dir: str,
                                  evaluate=True):
        '''
        Train the tagger, report progress to console and write to file.
        '''
        for epoch in range(1, total_epochs + 1):
            cum_loss = num_tagged = 0
            batch = 0

            random.shuffle(x_y)
            # x,y is a single training sentence: (tokens, tags) or ((tokens, c_tags), tags)
            for x, y in x_y:
                batch += 1
                vecs = self.build_tagging_graph(x)
                errs = []
                for v, t in zip(vecs, y):
                    tid = self.vt.w2i[t]
                    err = dy.pickneglogsoftmax(v, tid)
                    errs.append(err)
                loss = dy.esum(errs)

                cum_loss += loss.scalar_value()
                num_tagged += len(x)
                loss.backward()
                self.trainer.update()

                if batch % 10 == 0:
                    print(
                        f'{datetime.datetime.now()} epoch={epoch}/{total_epochs}, batch={batch}/{len(x_y)}, avg_loss={cum_loss / num_tagged}')

            # Evaluate
            if evaluate:
                evaluation = self.evaluate_tagging(
                    x_y_test, file_out=f'{out_dir}/{"fine" if self.coarse_features_flag else "coarse"}_eval_{epoch}_tags.txt')
                print(evaluation)
                write_results_to_file(file_out=f'{out_dir}/{"fine" if self.coarse_features_flag else "coarse"}_eval_results.tsv',
                                      epoch=epoch,
                                      evaluation=evaluation,
                                      loss=cum_loss / num_tagged,
                                      learning_rate=self.trainer.learning_rate,
                                      morphlex_flag=self.morphlex_flag,
                                      total_epochs=total_epochs,
                                      num_words=len(x_y_test))
            # decay
            if self.hp.learning_rate_decay:
                self.trainer.learning_rate = self.trainer.learning_rate * \
                    (1 - self.hp.learning_rate_decay)

        # Show hyperparameters used when we are done
        print("\nHP={}".format(self.hp))

    def evaluate_tagging(self,
                         test_data,
                         file_out: str):
        with open(file_out, "w") as f:
            good = total = good_sent = total_sent = unk_good = morphlex_good = morphlex_total = train_good = train_total = both_good = both_total = unk_total = 0.0
            for words, golds in test_data:
                tags = [t for t in self.tag_sent(words)]
                if tags == golds:
                    good_sent += 1
                total_sent += 1
                if self.coarse_features_flag:
                    words = words[0]
                for go, gu, w in zip(golds, tags, words):
                    f.write(f'{w}\t{go}\t{gu}\n')
                    total += 1
                    if self.morphlex_flag:
                        if go == gu:
                            good += 1
                            if w in self.morphlex.keys():
                                if self.word_frequency[w] == 0:
                                    morphlex_good += 1
                                else:
                                    both_good += 1
                            else:
                                if self.word_frequency[w] == 0:
                                    unk_good += 1
                                else:
                                    train_good += 1
                        if w in self.morphlex.keys():
                            if self.word_frequency[w] == 0:
                                morphlex_total += 1
                            else:
                                both_total += 1
                        else:
                            if self.word_frequency[w] == 0:
                                unk_total += 1
                            else:
                                train_total += 1
                    else:
                        if go == gu:
                            good += 1
                            if self.word_frequency[w] == 0:
                                unk_good += 1
                        if self.word_frequency[w] == 0:
                            unk_total += 1
        eval_text = str(total) + "|" + str(train_total) + "|" + \
            str(morphlex_total) + "|" + str(both_total) + "|" + str(unk_total)
        # 59169.0      3503.0      0.0          4482.0          54687.0     0.0
        print(total, total_sent, train_total,
              morphlex_total, both_total, unk_total)
        # A small hack to avoid devision by zero
        if total == 0:
            total = 1
        if total_sent == 0:
            total_sent = 1
        if train_total == 0:
            train_total = 1
        if morphlex_total == 0:
            morphlex_total = 1
        if both_total == 0:
            both_total = 1
        if unk_total == 0:
            unk_total = 1

        return good / total, good_sent / total_sent, train_good / train_total, morphlex_good / morphlex_total, both_good / both_total, (good - unk_good) / (total - unk_total), unk_good / unk_total, eval_text


def write_results_to_file(file_out,
                          epoch,
                          evaluation,
                          loss,
                          learning_rate,
                          morphlex_flag,
                          total_epochs,
                          num_words):
    word_acc, sent_acc, train_acc, morphlex_acc, both_acc, known_acc, unknown_acc, _ = evaluation

    with open(file_out, mode='a+') as results_file:
        results_writer = csv.writer(
            results_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([datetime.datetime.now(),
                                 str(epoch),
                                 str(word_acc),
                                 str(sent_acc),
                                 str(train_acc),
                                 str(morphlex_acc),
                                 str(both_acc),
                                 str(known_acc),
                                 str(unknown_acc),
                                 str(loss),
                                 str(learning_rate),
                                 str(morphlex_flag),
                                 str(num_words)])

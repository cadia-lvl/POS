#!/usr/bin/env python
import click
import torch

import data
from data import Corpus, Vocab, VocabMap
from model import ABLTagger

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

@click.group()
def cli():
    pass


@cli.command()
@click.argument('inputs', nargs=-1)
@click.argument('output', type=click.File('w+'))
@click.option('--coarse', is_flag=True, help='Maps the tags "coarse".')
def gather_tags(inputs, output, coarse):
    input_data = []
    for input in inputs:
        input_data.extend(data.read_tsv(input))
    _, tags = data.tsv_to_pairs(input_data)
    if coarse:
        tags = data.coarsify(tags)
    tags = data.get_vocab(tags)
    for tag in sorted(list(tags)):
        output.write(f'{tag}\n')


@cli.command()
@click.argument('training_files',
                nargs=-1
                )
@click.argument('test_file'
                )
@click.argument('output_dir'
                )
@click.option('--known_chars_file',
              default='./extra/characters_training.txt',
              help='A file which contains the characters the model should know. '
              + 'File should be a single line, the line is split() to retrieve characters.'
              )
@click.option('--morphlex_embeddings_file',
              default='./extra/dmii.vectors',
              help='A file which contains the morphological embeddings.'
              )
def train(training_files,
          test_file,
          output_dir,
          known_chars_file,
          morphlex_embeddings_file):
    """
    training_files: Files to use for training (supports multiple files = globbing).
    All training files should be .tsv, with two columns, the token and tag.
    test_file: Same format as training_files. Used to evaluate the model.
    output_dir: The directory to write out model and results.
    """
    # Read train and test data
    training_corpus = []
    print(training_files)
    for train_file in training_files:
        training_corpus.extend(data.read_tsv(train_file))
    test_corpus: Corpus = data.read_tsv(test_file)

    # Extract tokens, tags, characters
    train_tokens, train_tags = data.tsv_to_pairs(training_corpus)
    test_tokens, test_tags = data.tsv_to_pairs(test_corpus)

    # Prepare the coarse tags
    train_tags_coarse = data.coarsify(train_tags)
    testing_tags_coarse = data.coarsify(test_tags)

    # Define the vocabularies and mappings
    chars = data.read_known_characters(known_chars_file)

    train_token_vocab = data.get_vocab(train_tokens)
    test_token_vocab = data.get_vocab(test_tokens)
    train_tag_vocab = data.get_vocab(train_tags)
    test_tag_vocab = data.get_vocab(test_tags)

    char_vocap_map = data.VocabMap(chars, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID),
        (data.EOS, data.EOS_ID),
        (data.SOS, data.SOS_ID)
    ])
    token_vocab_map = data.VocabMap(train_token_vocab, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID),
    ])
    tag_vocab_map = data.VocabMap(train_tag_vocab)
    coarse_tag_vocab_map = data.VocabMap(data.get_vocab(train_tags_coarse))
    token_freqs = data.get_tok_freq(train_tokens)

    print('Token unk analysis')
    data.unk_analysis(train_token_vocab, test_token_vocab)
    print('Tag unk analysis')
    data.unk_analysis(train_tag_vocab, test_tag_vocab)
    # We filter the morphlex embeddings based on the training and test set. This should not be done in production
    filter_on = data.get_vocab(train_tokens)
    filter_on.update(data.get_vocab(test_tokens))
    m_vocab_map, embedding = data.read_embedding(morphlex_embeddings_file, filter_on=filter_on, special_tokens=[
        (data.UNK, data.UNK_ID),
        (data.PAD, data.PAD_ID)
    ])
    
    # Create training examples: List[w, m, List[c]]
    x: data.List[data.TrainSent] = data.create_examples(train_tokens[:10])
    # print("examples", x)
    x_idx: data.List[data.TrainSentidx] = []
    for sent in x:
        sent_idx = []
        for chars, w, m in sent:
            # TODO: add UNK
            sent_idx.append((
                [char_vocap_map.w2i[c] if c in char_vocap_map.w2i else data.UNK_ID for c in chars],
                token_vocab_map.w2i[w] if w in token_vocab_map.w2i else data.UNK_ID,
                m_vocab_map.w2i[m] if m in m_vocab_map.w2i else data.UNK_ID
            ))
        x_idx.append(sent_idx)
    # print("examples idx", x_idx)
    x_pad = data.pad_examples_(x_idx, pad_idx=data.PAD_ID)
    # print("padded examples idx", x_pad)
    x_tn = torch.tensor(x_pad, dtype=torch.long, device=device, requires_grad=False)
    # print(x_tn)
    print(x_tn.shape)
    tagger = ABLTagger(
        char_dim=len(char_vocap_map.w2i),
        token_dim=len(token_vocab_map.w2i),
        tags_dim=len(coarse_tag_vocab_map.w2i),
        morph_lex_embeddings=torch.from_numpy(embedding).float(),
        c_tags_embeddings=None
    )
    tagger(x_tn)
    # net = ABLTagger(token_dim=len(token_vocab_map),
    #                 char_dim=len(char_vocap_map),
    #                 morph_lex_embeddings=morphlex_embeddings,
    #                 c_tags_embeddings=)
    criterion = None
    optimizer = None
    epochs = 0
    for epoch in range(epochs):
        for i, (x, y) in enumerate(training_data):
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

if __name__ == '__main__':
    cli()

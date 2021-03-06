# POS tagger and lemmatizer for Icelandic
The goal of this project is to create a combined part-of-speech tagger and lemmatizer for Icelandic using the revised fine-grained tagging schema for Icelandic.
For further information about the schema see [MIM-Gold on CLARIN-IS](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/40) (the description pdf).

This work is based on the ABLTagger (in [References](#references)) but with considerable model modifications and runs on Python 3.8, PyTorch 1.7.0 and [transformers 4.1.1](https://github.com/huggingface/transformers).

# Table of Contents
- [Versions](#versions)
- [Installation](#installation)
  * [Command line usage](#command-line-usage)
  * [Python module](#python-module)
- [Docker](#docker)
- [License](#license)
- [Authors](#authors)
  * [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
  * [Installation](#installation-1)
  * [Running the tests](#running-the-tests)
  * [Continuous integration](#continuous-integration)
  * [Training data](#training-data)
  * [Additional training data (Morphological lexicon)](#additional-training-data--morphological-lexicon-)
    + [Filtering the morphological lexicon](#filtering-the-morphological-lexicon)
  * [Training models](#training-models)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Versions
- 2.0.1 Adding support for Python 3.6 and up. Model file from v2.0.0 is compatible.
- 2.0.0 Support for BERT-like language models and preliminary lemmatization. Adding the lemmatization required an overhaul of the exposed API. A model which supports lemmatization is not provided.
- 1.0.1 Bug fixes to Python module.
- 1.0.0 First release.

To see older versions we suggest looking through the git tags of the project (DyNet model + code, original DyNet model implemented in Pytorch).

# Installation
To use a pretrained model follow the instructions below.

```
# Using v2.0.1 - consider using the latest tag.
pip install git+https://github.com/cadia-lvl/POS.git@v2.0.1
# Download the model
wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/98/tagger-v2.0.0.pt
# Test the installation
pos tag path/to/tagger.pt example.txt example_tagged.txt
```
The tagger expects input to be tokenized and a tokenizer is not bundled with this package. We reccomend [tokenizer](https://github.com/mideind/Tokenizer) version 2.0+.

A docker container with the an in-built model is also provided, see [Docker](#docker).

Instructions for further development can be found in [Contributing](#Contributing).

## Command line usage
Note that the input and output should be paths (i.e. not stdin or stdout).

`example.txt` is correctly formatted input file: One token per line and sentences are separated with an empty line.
```Bash
cat example.txt 
Þar
sem
jökulinn
ber
við
loft
hættir
landið
að
vera
jarðneskt
,
en
jörðin
fær
hlutdeild
í
himninum
,
þar
búa
ekki
framar
neinar
sorgir
og
þess
vegna
er
gleðin
ekki
nauðsynleg
,
þar
ríkir
fegurðin
ein
, 
ofar
hverri
kröfu
.

Halldór
Laxness
```
Tagging this file
```Bash
pos tag path/to/tagger.pt example.txt example_tagged.txt
2021-01-27 13:01:20,442 - Setting device.
2021-01-27 13:01:20,443 - Using 1 CPU threads
2021-01-27 13:01:20,443 - Reading model file...
2021-01-27 13:01:20,785 - Reading dataset
2021-01-27 13:01:20,786 - No newline at end of file, handling it.
2021-01-27 13:01:20,786 - Splitting sentences in order to fit BERT-like model
2021-01-27 13:01:20,787 - Predicting tags
2021-01-27 13:01:20,934 - Processed 44 tokens in 0:00:00.145607 seconds
2021-01-27 13:01:20,934 - Done predicting!
2021-01-27 13:01:20,934 - Reversing the splitting of sentences in order to fit BERT-like model
2021-01-27 13:01:20,934 - Writing results
2021-01-27 13:01:20,935 - Done!
cat example_tagged.txt 
Þar     aa
sem     c
jökulinn        nkeog
ber     sfg3en
við     af
loft    nheo
hættir  sfg3en
landið  nheng
að      cn
vera    sng
jarðneskt       lhensf
,       pk
en      c
jörðin  nveng
fær     sfg3en
hlutdeild       nveo
í       af
himninum        nkeþg
,       pk
þar     aa
búa     sfg3fn
ekki    aa
framar  aam
neinar  fovfn
sorgir  nvfn
og      c
þess    fphee
vegna   af
er      sfg3en
gleðin  nveng
ekki    aa
nauðsynleg      lvensf
,       pk
þar     aa
ríkir   sfg3en
fegurðin        nveng
ein     lvensf
,       pk
ofar    afm
hverri  foveþ
kröfu   nveþ
.       pl

Halldór nken-s
Laxness nken-s
```
For additional flags and further details see `pos tag --help`
## Python module
Usage example of the tagger in another Python module [example.py](example.py).
```Python
"""An example of the POS tagger as a module."""
import pos

# Initialize the tagger
tagger = pos.Tagger(
    model_file="tagger.pt",
    device="cpu",
)

# Tag a single sentence
tags = tagger.tag_sent(["Þetta", "er", "setning", "."])
print(tags)
# ('fahen', 'sfg3en', 'nven', 'pl')
# Tuple[str, ...]

# Tag multiple sentences at the same time (faster).
tags = tagger.tag_bulk(
    [["Þetta", "er", "setning", "."], ["Og", "önnur", "!"]], batch_size=2
)
print(tags)
# (('fahen', 'sfg3en', 'nven', 'pl'), ('c', 'foven', 'pl'))
# Tuple[Tuple[str, ...], ...]

# Tag a correctly formatted file.
dataset = pos.FieldedDataset.from_file("example.txt")
tags = tagger.tag_bulk(dataset)
print(tags)
# (('aa', 'c', 'nkeog', 'sfg3en', 'af', 'nheo', 'sfg3en', 'nheng', 'cn', 'sng', 'lhensf', 'pk', 'c', 'nveng', 'sfg3en', 'nveo', 'af', 'nkeþg', 'pk', 'aa', 'sfg3fn', 'aa', 'aam', 'fovfn', 'nvfn', 'c', 'fphee', 'af', 'sfg3en', 'nveng', 'aa', 'lvensf', 'pk', 'aa', 'sfg3en', 'nveng', 'lvensf', 'pk', 'afm', 'foveþ', 'nveþ', 'pl'), ('nken-s', 'nken-s'))
# Tuple[Tuple[str, ...], ...]

```
For additional information, see the docstrings provided.
# Docker
Follow the [official installation guide for Docker](https://www.docker.com/).

Trained models are distributed with Docker and then the command line client is exposed by default, so see instructions above.

Before running the docker command be sure that the docker daemon has access to roughly 4GB of RAM.

```Bash
# Using version 2.0.0
docker run -v $PWD:/data haukurp/pos:2.0.0 /data/example.txt /data/example_tagged.txt
```

# License
[Apache v2.0](LICENSE)

# Authors
<a href="https://github.com/cadia-lvl/POS/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=cadia-lvl/POS" />
</a>
<!-- Made with [contributors-img](https://contributors-img.web.app). -->

- Haukur Páll Jónsson (current maintainer)
- Örvar Kárason
- Steinþór Steingrímsson

## Acknowledgments
- Reykjavík University

This project was funded (partly) by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by [Almannarómur](https://almannaromur.is/), is funded by the Icelandic Ministry of Education, Science and Culture.

# Contributing
For more involved installation instructions and how to train different models.

## Installation
We use [poetry](https://python-poetry.org/) to manage dependencies and to build wheels. Install poetry and do `poetry install`.
To activate the environment within the current shell call `poetry shell`.

## Running the tests
To run the tests simply run `pytest` within the `poetry` environment.
To run without starting the environment, run `poetry run pytest`.

This will run all the unit-tests and skip a few tests which rely on external data (model files).

To include these tests make sure to add additional options to the `pytest` command.
- `pytest --electra_model="electra_model/"` a directory containing all necessary files to load an electra model.
- `pytest --tagger="tagger.pt" --dictionaries="dictionaries.pickle"`, the necessary files to load a pretrained tagging model.

## Continuous integration
This project uses GitHub actions to run a number of checks (linting, testing) when a change is pushed to GitHub.
If a change does not pass the checks, a code fix is expected.
See `.github/workflows/python-package.yml` for the checks involved.

## Training data
The training data is a text file wich contains PoS-tagged sentences. The file has one token per line, as well as its corresponding tag. The sentences are separated by an empty line. 

```
Við     fp1fn
höfum   sfg1fn
góða    lveosf
aðstöðu nveo
fyrir   af
barnavagna      nkfo
og      c
kerrur  nvfo
.       pl

Börnin  nhfng
geta    sfg3fn
sofið   sþghen
úti     aa
ef      c
vill    sfg3en
.       pl
```

For Icelandic we used the [IDF](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/38) and [MIM-GOLD](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/40).
We use the 10th fold (in either dataset) for hyperparameter selection.

We provide some additional data which is used to train the model:
- `data/extra/characters_training.txt` contains all the characters which the model knows.
Unknown characters are mapped to `<unk>`

## Additional training data (Morphological lexicon)
We represent the information contained in the morphological lexicon with n-hot vectors.
To generate the n-hot vectors, different scripts will have to be written for different morphological lexicons.
We use the DMII morphological lexicon for Icelandic.
The script, `pos/vectorize_dim.py` is used to create n-hot vectors from DMII.
We first [download the data in SHsnid format](https://bin.arnastofnun.is/django/api/nidurhal/?file=SHsnid.csv.zip).
After unpacking the `SHsnid.csv` to `./data/extra`.
To generate the n-hot vectors we run the script:
```
python3 ./pos/vectorize_dim.py 
```
The script takes two parameters:
| Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --input 	       |	./data/extra/SHsnid.csv           |The file containing the DIM morphological lexicon in SHsnid format.
| -o  --output          | ./data/extra/dmii.vectors           |The file containing the DIM n-hot vectors.

### Filtering the morphological lexicon
Since the morphological lexicon contains more words than will be seen during training and testing, it is useful to filter out unseen words.

```
pos filter-embedding data/raw/mim/* data/raw/otb/* data/extra/dmii.vectors data/extra/dmii.vectors_filtered bin
```
For an explanation of the parameters run `pos filter-embedding --help`

## Training models
A model can be trained by invoking the following command.
```
pos train-and-tag \
  training_data/*.tsv \
  testing_data.tsv \
  out # A directory to write out training results
```
For a description of all the arguments and options, run `pos train-and-tag --help`.

Parameters with default values (options) are prefixed with `--`.

It is also useful to look at the BASH scripts in `bin/`

# References
[Augmenting a BiLSTM Tagger with a Morphological Lexicon and a Lexical Category Identification Step](https://www.aclweb.org/anthology/R19-1133/)
```
@inproceedings{steingrimsson-etal-2019-augmenting,
    title = "Augmenting a {B}i{LSTM} Tagger with a Morphological Lexicon and a Lexical Category Identification Step",
    author = {Steingr{\'\i}msson, Stein{\th}{\'o}r  and
      K{\'a}rason, {\"O}rvar  and
      Loftsson, Hrafn},
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019)",
    month = sep,
    year = "2019",
    address = "Varna, Bulgaria",
    url = "https://www.aclweb.org/anthology/R19-1133",
    doi = "10.26615/978-954-452-056-4_133",
    pages = "1161--1168",
}
```

# POS tagger for Icelandic
The goal of this project is to create a part-of-speech tagger for Icelandic using the revised fine-grained tagging schema for Icelandic.
For further information about the schema see [MIM-GOLD on CLARIN-IS](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/40) (the description pdf).

This work is based on the ABLTagger (in [References](#references)) but with some model modifications and runs on PyTorch 1.6.0.

# Table of Contents
- [POS tagger for Icelandic](#pos-tagger-for-icelandic)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Command line usage](#command-line-usage)
- [Python module](#python-module)
- [Docker](#docker)
- [License](#license)
- [Authors](#authors)
  * [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
  * [Installation](#installation-1)
  * [Training data](#training-data)
  * [Additional training data (Morphological lexicon)](#additional-training-data--morphological-lexicon-)
    + [Filtering the morphological lexicon](#filtering-the-morphological-lexicon)
  * [Training models](#training-models)
- [Versions](#versions)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Installation
The tagger expects input to be tokenized and a tokenizer is not bundled with this package. We reccomend [tokenizer](https://github.com/mideind/Tokenizer) version 2.0+.

Installing the PoS tagger locally without using Docker:
```
# Using version 1.0.0
pip install git+https://github.com/cadia-lvl/POS.git@1.0.0
# Download the model & additional files
wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/53/tagger.pt
wget https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/53/dictionaries.pickle
# Test the installation
pos path/to/tagger.pt path/to/dictonaries.pickle example.txt example_tagged.txt
```

For usage examples see next sections.

For Docker installation see [Docker](#docker).

For installation for further development see [Contributing](#Contributing).

## Command line usage
Note that current version of the tagger expects the input and output to be paths (i.e. not stdin or stdout).

example.txt is correctly formatted input file: One token per line and sentences are separated with an empty line.
```Bash
cat example.txt
Þetta
er
próf
.

Tvær
setningar
!
```
Tagging this file
```Bash
pos path/to/tagger.pt path/to/dictonaries.pickle example.txt example_tagged.txt
2020-09-18 12:43:50,345 - Setting device.
2020-09-18 12:43:50,346 - Reading dictionaries
2020-09-18 12:43:52,955 - Reading model file
2020-09-18 12:43:54,188 - Reading dataset
2020-09-18 12:43:54,188 - No newline at end of file, handling it.
2020-09-18 12:43:54,188 - Predicting tags
2020-09-18 12:43:54,212 - Tagged 8 tokens
2020-09-18 12:43:54,212 - Tagging took=0:00:00.023401 seconds
2020-09-18 12:43:54,212 - Done predicting!
2020-09-18 12:43:54,212 - Writing results
2020-09-18 12:43:54,213 - Done!
cat example_tagged.txt 
Þetta   fahen
er      sfg3en
próf    nhen
.       pl

Tvær    tfvfn
setningar       nvfn
!       pl

```
For additional flags and further details see `pos tag --help`
## Python module
Usage example of the tagger in another Python module [example.py](example.py).
```Python
"""An example of the POS tagger as a module."""
import pos

# Initialize the tagger
tagger = pos.Tagger(
    model_file="path/to/tagger.pt",
    dictionaries_file="path/to/dictionaries.pickle",
    device="cpu",
)

# Tag a single tokenized sentence
tags = tagger.tag_sent(["Þetta", "er", "setning", "."])
print(tags)
# ('fahen', 'sfg3en', 'nven', 'pl')

# Tag a correctly formatted file.
dataset = pos.SimpleDataset.from_file("example.txt")
tags = tagger.tag_bulk(dataset=dataset)
print(tags)
# (('fahen', 'sfg3en', 'nhen', 'pl'), ('tfvfn', 'nvfn', 'pl', 'aa'))
```
For additional information, see the docstrings provided.
# Docker
Follow the [official installation guide for Docker](https://www.docker.com/).

Trained models are distributed with Docker and then the command line client is exposed by default, so see instructions above.

Before running the docker command be sure that the docker daemon has access to roughly 4GB of RAM.

```Bash
# Using version 1.0.0
docker run -v $PWD:/data haukurp/pos:1.0.0 /data/example.txt /data/example_tagged.txt
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
We use poetry to manage dependencies and to build wheels. Install poetry and do `poetry install`.

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

For Icelandic we used the [IDF](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/38) and [MIM-GOLD](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/40). We use the 10th fold in MIM-GOLD to evaluate the trained models.

We provide some additional data which is used to train the model:
- `data/extra/characters_training.txt` contains all the characters which the model knows. Unknown characters are mapped to `<unk>`

## Additional training data (Morphological lexicon)
We represent the information contained in the morphological lexicon with n-hot vectors. To generate the n-hot vectors, different scripts will have to be written for different morphological lexicons.
We use the DIM morphological lexicon for Icelandic.
The script, `pos/vectorize_dim.py` is used to create n-hot vectors from DIM.
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
Since the morphological lexicon contains more words than will be seen in the training and testing sets it is useful *when developing/experimenting* with models to filter
out words which will not be seen during training and testing.

To do this we filter out the morphological lexicon based on the training and test sets and use the filtered morphological lexicon when developing models.

```
./main.py filter-embedding data/raw/mim/* data/raw/otb/* data/extra/dmii.vectors data/extra/dmii.vectors_filtered bin
```
For explaination of the parameters run `./main.py filter-embedding --help`

## Training models
A model can be trained by invoking the following command.
```
main.py train-and-tag \
  training_data/*.tsv \
  testing_data.tsv \
  out # A directory to write out training results
```
For a description of all the arguments and options, run `main.py train-and-tag --help`.

Parameters with default values (options) are prefixed with `--`.

It is also useful to look at the BASH scripts provided in the main directory.

# Versions
- 1.0.0 First release as a docker container.

To see older versions we suggest looking through the git tags of the project.

# References
ABLTagger is a bidirectonal LSTM Part-of-Speech Tagger with combined Word and Character embeddings, augmented with a morphological lexicon and a lexical category identification step. The work is described in the paper [Augmenting a BiLSTM Tagger with a Morphological Lexicon and a Lexical Category Identification Step](https://www.aclweb.org/anthology/R19-1133/)

The paper describes a method for achieving high accuracy in part-of-speech tagging a fine grained tagset. We show how the method is used to reach the highest accuracy reported for PoS-tagging Icelandic. The tagger is augmented by using a morphological lexicon, [The Database of Icelandic Morphology (DIM)](https://www.aclweb.org/anthology/W19-6116/), and by running a pre-tagging step using a very coarse grained tagset induced from the fine grained data.

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
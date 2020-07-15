# POS tagger for Icelandic
The goal of this project is to create a part-of-speech tagger for Icelandic using the revised fine-grained tagging schema for Icelandic. For further information about the schema see [CLARIN-IS](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/40).

This work is based on the ABLTagger but with some model modifications and is compatible with PyTorch 1.5.1.

# Table of Contents
- [POS tagger for Icelandic](#pos-tagger-for-icelandic)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Running (Tagging text)](#running--tagging-text-)
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
Pretrained models are distributed with Docker, so follow the [official installation guide for Docker](https://www.docker.com/).

For installation for further development see [Contributing](#Contributing).

# Running (Tagging text)
The docker image comes with a pretrained model. To tag tokens, execute the following command:
```
cat tokenized_untagged.tsv | docker run -i haukurp/pos - - > tagged.tsv
```
The file `tokenized_untagged.tsv` is tokenized and correctly formatter input text which should be tagged.
The tagger expects the input to be one token per line and sentences should be separated by with an empty line.
A tokenizer is not bundled with the tagger but we recommend [tokenizer](https://github.com/mideind/Tokenizer) version 2.0+.

This input is piped to the docker command and read in by the tagger from stdin (`-`).
The output is also piped out to stdout (`-`) and is then redirected to the file `tagged.tsv`.
The output has the same format as the input file except that after each token there is a tab and then the corresponding tag.
The output thus contains the original text as well as the tags.

Input example (`tokenized_untagged.tsv`):
```
Við     
höfum   
góða    
aðstöðu 
fyrir   
barnavagna
og      
kerrur  
.       

Börnin  
geta    
sofið   
úti     
ef      
vill    
.       
```

Output example (`tagged.tsv`):
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

This project was funded (partly) by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by [Almannarómur](https://almannaromur.is/), is funded by the Icelandic Ministry of Education, Science and Culture."

# Contributing
For more involved installation instructions and how to train different models.

## Installation
We assume that development is made with `conda`. Start by creating a new conda environment and setup all necessary development tools.

```
conda create --name pos python==3.7.7
conda activate pos
conda install pytorch -c pytorch
conda install click pytest black mypy pydocstyle flake8 tqdm
```
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
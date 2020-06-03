# ABLTagger
ABLTagger is a bidirectonal LSTM Part-of-Speech Tagger with combined Word and Character embeddings, augmented with a morphological lexicon and a lexical category identification step. The work is described in the paper [Augmenting a BiLSTM Tagger with a Morphological Lexicon and a Lexical Category Identification Step](https://www.aclweb.org/anthology/R19-1133/)
NOTE: This code has been updated to work with a revised fine-grained tagging schema for Icelandic.
If you find this work useful in your research, please cite the paper: 

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

The paper describes a method for achieving high accuracy in part-of-speech tagging a fine grained tagset. We show how the method is used to reach the highest accuracy reported for PoS-tagging Icelandic. The tagger is augmented by using a morphological lexicon, [The Database of Icelandic Morphology (DIM)](https://www.aclweb.org/anthology/W19-6116/), and by running a pre-tagging step using a very coarse grained tagset induced from the fine grained data.   

The code has been adjusted to run with PyTorch.
# Training
We assume that development is made with `conda`. Setup the environment as described in `environment.yml` before training.
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

In the paper we use the training sets from [The Icelandic Frequency Dictionary](http://www.malfong.is/index.php?lang=en&pg=ordtidnibok) and the [MIM-GOLD](http://www.malfong.is/index.php?lang=en&pg=gull).

In `data/extra` we provide some additional data used for training.
- `characters_training.txt` contains all the characters which the model knows. Unknown characters are mapped to `<unk>`

TODO: Remove other files. Need to remove dependency in code.

## Additional information (Morphological lexicon)
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

# Analysis
We provide some tools to analyse the data and model.
## Tag analysis
TODO: Gather tags.
## Error analysis
TODO (`evaluate.py`)

# Tagging
You can use trained model to tag a sentence. You can either use a model trained by you or download a pretrained model.

## Downloading a pretrained model
TODO

## Tagging texts
TODO


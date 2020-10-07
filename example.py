"""An example of the POS tagger as a module."""
import pos

# Initialize the tagger
tagger = pos.Tagger(
    model_file="tagger.pt", dictionaries_file="dictionaries.pickle", device="cpu",
)

# Tag a single sentence
tags = tagger.tag_sent(["Þetta", "er", "setning", "."])
print(tags)
# ('fahen', 'sfg3en', 'nven', 'pl')

# Tag a correctly formatted file.
dataset = pos.SimpleDataset.from_file("example.txt")
tags = tagger.tag_bulk(dataset=dataset)
print(tags)
# (('aa', 'c', 'nkeog', 'sfg3en', 'af', 'nheo', 'sfg3en', 'nheng', 'cn', 'sng', 'lhensf', 'pk', 'c', 'nveng', 'sfg3en', 'nveo', 'af', 'nkeþg', 'pk', 'aa', 'sfg3fn', 'aa', 'aam', 'fovfo', 'nvfo', 'c', 'fphee', 'af', 'sfg3en', 'nveng', 'aa', 'lvensf', 'pk', 'aa', 'sfg3en', 'nveng', 'lvensf', 'pk', 'afm', 'foveþ', 'nveþ', 'pl'), ('nken-s', 'nken-s',))

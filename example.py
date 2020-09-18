"""An example of the POS tagger as a module."""
import pos

# Initialize the tagger
tagger = pos.Tagger(
    model_file="out/full-v2/tagger.pt",
    dictionaries_file="out/full-v2/dictionaries.pickle",
    device="cpu",
)

# Tag a single sentence
tags = tagger.tag_sent(["Þetta", "er", "setning", "."])
print(tags)
# ('fahen', 'sfg3en', 'nven', 'pl')

# Tag a correctly formatted file.
dataset = pos.SimpleDataset.from_file("example.txt")
tags = tagger.tag_bulk(dataset=dataset)
print(tags)
# (('fahen', 'sfg3en', 'nhen', 'pl'), ('tfvfn', 'nvfn', 'pl', 'aa'))

"""An example of the POS tagger as a module."""
import torch

import pos

# Initialize the tagger
device = torch.device("cpu")  # CPU
tagger: pos.Tagger = torch.hub.load(
    repo_or_dir="cadia-lvl/POS:dev",
    model="tag",
    device=device,
    force_reload=False,
    force_download=False,
)

# Tag a single sentence
tags = tagger.tag_sent(("Þetta", "er", "setning", "."))
print(tags)
# ('fahen', 'sfg3en', 'nven', 'pl')
# Tuple[str, ...]

# Tag multiple sentences at the same time (faster).
tags = tagger.tag_bulk(
    (("Þetta", "er", "setning", "."), ("Og", "önnur", "!")), batch_size=2
)  # Batch size works best with GPUs
print(tags)
# (('fahen', 'sfg3en', 'nven', 'pl'), ('c', 'foven', 'pl'))
# Tuple[Tuple[str, ...], ...]

# Tag a correctly formatted file.
dataset = pos.FieldedDataset.from_file("example.txt")
tags = tagger.tag_bulk(dataset)
print(tags)
# (('aa', 'ct', 'nkeog', 'sfg3en', 'af', 'nheo', 'sfg3en', 'nheng', 'cn', 'sng', 'lhensf', 'pk', 'c', 'nveng', 'sfg3en', 'nveo', 'af', 'nkeþg', 'pk', 'aa', 'sfg3fn', 'aa', 'aam', 'fovfn', 'nvfn', 'c', 'fphee', 'af', 'sfg3en', 'nveng', 'aa', 'lvensf', 'pk', 'aa', 'sfg3en', 'nveng', 'lvensf', 'pk', 'afm', 'foveþ', 'nveþ', 'pl'), ('nken-s', 'nken-s'))
# Tuple[Tuple[str, ...], ...]

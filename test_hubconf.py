import logging

import torch

from pos import Tagger

logging.basicConfig(level=logging.DEBUG)

model: Tagger = torch.hub.load(repo_or_dir="cadia-lvl/POS:hubconf", model="pos-small")
tags = model.tag_sent(("Þetta", "er", "prófun."))
print(tags)

import logging
from logging import getLogger

import torch

from pos import Tagger

logging.basicConfig(level=logging.DEBUG)

# pip install git... POS
# repo = 'pytorch/vision'
# >>> model = torch.hub.load(repo, 'resnet50', pretrained=True)
# repo_owner/repo_name[:tag_name]
model: Tagger = torch.hub.load(repo_or_dir="cadia-lvl/POS[:hubconf]", model="pos", source="local")
tags = model.tag_sent(("Þetta", "er", "prófun."))
print(tags)

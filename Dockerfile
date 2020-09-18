FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Setup
RUN pip install git+https://github.com/cadia-lvl/POS.git@1.0.0

# Trained model - it probably needs to be manually downloaded from the training location.
COPY dictionaries.pickle /dictionaries.pickle
COPY tagger.pt /model.pt

ENTRYPOINT [ "pos", "tag", "/model.pt", "/dictionaries.pickle" ]
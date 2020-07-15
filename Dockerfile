FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

# Setup
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Model code
COPY main.py .
COPY pos ./pos

# Trained model - it probably needs to be manually downloaded from the training location.
COPY full/dictionaries.pickle ./model/dictionaries.pickle
COPY full/tagger.pt ./model/model.pt

ENTRYPOINT [ "python", "main.py", "tag", "./model/model.pt", "./model/dictionaries.pickle" ]
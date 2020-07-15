FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

WORKDIR /app
ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

ADD main.py .
ADD ./pos .

CMD [ "python", "main.py" ]
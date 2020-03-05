FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

USER root
COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt
RUN python3 setup.py develop

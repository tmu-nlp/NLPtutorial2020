FROM python:3.8.3-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libsndfile-dev apt-utils && \
    apt clean autoclean && \
    apt autoremove -y

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

RUN python3 -m pip install numpy matplotlib seaborn

WORKDIR /work
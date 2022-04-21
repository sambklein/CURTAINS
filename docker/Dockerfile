ARG TORCH_VERSION=latest

FROM pytorch/pytorch:${TORCH_VERSION}#this will get replaced in the ci though

USER root
RUN apt-get -qq -y update && \
    apt-get -qq -y upgrade && \
    apt-get -qq -y install \
        wget \
        curl \
        git \
        make \
        texlive \
        sudo

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

EXPOSE 8888

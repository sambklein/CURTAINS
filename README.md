# CURTAINS

This repository contains the code used to run the experiments in the paper [Curtains for your sliding window](https://arxiv.org/abs/2203.09470).

## Dependencies
The recipe for building the container that was used to run these experiments can be found in ```docker```.

[//]: # (TODO add container)

## Usage

The main driving script for training Curtains models is ```CURTAINS.py```, multiple runs over different classifiers is handled by ```run_classifiers.py```. Cathode models are trained with ```CATHODE.py```.

## Data
To download the data run the following commands.

```angular2html
mkdir data/downloads
cd data/downloads
wget https://zenodo.org/record/4536377/files/events_anomalydetection_v2.features.h5
```


# implicitBIBae

The current description of the project can be found [here](https://www.overleaf.com/read/pxgsjkfmgqxy)

- [ ] Set up IDE.
- [x] Decide on project structure (classes, organize different architecture, loss functions, etc) (Check code below to explore different architectures and [this](https://github.com/bayesiains/nsf) for project structure).
- [x] Add additional hyper parameters like activation functions on outputs of outer_encoder and inner decoder including) tanh, relu, sigmoid,... batch norm
- [x] Install packages we need (pytorch, geomloss, nflows).
- [x] Translate current code from tensorflow to pytorch.
- [x] Implement INNs.
- [x] Define proper train/validation/test scheme.
- [x] Set up run_tests to evaluate model performance post training
- [x] Better training tracking functionality and saving checkpoints etc
- [x] Implicit models
- [x] sythetic 2D datasets
- [ ] Mflows datasets
- [x] make a /logs dir for dumping training data 
- [x] add functions to output model capacities
- [ ] build tests for different functionalities - does regularization work, does model saving work, etc)
- [ ] change ae_inn_plane to have option to use coupling layers instead of autoregressive so sampling is faster
- [x] ae_inn_plane currently doesn't work unless you use a tanh activation, fix iot
- [x] [spineflows HEP dataset](https://archive.ics.uci.edu/ml/datasets/HEPMASS)
- [x] INN bounds and scaling of the data
- [x] Supervised learning test
- [x] AAE
- [ ] Fix binning for histograms in the supervised (and all other) trainings
- [ ] Supervised learning 2d gaussian to 2d checkerboard hyperparameter scan including dropout, loss type, depth, width, lr, wd, .... everything. Number of OOD should be the benchmark quality metric.
- [ ] Make a utility to plot the latent space in N dimensions as a series of 2D marginals, use contour instead of scatter
- [ ] Make a utility/class to time experiments and print an estimate of a single epoch so that you can tell if a job has enough time to run to completion

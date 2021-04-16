# CURTAINS

## TO DO:
- [x] Set up tests to validate method, are samples from quantile 1 'further away' from samples drawn from quantile 1 than samples drawn from quantile 2 - if we do not have a measure where a given quantile is more similar to itself than the next quantile, then the problem is ill posed.
- [x] Basic training to learn map from quantile 1 to quantile 2, taking continuous masses and distribution matching across batches. See what the model converges to.
- [x] Train to map from quantile 1 to quantile 3 and validate on quantile 2
- [ ] Add titles to current plots, what row corresponds to which map
- [ ] Show how distributions change from inputs to outputs
- [ ] HPO scan to find good set up for INN

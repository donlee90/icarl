# A PyTorch Implementation of iCaRL
A PyTorch Implementation of [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725).

The code implements experiments on CIFAR-10 and CIFAR-100

### Notes
* This code does **not** reproduce the results from the paper. This may be due to following reasons:
  1. Different hyperparameters than the original ones in the paper (e.g. learning rate schedules).
  2. Different loss function; I replaced BCELoss to CrosEntropyLoss since it seemd to produce better results.
  3. I tried to replicate the algorithm exactly as described in the paper, but I might have missed some of the details.
* Versions
  - Python 2.7
  - PyTorch v0.1.12

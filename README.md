[![Latest Release](https://gitlab.com/raccs-lab/auramask-library/SoftAdapt-Keras/-/badges/release.svg)](https://gitlab.com/raccs-lab/auramask-library/SoftAdapt-Keras/-/releases) [![coverage report](https://gitlab.com/raccs-lab/auramask-library/SoftAdapt-Keras/badges/main/coverage.svg)](https://gitlab.com/raccs-lab/auramask-library/SoftAdapt-Keras/-/commits/main) [![arXiv:10.48550/arXiv.1912.12355](http://img.shields.io/badge/arXiv-110.48550/arXiv.2206.04047-A42C25.svg)](https://doi.org/10.48550/arXiv.1912.12355)

# SoftAdapt

This repository contains an implementation of the [SoftAdapt algorithm](https://arxiv.org/pdf/1912.12355.pdf)(techniques for adaptive loss balancing of multi-tasking neural networks).

Since 2020 (when SoftAdapt was first published), SoftAdapt has been applied to a variety of applications, ranging from generative models (e.g. these papers for [VAEs](https://arxiv.org/abs/2009.11693) and [GANs](https://www.sciencedirect.com/science/article/pii/S0167739X2100488X)), [model compression](https://arxiv.org/abs/2012.01604), and [Physics Informed Neural Networks](https://arxiv.org/pdf/2211.16753.pdf), to name a few.[^1]

[^1]: **SoftAdapt is also available within [NVIDIA's Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/theory/advanced_schemes.html#:~:text=annular%20ring%20example.-,SoftAdapt,-Softadapt%20is%20a). [Modulus](https://developer.nvidia.com/modulus) is a an open-source framework for building, training, and fine-tuning Physics-based DL models in Python.**

This implementation utilizes the multi-backend [Keras 3](https://keras.io) machine learning library and provides a Keras callback for use during Keras training.

## Installing SoftAdapt
### From PyPi:
```bash
pip install softadapt
```
### From the Repository
``` bash
git clone https://github.com/raccs-lab/auramask-library/SoftAdapt-Keras.git
cd SoftAdapt-Keras
pip install .
```

### Running Tests
```bash
uv run --with pytest pytest
```
For simplicity, we test primarily with PyTorch, so be sure that `KERAS_BACKEND` environment is set to `torch` before running the script or with the following:
```bash
KERAS_BACKEND=torch uv run --with pytest pytest
```

## General Usage and Examples
SoftAdapt consists of three variants. These variants are the "original" `SoftAdapt`, `NormalizedSoftAdapt`, and `LossWeightedSoftAdapt`. Below, we discuss the logic of SoftAdapt and provide some simple examples for calculating SoftAdapt weights.

### Example 1
SoftAdapt is designed for multi-tasking neural networks, where the loss component that is being optimized consists of `n` parts (`n` > 1). For example, consider a loss function that consists of three components:

```python
criterion = loss_A + loss_B + loss_C
```

Traditionally, these loss components are weighted the same (i.e. all having coefficients of 1); however, as shown by many studies, using different balancing coefficients for each component based on the optimization performance can significantly improve model training. SoftAdapt aims to calculate the most optimal set of (convex) weights based on live statistics.

Considering the example above, let us assume that the first 5 epochs have resulted in the following loss values:
```python
loss_A = ops.array([1, 2, 3, 4, 5])
loss_B = ops.array([150, 100, 50, 10, 0.1])
loss_C = ops.array([1500, 1000, 500, 100, 1])
```
Clearly, the first loss component is not being optimized as well as the other two parts, since it is increasing while the rates of change for component B and C being negative (with component C decreasing 10x faster than component B). Now let us see the different variants of SoftAdapt in action for this problem.

```python
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from keras import ops
# We redefine the loss components above for the sake of completeness.
loss_A = ops.array([1, 2, 3, 4, 5])
loss_B = ops.array([150, 100, 50, 10, 0.1])
loss_C = ops.array([1500, 1000, 500, 100, 1])

# Here we define each SoftAdapt algorithm as a new object
softadapt_object = SoftAdapt(beta=0.1)
normalized_softadapt_object = NormalizedSoftAdapt(beta=0.1)
loss_weighted_softadapt_object = LossWeightedSoftAdapt(beta=0.1)
```
(1) The original variant calculations are: 
```python
softadapt_object.get_component_weights(loss_A, loss_B, loss_C)
# >>> tensor([9.9343e-01, 6.5666e-03, 3.8908e-22])
```
(2) Normalized slopes variant outputs:
```python
normalized_softadapt_object.get_component_weights(loss_A, loss_B, loss_C)
# >>> tensor([0.3221, 0.3251, 0.3528])
```
and (3) the loss-weighted variant results in:
 ```python
loss_weighted_softadapt_object.get_component_weights(loss_A, loss_B, loss_C)
#>>> tensor([8.7978e-01, 1.2022e-01, 7.1234e-20])
```
as we see above, the first variant only focuses on the rates of change of each loss function, and since the values in component 3 are decreasing much faster than the other two components, the algorithm assigns it a weight very close to 0. Similarly, the second component also gets a very small weight while the first component has a weight close to 1. This means that the optimzer should primarily focus on the first component, and in a sense, not worry about components B and C. On the other hand, the second variant normalizes the slopes, which significantly reduces the differences in the rate of change, resulting in a much more moderate distribution of weights across the three components. Lastly, Loss-Weighted SoftAdapt not only considers the rates of change, but also considers the value of loss functions (an average of each over the last `n` iterations, in this case `n=5`). Though the first component still recieves the highest attention value in the Loss-Weighted SoftAdapt, the value of the second component is slightly larger than in the first case.

### Example 2
Let us consider the same three loss components as before, but now with another loss that is performing the worst. That is:

```python
loss_A = ops.array([1, 2, 3, 4, 5])
loss_B = ops.array([150, 100, 50, 10, 0.1])
loss_C = ops.array([1500, 1000, 500, 100, 1])
loss_D = ops.array([10, 20, 30, 40, 50])
```
Intuitively, the fourth loss function should recieve the most amount of attention from the optimizer, followed by loss A. The loss components B and C should recieve the least, with component B recieving slightly higher weight than component C. Using the same objects we defined in Example 1, we now want to see how each variant calculates the weights:

(1) `SoftAdapt`: 
```python
softadapt_object.get_component_weights(loss_A, loss_B, loss_C, loss_D)
# >>> tensor([2.8850e-01, 1.9070e-03, 1.1299e-22, 7.0959e-01])
```
(2) `NormalizedSoftAdapt`:
```python
normalized_softadapt_object.get_component_weights(loss_A, loss_B, loss_C, loss_D)
# >>> tensor([0.2436, 0.2459, 0.2673, 0.2432])
```
(3) `LossWeightedSoftAdapt`
 ```python
loss_weighted_softadapt_object.get_component_weights(loss_A, loss_B, loss_C, loss_D)
#>>> tensor([3.8861e-02, 5.3104e-03, 3.1465e-21, 9.5583e-01])
```
As before, we see that `SoftAdapt` and `LossWeightedSoftAdapt` follow our intuition more closely. [^2]

[^2]:***In general, we highly recommend using the loss-weighted variant of SoftAdapt since it considers a running average of previous loss values as well as the rates of change***.

## Usage for Training Neural Networks
SoftAdapt is easy to integrate into a Keras training script with the provided `AdaptiveLossCallback` which inherits from the Keras base `Callback` class.

```python
AdaptiveLossCallback(
                components=[lss.name for lss in losses],
                weights=losses_w,
                frequency="epoch",
                beta=0.1,
                algorithm="base",
                clip_weights=True,
                backup_dir=os.path.join(hparams["log_dir"], "backup"),
            )
```

*Please feel free to open issues for any questions or [contribute](CONTRIBUTING.md)!*


## Citing Our Work
If you found our work useful for your research, please cite us as:
```
@article{DBLP:journals/corr/abs-1912-12355,
  author    = {A. Ali Heydari and
               Craig A. Thompson and
               Asif Mehmood},
  title     = {SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks
               with Multi-Part Loss Functions},
  journal   = {CoRR},
  volume    = {abs/1912.12355},
  year      = {2019},
  url       = {http://arxiv.org/abs/1912.12355},
  eprinttype = {arXiv},
  eprint    = {1912.12355},
  timestamp = {Fri, 03 Jan 2020 16:10:45 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1912-12355.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



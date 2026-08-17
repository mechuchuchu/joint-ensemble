# joint-ensemble
``` bash
We have k independently parameterized models f₁,...,fₖ.

For an input x, each model produces logits:

    zᵢ = fᵢ(x)

The ensemble output is defined as the SUM of logits:

    z = Σᵢ zᵢ

The training objective is a SINGLE cross-entropy loss on this
summed-logit output:

    L = CE(Σᵢ zᵢ, y)

The loss is backpropagated through the sum into every member.

There is no per-member CE loss and no peer-to-peer KL/distillation loss.

At inference time, the deployed object is the same summed-logit
ensemble:

    z(x) = Σᵢ fᵢ(x)
```

## Experiment

`experiment.py` downloads [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100)
from Hugging Face and runs a CIFAR-100 comparison with a fixed 45k/5k
train/validation split and the official 10k-image test set. It trains a ResNet-20
under four conditions: single model, conventional independently trained 3-model
ensemble, the summed-logit objective above, and a joint mean-logit control. It
uses mixed precision and `torch.compile`; temperature is fitted only on validation logits.
The script materializes the small image dataset as CPU tensors once before training,
so repeated epochs do not incur PIL decoding overhead.

```bash
source /venv/main/bin/activate
python experiment.py --epochs 20 --members 3
```

Results are written to `results/summary.json` and `results/summary.csv`.
For ensemble conditions, the JSON also records member-only accuracy, true-vs-best-wrong
logit margins, pairwise logit correlations, and leave-one-out ensemble accuracy.

## Width-2 MNIST member-count ablation

`mnist_width2_ablation.py` studies member scaling with independently parameterized
`784 -> 2 -> 10` MLPs on the Hugging Face `ylecun/mnist` dataset. It evaluates
both conventional independent training and the joint summed-logit objective for
`K = 1, 2, 3, 5, 8, 16` members. The member axis is vectorized, so the model
does not pay a Python-call overhead for each member.

```bash
python mnist_width2_ablation.py --epochs 30
```

The output CSV records ensemble accuracy/NLL, member-only accuracy, pairwise
logit correlation, and leave-one-out accuracy for every member count.

## Deep-bottleneck gradient check

Before comparing an ensemble against a single deep network, run the gradient-flow
check for the `784 -> 128 -> 64 -> 32 -> 2 -> 10` MLP:

```bash
python deep_bottleneck_gradient_check.py --epochs 30
```

It records RMS gradient, parameter RMS, and their ratio for every linear layer on
the first batch of each epoch, together with test accuracy. Results are written to
`deep-bottleneck-gradient-results/gradient_flow.csv` and `.json`.

## Random-vector memorization

`random_vector_memorization.py` measures how random-label memorization changes as
the number of 64-dimensional Gaussian input vectors grows. It compares a single
small MLP, eight independently trained small models (bagging), eight jointly
trained small models, and a parameter-matched dense MLP. The total parameter count
is 11,856 for the eight-member ensemble and 11,791 for the dense reference.

```bash
python random_vector_memorization.py
```

To repeat the comparison with full-batch L-BFGS and line search rather than
mini-batch AdamW, use for example:

```bash
python random_vector_memorization.py --optimizer lbfgs --vectors 2048 8192 --trials 1
```

## Joint-initialization divergence

`joint_initialization_divergence.py` studies whether jointly trained members stay
collapsed or diversify when initialized independently versus as a shared base plus
Gaussian noise from `0` through `1e-1`. It records parameter cosine similarity,
relative parameter spread, logit correlation, prediction disagreement, and member
and ensemble accuracy throughout training.

```bash
python joint_initialization_divergence.py
```

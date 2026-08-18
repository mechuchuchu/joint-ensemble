# Research Notes: Joint Logit Ensembles

Date: 2026-08-17 (UTC)

## Objective

The practical setting is resource-constrained distributed training. Each participant
can host only a small model, while the desired collective capability exceeds any
single participant's model capacity. The primary comparison is therefore:

1. independently trained small models, combined as a logit bagging ensemble; and
2. independently parameterized small models jointly trained with one cross-entropy
   loss on their summed logits.

A larger dense model is not the primary competitor. It is an upper-reference model
when it can fit on one device, and a useful probe of representational limitations.

For a joint ensemble with member logits `z_i`, the objective is:

`CE(sum_i z_i, y)`.

There are no member-local CE losses and no distillation losses in the joint condition.
The gradient supplied to every member logit is identical:

`dL/dz_i = softmax(sum_j z_j) - one_hot(y)`.

## CIFAR-100 pilot

Setup: three ResNet-20 members, a fixed 45k/5k train/validation split, CIFAR-100
test evaluation, mixed precision, and Inductor compilation without CUDA Graph output
reuse. The 10-epoch pilot produced:

| Condition | Test accuracy | NLL | ECE |
| --- | ---: | ---: | ---: |
| Single model | 49.11% | 1.862 | 1.36% |
| Independent mean ensemble | 53.00% | 1.691 | 2.59% |
| Joint summed-logit ensemble | **55.48%** | **1.577** | 2.40% |
| Joint mean-logit control | 49.94% | 2.967 | 29.95% |

The joint-sum ensemble improved accuracy by 2.48 percentage points over independent
bagging in this pilot. The mean-logit control was substantially worse under the same
learning rate; dividing the logits also divides each member's loss gradient, so it
is an optimization control rather than a scale-only control.

Member diagnostics were strongly different between independent and joint training:

| Diagnostic | Independent ensemble | Joint sum |
| --- | ---: | ---: |
| Mean member-only accuracy | 49.44% | 15.58% |
| Mean pairwise logit correlation | 0.91 | 0.06 |
| Full ensemble accuracy | 53.00% | 55.48% |
| Accuracy after worst leave-one-out removal | 51.85% | 20.22% |

Joint members were weak stand-alone classifiers but highly complementary. The third
member was materially more important than the other two, so the observed solution
was complementary but not evenly balanced.

## Width-2 MNIST member-count ablation

Setup: each member was a `784 -> 2 -> 10` MLP. The member axis was vectorized but
the parameters were independent. Training used 30 epochs on MNIST.

| Members | Independent accuracy | Joint-sum accuracy | Joint member-only accuracy | Joint logit correlation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.90% | 69.92% | 69.92% | -- |
| 2 | 72.39% | 78.30% | 27.56% | -0.32 |
| 3 | 80.57% | 86.97% | 18.82% | -0.24 |
| 5 | 80.14% | 91.89% | 14.75% | -0.11 |
| 8 | 80.94% | 94.10% | 14.36% | -0.04 |
| 16 | 80.52% | **96.11%** | 14.44% | 0.01 |

Independent bagging plateaued around 80--81%, while joint training continued to
improve with member count. Individual joint members became poor stand-alone
classifiers but retained complementary output signals.

## Dense reference and gradient flow

The deep bottleneck reference `784 -> 128 -> 64 -> 32 -> 2 -> 10` was checked before
using it as a dense reference. Across 30 epochs, its first-layer relative gradient
(gradient RMS divided by parameter RMS) averaged `1.27e-2` and never fell below
`2.70e-3`; it reached 97.01% MNIST test accuracy. Gradient flow was therefore
adequate for this experiment.

The deep reference has 110,912 parameters. A width-2 joint ensemble with 69 members
has 110,400 parameters. Their MNIST test accuracies were 97.01% and 96.95%,
respectively. This is a performance tie on this task, but it does not make the
architectures functionally identical: the dense model has hidden-layer interactions,
whereas joint members communicate only through the output loss and logit sum.

## Random-vector memorization

Random 64-dimensional Gaussian vectors were assigned random 10-class labels. The
small model was `64 -> 16 -> 16 -> 10` (1,482 parameters). Eight-member ensembles
had 11,856 parameters. The parameter-matched dense reference was
`64 -> 77 -> 77 -> 10` (11,791 parameters).

### AdamW, fixed training budget

Three-trial mean training accuracy:

| Vectors | Single | Bagging | Joint sum | Dense |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 100.0% | 100.0% | 100.0% | 100.0% |
| 512 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2,048 | 51.3% | 94.1% | 100.0% | 100.0% |
| 8,192 | 23.3% | 60.1% | **83.8%** | 79.0% |

Under this specific mini-batch AdamW budget, joint training exceeded bagging by a
large margin and modestly exceeded the dense reference at 8,192 vectors. This was
not a capacity theorem: the dense model had not necessarily converged.

### Full-batch L-BFGS with line search

One trial, 30 outer L-BFGS steps, up to 10 closure iterations per step, and
strong-Wolfe line search:

| Vectors | Single | Bagging | Joint sum | Dense |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 52.88% | 98.49% | 100.00% | 100.00% |
| 8,192 | 24.51% | 54.97% | 86.99% | **100.00%** |

L-BFGS changed the conclusion about dense capacity: at 8,192 vectors, the dense
reference fully memorized the data while the joint ensemble did not. Joint training
still substantially outperformed independently trained bagging. This supports the
resource-constrained framing: joint training is a way to obtain more collective
capacity from distributed small models than bagging, not a claim that it always
beats an equally resourced, co-located dense model.

## Initialization-divergence experiment

Setup: the eight-member joint ensemble above trained on 2,048 random vectors for 800
AdamW steps. Conditions were independent random initialization and a shared base
initialization plus independent Gaussian perturbations.

| Initialization | Ensemble accuracy | Mean parameter cosine | Mean logit correlation | Prediction disagreement |
| --- | ---: | ---: | ---: | ---: |
| Independent random | 100.0% | 0.009 | -0.087 | 92.40% |
| Shared base + 0 | 41.57% | 1.000 | 1.000 | 0.00% |
| Shared base + 1e-5 | 100.0% | 0.360 | -0.064 | 91.24% |
| Shared base + 1e-4 | 100.0% | 0.317 | -0.064 | 91.62% |
| Shared base + 1e-3 | 100.0% | 0.285 | -0.060 | 91.12% |
| Shared base + 1e-2 | 100.0% | 0.169 | -0.094 | 93.36% |
| Shared base + 1e-1 | 100.0% | 0.045 | -0.073 | 92.15% |

Exact symmetry is invariant: equal weights receive equal gradients and stay equal.
However, a `1e-5` perturbation was enough to trigger functional diversification.
For that condition, mean parameter cosine changed from approximately 1.000 at step
0 to 0.688 at step 100 and 0.360 at step 800; mean logit correlation changed from
1.000 to 0.129 and then -0.064. Thus tiny initialization differences are amplified
into low-correlation member functions under the joint objective.

## Reproducibility and limitations

- Source code uses public Hugging Face CIFAR-100 and MNIST datasets where applicable.
- Generated datasets, and smoke-test outputs are ignored by Git.
- The CIFAR and MNIST runs are pilots, not multi-seed statistical studies.
- The L-BFGS random-vector result currently has one trial and should be repeated over
  multiple seeds before making a variance-sensitive claim.
- Random-label memorization measures optimization and capacity, not useful-data
  generalization or distributed communication efficiency.

## CIFAR-100 ResNet-18: logit-sum versus hidden-sum

Setup: four independently parameterized CIFAR-style ResNet-18 members with
width multiplier 16, trained jointly with one cross-entropy loss. The member
axis was folded into the channel axis and processed with grouped convolutions
so that the four members could be evaluated in a single vectorized model.
Training used the Hugging Face `uoft-cs/cifar100` dataset, the same fixed
45k/5k train/validation split convention, standard CIFAR augmentation, SGD
with momentum, cosine learning-rate decay, and 20 epochs.

The experiment compared two aggregation locations:

1. `logit_sum`: each member has its own classifier and the four logits are
   summed, `sum_i W_i h_i`.
2. `hidden_sum`: the four penultimate hidden vectors are summed first and a
   single shared classifier produces the logits, `W sum_i h_i`.

The hidden-sum condition therefore introduces communication before the final
classifier and has one shared head, while logit-sum preserves independent
member heads. This creates a small parameter-count difference that is recorded
below; the comparison is not parameter-exact.

| Condition | Parameters | Test accuracy | Raw NLL | Calibrated NLL | Temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logit sum | 2,852,304 | **65.33%** | **1.2254** | **1.2095** | 1.171 |
| Hidden sum | 2,813,604 | 63.27% | 1.3072 | 1.2924 | 1.164 |

Under this single 20-epoch run, logit aggregation exceeded hidden aggregation
by 2.06 percentage points. Hidden-sum learned successfully but started more
slowly under the same optimization schedule, consistent with the shared head
having to decode a summed representation rather than receiving member-specific
class logits. This is a pilot result from one seed, not a statistical claim.

The implementation and outputs are in `resnet18_hidden_vs_logit_sum.py` and
`resnet18-hidden-results/`. The grouped-convolution implementation was used
for throughput and vectorization; the result should therefore be followed up
with multiple seeds and a parameter-matched hidden aggregation control before
drawing architectural conclusions.

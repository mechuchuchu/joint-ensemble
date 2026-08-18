### Additive Logit Correction

At each stage, the previous model’s logits are treated as a fixed baseline using stop-gradient, and a new neural network learns **correction logits** that are added to the previous prediction.

[
z_t = \operatorname{stopgrad}(z_{t-1}) + f_t(x)
]

The new model is trained by minimizing

[
\mathcal{L}_t = CE(z_t, y),
]

so it does not learn the entire prediction from scratch. Instead, it learns to **correct the remaining errors of the previous model in logit space**.

Repeating this process gives

[
z_T = f_1(x)+f_2(x)+\cdots+f_T(x).
]

In short:

> **“A stage-wise additive logit correction scheme where each new neural network learns a residual correction on top of the previous model’s frozen logits.”**

The idea is conceptually related to additive modeling and boosting, but the method itself is defined specifically as **neural logit-level residual correction**, rather than assuming any particular existing boosting algorithm.

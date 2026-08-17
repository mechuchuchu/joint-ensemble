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

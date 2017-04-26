# ELEG5491

All models are trained using [``Parrots``](http://parrotsdnn.org/).
Traning settings can be found in ``session.yaml``s.
Code for filter visulization is [here](plot.jl). It uses [julia](http://julialang.org/).

## Network structures

* Original: [code](net.original.py), [pdf](net.original.pdf), [yaml](net.yaml).
* With msra init: [code](net.msra.py), [pdf](net.msra.pdf), [yaml]().
* With BN: [code](net.bn.py), [pdf](net.bn.pdf), [yaml](runs/net1_bn/model.yaml).

Note: See [runs](runs) folder for training logs.

## Training curves referenced in the report

Note: access to the links are restricted to campus networks.

* [Original](http://pavi.goforget.com/Home/Monitor/3437) ([model](runs/net1/model.yaml), [session](runs/net1/session.yaml))
* [No weight decay](http://pavi.goforget.com/Home/Monitor/3438) ([model](runs/net1_no_weight_decay/model.yaml), [session](runs/net1_no_weight_decay/session.yaml))
* [Msra](http://pavi.goforget.com/Home/Monitor/3446) ([model](runs/net1_msra/model.yaml), [session](runs/net1_msra/session.yaml))
* [BN](http://pavi.goforget.com/Home/Monitor/3453) ([model](runs/net1_bn/model.yaml), [session](runs/net1_bn/session.yaml))
* [Augmentation](http://pavi.goforget.com/Home/Monitor/3457) ([model](runs/net1_aug/model.yaml), [session](runs/net1_aug/session.yaml))

## Comparisons

* [Original vs No weight decay](http://pavi.goforget.com/Home/Comparing/214)
* [Original vs Msra](http://pavi.goforget.com/Home/Comparing/215)
* [Original vs BN](http://pavi.goforget.com/Home/Comparing/216)
* [Original vs Augmentation](http://pavi.goforget.com/Home/Comparing/217)

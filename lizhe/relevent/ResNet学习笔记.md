# Is learning better networks as easy as stacking more layers?
## Overview
* Q1: **Why we need a deeper network?**
* A1: CNN can extract low/mid/high-level features through different filters, that means the deeper network is, the more features we can extract.
* Q2: **Why not simply add more and more layers to networks?**
* A2: On the one hand it will cause vanishing/exploding gradient(addressed by normalized initialization/BN)，on the other hand, we can't ignore the *degradation*(with the network depth increasing, accuracy gets saturated and then degrades rapidly)<br>

![degradation](https://mohitjainweb.files.wordpress.com/2018/06/degradation-problem1.png?w=700)


## Experiment
```
shallow model -> add identity mapping layers -> deep model
```
&emsp;&emsp;It's abnormal that deep model is worse than shallow one 

## Key point
&emsp;&emsp;The solvers have difficulties in approximating identity mapping by multiple non-linear layers

## Method
* Denoting the desired underlying mapping as H(x)
* Let the stacked non-linear layers fit F(x)
* Define F(x) = H(X) - x
* Then H(X) = F(X) + x

&emsp;&emsp;By adding x (realized by shortcut connection), we prone to fit F(X) rather than H(X).

![demo](https://upload-images.jianshu.io/upload_images/6095626-49ac0caeb5525b93.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Note
&emsp;&emsp;I suppose that residual function vanish the error accumulated to some extents. The 'x' can fully be transmitted to next layer

## Reference
* VLAD in image recognition
* Multi-grid method in PDEs
* Shortcut connection in MLP
* Highway network - gating functions

## Doubt
* convergence rate ~ # layers
* error accumulation
r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        raise NotImplementedError()
    if opt_name == 'momentum':
        raise NotImplementedError()
    if opt_name == 'rmsprop':
        raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Analyze your results from experiment 1.1. In particular,
1. Explain the effect of depth on the accuracy. What depth produces the best results and why do you think that’s the case?
2. Were there values of L for which the network wasn’t trainable? what causes this? Suggest two things which may be done to resolve it at least partially.


In experiment 1.1, we examine the effect of increasing the network depth, given a fixed width. We looked on two width: 
32 and 64. 
We can see that when increasing the depth of the network with 32 width from 2 to 4 the network improves. It is larger, 
and able to over fit the training set better and get better result on the test set as-well. But, when increasing the 
depth even more (depth 4 and 8) we are not able to train the network, and to loss and acc
is not improving at all over the epochs.
Also, note, that in width 62 the best depth is actually 2 (and not 4).
This happen due to vanishing/exploding gradients (I printed the norm of the model weight gradient to verify). 
As the network gets deeper the gradient vanish in the back propagation (When the network is wider it may happen sooner, 
as we saw here).

Possible solution can be: 
1. Implement skip-connection, which enable the gradient to propagate to earlier layers better.
2. Implement Batch Normalization, which prevents the gradient to become too small/ too big.

"""

part3_q2 = r"""
We can see that in general given the same depth, wider is better. Fitting the training set better, and usually also 
the test set. You can also more easily increase the width than the depth, we don't observe the vanishing/exploding 
gradients as when increasing the depth. Admittedly, the loss-difference given different depth seemed more singificant
than given different width.
"""

part3_q3 = r"""
I was not able to run with L > 1 with different kernels. 
As we saw depth >4 with width network resulted in vanishing/exploding gradients  in the former experiments. 
So as here. To be able to run dipper networks with varying number of filters I have to solve this prolbem first.
"""


part3_q4 = r"""
### Question 4 
 ___________________________________________________________________________________________________
1.  Explain your modifications to the architecture which you implemented in the `YourCodeNet` class:
 ___________________________________________________________________________________________________
 The modifications we made in the architecture:
In all of the experiments above we saw 
1. significant over-fitting
2. dipper network result in the problem of vanishing gradient and hard to train (but solving it may promise better
results).

So we wanted to add regularization and solution for the vanishing gradient. 
Therefore we implemented skip-connection with batch normalization. We took inspiration from ResNet 
(torchvision implementation) implementing residual block. 

Each block size (between skip-connection) should be according to the size of K and the number of block will be set by L.
 We only pool in the first & last blocks. 
The block:
(Conv -> BatchNorm -> Relu) * block_size + Residual connection (after linear projection to the right dimension) -> Relu.
(And max pol for the first and last block)

Also, as we observed there is a point where the loss stack - small absolution around some local minim - I decided
to add implementation of reducing the learning rate during the training.

# Note: To be able to use the API as is, we will se pool_every = L.

 ___________________________________________________________________________________________________
2. Analyze the results of experiment 2. Compare to experiment 1.
 ___________________________________________________________________________________________________
The first and most important - I was now able to train deeper network. 
Before I could only run for L=1, while now I got up to 8.
I was not able to train the model for L=12, I got vanishing gradients again. Maybe I need to add more 
skip connection (not just in the blocks but between blocks). Or find other solutions. 


"""

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
    wstd = 0.2
    lr = 0.03
    reg = 0.05
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======

    if opt_name == 'vanilla':
        wstd = 0.1
        lr = 0.001
        reg = 0.001
    if opt_name == 'momentum':
        wstd = 0.1
        lr = 0.001
        reg = 0.001
    if opt_name == 'rmsprop':
        wstd = 0.1
        lr = 0.0001
        reg = 0.001

    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 3e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
Since dropout is a tecnique for improving the generalization of a network, we expect that it would roll away the performance on the training set, proportionly to the dropout-rate. On the test set, we cannot predict which drop out rate would lead to the best preformance (we don't know which rate leads the best generalization), but it sounds reasonable that dropout=0 would lead to uderfit, dropout=0.8 would lead to overfit and that dropout=0.4 would lead the best performance.
Indeed,the results described in the graphs shows that as the dropout rate increases, the performance on the train set rolls away. On the test set, we get similar satefying results for dropout=0.4 and dropout=0, better then those recieved when dropout=0.8. That can be be explained by that the model underfits when dropout=0.8 and does not tend to significantly overfit not using dropout\using low dropout rate. It make sense that dropout between 0 to 0.4 (i.e, 0.2) would give the best performance. 

"""

part2_q2 = r"""The input of the cross entropy loss is an unnormalized raw value (logits) and the target is class labels. It outputs a continous value that describes the closness of the predicted values to the class labels. 
The acuuracy only measures the poroportion of correct labels and thus, is summing up discrete values.Thus, is only afected by the given label, not by the excact logit values.
Namely, accuracy represents if I'm right, cross entropy represents how "bad" my errors are. Thus, one of these values may increase for a few epochs while the other also increases 


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
 ___________________________________________________________________________________________________
1.  Explain your modifications to the architecture which you implemented in the `YourCodeNet` class:
 ___________________________________________________________________________________________________
In all of the experiments above we saw 
1. significant over-fitting
2. dipper network result in the problem of vanishing gradient and hard to train (but solving it may promise better
results).

So we wanted to add regularization and solution for the vanishing gradient. 
Therefore we implemented skip-connection with batch normalization. We took inspiration from ResNet 
(torchvision implementation) implementing residual block. 

Each block size (between skip-connection) should be according to the size of K and the number of block will be set by L.
 We only pool in the first & last blocks. 
### The block:
(Conv -> BatchNorm -> Relu) * block_size + Residual connection (after linear projection to the right dimension) -> Relu.
(And max pol for the first and last block)

Also, as we observed there is a point where the loss stack - small absolution around some local minim - I decided
to add implementation of scheduler to reduce the learning rate during the training. Honestly, it didn't help as I 
wished. 

Note: To be able to use the API as is, we will se pool_every = L.

 ___________________________________________________________________________________________________
2. Analyze the results of experiment 2. Compare to experiment 1.
 ___________________________________________________________________________________________________
 
The first and most important - we are now able to train deeper network. 
Before we could only run for L=1, while now I got up to 8.
We was not able to train the model for L=12, I got vanishing gradients again. Maybe I need to add more 
skip connection (not just in the blocks but between blocks). Or find other solutions. 

Comparing the impact of depth on the network performance:
With L=2 we get the best result, which are much better than the previous Cov net implementation (~82% instead of ~68).
Interesting to see that when we increase L the result is not improving, even due the network still learning. 
I believe it also due to the fact that I used max batches 100, so we actually use 1/4 of the dataset.
Maybe if we would use all of the data we could increase the depth and get better accuracy. But the running time was 
too slow and I didn't have the time to do so.

We can see that the network still over-fit but significantly less (10% difference between train to test compared to 
25% before). I believe we could reduced it even more using dropout or other solution, but I didn't got to it.

__________________________________________________________________________________________________________
Selection of Hyperparameter across the experiment: 
__________________________________________________________________________________________________________

I manually tried different setting; to focus on the main objective we are experimenting on I didn't change the 
hyperparameter for the specific experiment (aka if we are changing L to measure L effect I left all the 
hyper-parameter the same while doing so, to avoid noice result).

"""

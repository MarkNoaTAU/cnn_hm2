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
    wstd=0.2
    lr=0.03
    reg=0.05
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    
    if opt_name == 'vanilla':
        wstd=0.1
        lr=0.001
        reg=0.001
    if opt_name == 'momentum':
        wstd=0.1
        lr=0.001
        reg=0.001
    if opt_name == 'rmsprop':
        wstd=0.1
        lr=0.0001
        reg=0.001
		
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd=0.1
    lr=3e-4
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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

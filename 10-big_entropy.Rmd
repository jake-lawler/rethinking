# Big Entropy and the Generalized Linear Model {#big_entropy}

## Chapter Notes


### Maximum Entropy {-}

The chapter introduces a justification for maximum entropy approaches that appears in Jaynes' Probability Theory. Jaynes attributes the approach to Graham Wallis. We have $m$ different possibilities, and we want to assign probabilities $\{ p_1, \dots, p_m \}$ to them, with the probabilities summing to 1. We want to do this by making use of some information $I$ that we have. 

Jaynes described a thought experiment in which a blindfolded person throws pennies into $m$ equal boxes, so that any penny has an equal chance of landing in any of the boxes. The person throws some large number $n >> m$ of pennies and at the end we count up all the pennies in each box, divide by the total number of pennies and take this to be the probability assigned to the boxes by our experiment. For each box $i = 1,2,\dots,m$

$$
p_i = \frac{n_i}{n} 
$$

where $n_i$ is the observed number of pennies in box $i$.

The probability of any particular assignment is given by the multinomial distribution:

$$
m^{-n} \frac{n!}{n_1! \dots n_m!}.
$$

After the experiment, we check whether the probability assignment is consistent with our information $I$. If it is not, we ask the blindfolded person to try again. We continue in this way until a probability assignment is accepted. 

What is the most likely probability distribution to be chosen by this experiment? The answer is whatever one maximises 

$$
W = m^{-n} \frac{n!}{n_1! \dots n_m!}
$$

subject to the constraints of $I$. This is equivalent to finding the distribution which maximises $\frac{1}{n} \log(W)$:

$$
\begin{aligned}
\frac{1}{n} \log(W) &= \frac{1}{n} \left( \log(n!) - \log(n_1!) - \dots - \log(n_m!) \right) \\
 &= \frac{1}{n} \left( n \log(n) - n + \sqrt{2 \pi n} + \frac{1}{12n} + \mathcal{O}\left(\frac{1}{n^2}\right) \right) \\ 
 &\quad - \frac{1}{n} \left( n_1 \log(n_1) - n_1 + \sqrt{2 \pi n_1} + \frac{1}{12n_1} + \mathcal{O}\left(\frac{1}{n_1^2}\right)  \right) \\
 &\vdots \\
 &\quad- \frac{1}{n} \left( n_m \log(n_m) - n_m + \sqrt{2 \pi n_m} + \frac{1}{12n_m} + \mathcal{O}\left(\frac{1}{n_m^2}\right)  \right)  \\
 &= \left( \log(n) - 1 + \sqrt{2 \pi \frac{1}{n}} + \mathcal{O}\left(\frac{1}{n^2}\right) \right) \\ 
 &\quad- \left( p_1 \log(np_1) - p_1 + \sqrt{2 \pi \frac{1}{n}p_1} + \frac{1}{12n^2p_1} + \mathcal{O}\left(\frac{1}{n_1^2}\right)  \right) \\
 &\vdots \\
 &\quad- \left( p_m \log(np_m) - p_m + \sqrt{2 \pi \frac{1}{n}p_m} + \frac{1}{12n^2p_m} + \mathcal{O}\left(\frac{1}{n_m^2}\right)  \right) \\
 &\to -\sum p_i \log(p_i) +\log(n) -\sum p_i \log(n) - 1 + \sum p_i \\
 &= -\sum p_i \log(p_i)
\end{aligned}
$$

with the limit taken as $n \to \inf$ and $n_i \to \inf$ so that $p_i$ remains constant. 

Note: I used the Stirling approximation above. Initially used a tag in the latex to explain but doesn't seem to work with the aligned environment. Correct later. 

We've recovered the formula for information entropy introduced in Chapter 7.

The chapter then goes on to introduce proofs that the Gaussian distribution is the maximum entropy distribution given only a finite variance, and that the binomial is the maximum entropy distribution given only some constant expected value and two unordered possible events. First the Gaussian.   


Here's the probability density function of the Gaussian:

$$
p(x) = (2 \pi \sigma^2)^{-1/2} \exp \left( - \frac{(x- \mu)^2}{2 \sigma^2} \right)
$$
and its entropy:

$$
H(p) = - \int p(x) \log p(x) dx = \frac{1}{2} \log(2 \pi e \sigma^2) 
$$

We want to consider $q(x)$, some other probability density function with the same variance $\sigma^2$. The basic structure of this proof is that we reintroduce KL divergence from Chapter 7


### Generalized Linear Models {-}

This part of the chapter extends the notion of a linear model we've been working with so far to include non-Gaussian likelihoods. There is an introduction to the exponential family, and then a discussion of two common link functions that we'll be using over the rest of the book: the logit link and the log link.

The logit link is used for parameters that represent probabilities, and that therefore must be between 0 and 1. Since a linear function of a predictor may well return values for parameters outside of these boundaries, we want a function to transform the output of our linear function. E.g.

$$
\begin{aligned}
y_i &\sim \text{Binomial}(n,p_i) \\
\text{logit}(p_i) &= \alpha + \beta x_i
\end{aligned}
$$

with the logit function representing the log odds like so:
$$
\text{logit}(p_i) = \frac{p_i}{1 - p_i}
$$

So in this model, our parameter $p_i$ is the inverse-logit transform of the linear model:

$$
p_i = \frac{\exp(\alpha + \beta x_i)}{1 + \exp(\alpha + \beta x_i)}.
$$

The log link function is for parameters that are only defined over positive real numbers. E.g.

$$
\begin{aligned}
y_i &\sim \text{Normal}(\mu,\sigma) \\
\log(\sigma_i) &= \alpha + \beta x_i
\end{aligned}
$$

Definitionally, $\sigma$ cannot be negative, and the log transform keeps this from happening. In the model above, our $sigma$ is modelled as the exponentiation of the linear model. 

## Questions

There are no questions at the end of this chapter.


## Further Resources {-}


On the link between Bayesian conditioning and entropy maximisation:

Williams (1980): Bayesian Conditionalisation and the Principle of Minimum Information (http://www.yaroslavvb.com/papers/williams-conditionalization.pdf)

Caticha, A. and Griffin, A. (2007). Updating probabilities. In Mohammad-Djafari, A., editor, Bayesian Inference and Maximum Entropy Methods in Science and Engineering, volume 872 ofAIP Conf. Proc.

Griffin (2008): Maximum Entropy: The Universal Method for Inference (https://arxiv.org/ftp/arxiv/papers/0901/0901.2987.pdf)

Conrad's paper deriving various maximum entropy distributions. https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf
Work through this and fill out the Gaussian and Binomial arguments above.

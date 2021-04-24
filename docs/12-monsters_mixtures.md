# Monsters and Mixtures {#monsters_mixtures}




## Chapter Notes


### Over-Dispersed Counts {-}

The chapter opens with a discussion of over-dispersion in count data - when the data exhibits more variation than can be explained by a binomial or Poisson distribution. We'll try to address this using two types of continuous mixture models - beta-binomial and negative-binomial models. 

The beta-binomial distribution is the binomial distribution, except that instead of the probability of success p being fixed, it is drawn from some beta distribution.

The chapter example returns to the UCB admissions data from the previous chapter, except this time we allow each row of the data (i.e. each department / gender combination) is allowed to have a different probability of admission - drawn from a beta distribution. I'd previously seen beta distributions with $\alpha$ and $\beta$ parametrisation, but the chapter uses $\bar{p}$ and $\theta$, with $\alpha = \bar{p}\theta$ and $\beta = (1-\bar{p})\theta$. Here $\bar{p}$ is the average probability and $\theta$ is a shape parameter.

Here is the model used for the UCB data:

$$
\begin{aligned}
A_i &\sim \text{BetaBinomial}(N_i,\bar{p_i},\theta) \\
\text{logit}(\bar{p_i}) &= \alpha_{\text{gen}[i]} \\
\alpha_j &\sim \text{Normal}(0,1.5) \\
\theta & \sim \phi + 2 \\
\phi & \sim \text{Exponential}(1)
\end{aligned}
$$
The higher the value of $\theta$, the more concentrated the probability. When $\bar{p}_i$ is 0.5, a $\theta$ of 2 gives a completely flat distribution. This is why $\theta$ is assigned a minimum of two in the model above.

We fit the model, and examine the posterior:




```
##             mean        sd       5.5%     94.5%    n_eff     Rhat4
## a[1]  -0.4345426 0.4057828 -1.0862455 0.2106073 1230.289 0.9999798
## a[2]  -0.3240307 0.4072708 -0.9423233 0.3258636 1251.730 1.0030795
## phi    1.0225527 0.8188769  0.1004436 2.5963127 1554.513 0.9998056
## theta  3.0225526 0.8188769  2.1004399 4.5963127 1554.513 0.9998056
```

The probability of admission increases with the value of $\alpha$. The difference between $\alpha$ for men and women is 


```r
mean(post_UCB$diff_a)
```

```
## [1] -0.1105119
```

suggesting the model believes women are more likely to be admitted. However the standard deviation of this value is 0.5877854;the model is very uncertain. We contrast this with model m11.7 in the last chapter, which predicted that men were more likely to be admitted, and was quite a bit more confident about this. Even though we haven't included department in the model, allowing $p$ to vary by department / gender combination has captured some of the variation between departments.

Here's a plot: 

![](12-monsters_mixtures_files/figure-epub3/unnamed-chunk-5-1.png)<!-- -->


The chapter then moves on to the use of negative binomial (or gamma-Poisson) continuous mixture models to address over-dispersion. These are Poisson models, where the rate is allowed to vary across observations by drawing it from a gamma distribution. The gamma-Poisson distribution has two parameters, one is a rate parameter $\lambda$ and one ($\phi$) controls the variance. The distribution has $\text{var} = \lambda + \frac{\lambda^2}{\phi}$ so smaller $\phi$ implies larger variance.

The chapter refits the tool data from chapter 11 with a gamma-Poisson distribution, the idea is that we expect an outlier point like Hawaii to become less influential, because the model can accommodate more variation (in a Poisson distribution the variance necessarily equals the mean).



Here are the posterior plots of the tools model using a Poisson distribution, and using the gamma-Poisson:



![](12-monsters_mixtures_files/figure-epub3/unnamed-chunk-8-1.png)<!-- -->

Here blue dots are high contact, and red low contact societies. The size of the points is scaled by Pareto k-value. The gamma-Poisson is less influenced by Hawaii, and consequently much more uncertain in large populations.


### Zero-Inflated Outcomes {-}

The zero-inflated Poisson model is introduced as an example of a mixture model: models that use multiple probability distribution to measure the influence of more than one cause. With zero-inflation, we aim to model a count variable where zeros can be produced in more than one way. In the monastery example in the chapter, each day monks have a fixed probability of taking the day off (maybe they spend the day drinking wine). On these days they will produce zero manuscripts. If they do work, they will produce some (low) number of manuscripts over the course of the day, and this might also be zero (maybe they just finished a bunch). So a zero can be produced two ways (broadly, as the outcome of a binomial process, or a Poisson process).

The chapter introduces the zero-inflated Poisson distribution: a binomial / Poisson mixture. The probability of a zero is:

$$
\begin{aligned}
\text{Pr}(0|p,\lambda) & = \text{Pr}(\text{drink}|p) + \text{Pr}(\text{work}|p) \times \text{Pr}(0|\lambda) \\
&= p + (1-p)\exp(-\lambda)
\end{aligned}
$$
and the probability of some non-zero figure is:

$$
\begin{aligned}
\text{Pr}(y > 0|p,\lambda) & = \text{Pr}(\text{work}|p) \times \text{Pr}(y|\lambda) \\
&= (1-p)\frac{\lambda^y \exp(-\lambda)}{y!}
\end{aligned}
$$
The formulas here come the Poisson likelihood (rate $\lambda$) and the binomial (probability $p$ of taking the day off).

A zero-inflated Poisson model will look something like this, for some predictor x:

$$
\begin{aligned}
y_i & \sim \text{ZIPoisson}(p_i,\lambda_i) \\
\text{logit}(p_i)&= \alpha_p + \beta_p x_i\\
\log(\lambda_i)&= \alpha_\lambda + \beta_\lambda x_i\\
\end{aligned}
$$

The chapter expands on zero-inflation Poisson models by simulating some data from our fictional monastery, fitting a model, and attempting to recover the data-generating process.


### Ordered Categorical Outcomes {-}

Here the outcome we want to predict is made up of some number of categories, like a multinomial. Except that the categories are ordered, e.g. an approval rating from 1 (strongly disapprove) to 5 (strongly approve). The ordering is important, but the scale is not necessarily linear, and so shouldn't be modelled as a continuous outcome.

The way of dealing with this described in the chapter is to use a log cumulative odds function, as we have used the log odds link in previous chapters. The chapter introduces a trolley problem example where respondents grade the moral permissability of action in a scenario on a scale of 1 to 7.

Here I've reproduced some charts in the chapter that show the counts of each response, the cumulative proportion, and then the log cumulative odds.

![](12-monsters_mixtures_files/figure-epub3/unnamed-chunk-9-1.png)<!-- -->

A model with no predictors is introduced, to check that we can recover the cumulative proportions in the data in the posterior distribution: 


```r
m12.4 <- ulam( alist( 
  R ~ dordlogit( 0 , cutpoints ), 
  cutpoints ~ dnorm( 0 , 1.5 )
) , data=list( R=data_trol$response ), chains=4 , cores=4,cmdstan = TRUE )


# cumulative proportions in the data
round(hist_trol$prop_cum, 3)

# model expectations for cumulative proportions
round( inverse_logit(coef(m12.4)) , 3 )
```

Then the chapter explains how to include predictors in this kind of model. The log cumulative odds for each response k is modelled as a linear combination of it's intercept $\alpha_k$ and a standard linear model:

$$
\begin{aligned}
\log\frac{\text{Pr}(y_i \leq k)}{1 - \text{Pr}(y_i \leq k)} &= \alpha_k - \phi_i \\
\phi_i & = \beta x_i
\end{aligned}
$$

The subtraction is conventional, to ensure that positive $\beta$ means the predictor $x$ is positively associated with the outcome $y$.


The model actually used for the trolley data looks like this:

$$
\begin{aligned}
\log\frac{\text{Pr}(y_i \leq k)}{1 - \text{Pr}(y_i \leq k)} &= \alpha_k - \phi_i \\
\phi_i & = \beta_A A_i + \beta_C C_i + \beta_{I,i} I_i \\
\beta_{I,i} &= \beta_I + \beta_{IA} A_i + \beta_{IC} C_i 
\end{aligned}
$$
Here:
* $A_i$ is the value of action on row $i$, 0 or 1.
* $I_i$ is the value of intention on row $i$, 0 or 1.
* $C_i$ is the value of contact on row $i$, 0 or 1.
* $B_I,i$ introduces an interaction effect between intention and action, and intention and contact.



![](12-monsters_mixtures_files/figure-epub3/unnamed-chunk-12-1.png)<!-- -->

We can see that all of the predictors (action - bA, contact - bC, intention - bI) are all negatively associated with permissability. 

Need to revisit this for posterior plots. And also the section on ordered categorical predictors.


## Questions

### 12E1 {-}

#### Question {-}


#### Answer {-}



## Further Resources {-}



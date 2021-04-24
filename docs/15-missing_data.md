# Missing Data and Other Opportunities {#missing_data}



## Chapter Notes

### Measurement Error {-}

We want to build models that allow for measurement error. The book returns to the Waffle House / Divorce example from chapter 5. Both the marriage and divorce columns in the data come with standard errors that we did not make use of back when we first saw this example. The plot on the left here is a straightforward plot of the data, including error bars, on divorce against age of marriage. There's one data point per U.S. state.

![](15-missing_data_files/figure-epub3/unnamed-chunk-2-1.png)<!-- -->

The plot on the left is meant to demonstrate that the standard error is much larger for states with small populations as you'd expect. This is important, because variation in the size of the error among states is likely to introduce biases.

In order to motivate the approach to incorporating measurement data, the chapter draws the following graph of the data generating processes:

![](15-missing_data_files/figure-epub3/unnamed-chunk-3-1.png)<!-- -->

As usual, variables in circles are unobserved. Here the DAG assumes that the marriage rate ($M$) and age at marriage ($A$) influence the divorce rate ($D$). But we don't observe the divorce rate, we observe $D_obs$ which is also influenced by measurement error $e_D$. We can attempt to recover $D$ by assuming a distribution for it, and assigning it a parameter in our model with a specified error. E.g:

$$
D_{\text{OBS},i} \sim \text{Normal}(D_{\text{TRUE},i},D_{\text{SE},i})
$$

Our model will look like this:

$$
\begin{aligned}
D_{\text{OBS},i} &\sim \text{Normal}(D_{\text{TRUE},i},D_{\text{SE},i}) \\
D_{\text{TRUE},i} &\sim \text{Normal}(\mu_i,\sigma) \\
\mu_i &= \alpha + \beta_A A_i + \beta_M M_i\\
\end{aligned}
$$



Here's the posterior for (some of) the model parameters:


```
##              mean         sd       5.5%       94.5%    n_eff    Rhat4
## a     -0.05363235 0.09550644 -0.2044767  0.09792258 3711.458 1.000441
## bA    -0.61627849 0.16425257 -0.8780882 -0.35829778 2274.444 1.001044
## bM     0.05078094 0.16566590 -0.2202174  0.30963604 2115.038 1.001503
## sigma  0.58807411 0.10642548  0.4287034  0.76349803 1361.365 1.001046
```

Compared to the chapter 5 model, $bA$ has almost halved. In this case the impact of measurement error was to exaggerate the effect of marriage age on divorce. However you can't assume that measurement error will always increase the effects of interest, sometimes it can obscure them. Endnote 223 points to some papers on this.

What if there is also measurement error on the predictor variables e.g. marriage rate? Here's the DAG:

![](15-missing_data_files/figure-epub3/unnamed-chunk-6-1.png)<!-- -->

and here's the model:

$$
\begin{aligned}
D_{\text{OBS},i} &\sim \text{Normal}(D_{\text{TRUE},i},D_{\text{SE},i}) \\
D_{\text{TRUE},i} &\sim \text{Normal}(\mu_i,\sigma) \\
\mu_i &= \alpha + \beta_A A_i + \beta_M M_{\text{TRUE},i}\\
M_{\text{OBS},i} &\sim \text{Normal}(M_{\text{TRUE},i},M_{\text{SE},i}) \\
M_{\text{TRUE},i} &\sim \text{Normal}(0,1) \\
\end{aligned}
$$

Standardising the observed marriage rate helps us choose a sensible prior distribution for the true marriage rate. Although later in the chapter (and in an exercise) a prior more informed by the data generating process is trialled.

Revisit: Fit the model, plot figure 15.3.

### Missing Data {-}

Sometimes data is simply missing. We want a principled approach that considers the data generating process.

The chapter introduces a simple example about dogs eating homework to demonstrate:

<img src="15-missing_data_files/figure-epub3/unnamed-chunk-7-1.png" width="50%" /><img src="15-missing_data_files/figure-epub3/unnamed-chunk-7-2.png" width="50%" /><img src="15-missing_data_files/figure-epub3/unnamed-chunk-7-3.png" width="50%" /><img src="15-missing_data_files/figure-epub3/unnamed-chunk-7-4.png" width="50%" />

$S$ is the amount a student studies. It influences homework quality. $D$ is whether a dog has eaten the homework. $H_\text{obs}$ is the quality of observed homework. It is influenced by true homework quality, but is missing in cases when $D$=1 (i.e. a dog has eaten the homework). There are four possible generative processes discussed.

Until I figure out how to caption dagitty objects, let's call these (a), (b), (c), (d) going from the top left corner to the top right, bottom right then bottom left.

 (a)  Dogs eat homework at random
 (b)  Dogs eat the homework of students who study a lot (not paying enough attention to the dog)
 (c)  Noisiness ($X$) influences both homework quality and tendency for homework to be eaten
 (d)  Dogs prefer to eat bad homework

In the first case (a), because whether the dogs eat the homework at random, H is independent of D and so we wouldn't expect the dogs to change the inferences we make about the effect of $S$ on $H$.

The second case (b) is also not so bad. There is a backdoor path from $D$ to $H$ through $S$, but since we want to condition on $S$ anyway it's not terrible.

In both of these cases, the exercises include comparison of inferences made with complete data and when some data is missing (eaten).

The main body of this chapter gives a fuller treatment to scenarios (c) and (d), where things get trickier. We simulate some data:


```r
set.seed(501) 
N <- 1000 
X <- rnorm(N) 
S <- rnorm(N)
H <- rbinom( N , size=10 , inv_logit( 2 + S - 2*X ) )

D <- if_else( X > 1 , 1 , 0 ) 
H_obs <- H
H_obs[D==1] <- NA
```

What's happening here:

* Homework is a binomial variable with 10 trials, where the probability of success is increased by $S$ and decreased by $X$. The chapter says that "the true coefficient on S should be 1.00." but I don't understand why.
* If $X$ is greater than 1, the dog eats the homework. Increased noise is therefore associated both with worse quality homework and missing homework.

Here's a summary of the posterior parameter distributions we get assuming we can see $H$ directly:




```
##         mean         sd      5.5%     94.5%    n_eff    Rhat4
## a  1.1135375 0.02410944 1.0758589 1.1523427 2597.326 1.000184
## bS 0.6898362 0.02490829 0.6500868 0.7300311 2316.962 1.000640
```
Now here's the outcome of the same model where the missing cases are simply dropped:




```
##         mean         sd      5.5%     94.5%    n_eff     Rhat4
## a  1.7946147 0.03356130 1.7408573 1.8485027 1801.153 1.0000524
## bS 0.8276137 0.03382677 0.7750254 0.8813257 1908.130 0.9994402
```

We can see that $bS$ is now closer to the true value of 1. This is because on average homework is missing from noisy houses, and it's usually noisy houses where our estimate of the effect of studying is confounded. In this case the missingness made our inference easier, but in another scenario it could easily make things worse.

In scenario (d) dogs prefer to eat bad homework. But the variable causes its own missingness through the non-causal path $S \rightarrow H \rightarrow D \rightarrow H_{obs}$. This is the most difficult situation to deal with. 

The next section of the chapter applies the above to the problem of imputing missing data in the primate milk example from earlier in the book. Revisit.




## Questions

## Further Reading {-}

Endnote 225: "See Molenberghs et al. (2014) for an overview of contemporary approaches, Bayesian and otherwise"

Endnote 226: "In ecology, the absence ofan observation ofa species is a subtle kind of observation. It could mean the species isn’t there. Or it could mean it is there but you didn’t see it. An entire category of models, occupancy models, exists to take this duality into account".

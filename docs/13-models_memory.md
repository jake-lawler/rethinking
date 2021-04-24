# Models With Memory {#models_memory}




## Chapter Notes

This chapter introduces multi-level models, starting with an example using tadpole mortality data. Each row in the data set is a bucket that starts off with some number of tadpoles, each bucket has it's own experimental conditions (number of tadpoles, presence / absence of predators etc.) and at the end the number of surviving tadpoles are counted. 

The chapter explains that we do not want to assign the same intercept estimate to each of the buckets - we don't want to accidentally mask any variation that may be due to some of our measured variables. But we also don't want to assign each bucket its own independent intercept - learning about one bucket should tell us something about the next. This is the motivation for multi-level models - in particular we start off with a *varying intercepts* model.

Compare the kind of model we would try to fit in previous chapters:

$$
\begin{aligned}
S_i &\sim \text{Binomial}(N_i,p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} \\
\alpha_j &\sim \text{Normal}(0,1.5)
\end{aligned}
$$

With the varying intercepts model:

$$
\begin{aligned}
S_i &\sim \text{Binomial}(N_i,p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} \\
\alpha_j &\sim \text{Normal}(\bar{\alpha},\sigma) \\
\bar{\alpha} & \sim \text{Normal}(0,1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{aligned}
$$

In the first model survival is modelled as binomial, each tank is assigned its own intercept, with each of these intercepts sharing the same fixed prior.

In the multi-level model, the intercept prior is a function of two *hyperpriors*, $\bar{\alpha}$ and $\sigma$. The model updates both "levels" of the model as it sees the data. We fit both models and compare.




```r
compare(m13.1,m13.2)
```

```
##           WAIC       SE    dWAIC     dSE    pWAIC       weight
## m13.2 199.5405 7.271310  0.00000      NA 20.67711 0.9993390343
## m13.1 214.1828 4.845674 14.64229 3.87042 25.34199 0.0006609657
```

The measure of "effective parameters", pWAIC is lower for the multi-level model (m13.2), because of the stronger regularising effect of the hyperpriors.

Here is a plot of the posterior of the multi-level model:

![](13-models_memory_files/figure-epub3/unnamed-chunk-4-1.png)<!-- -->

The blue points are the survival proportions in the raw data, the black circles are the survival proportions estimates by the model. The dashed line is the average survival proportion across all tanks. The survival estimates are pulled towards the mean, and this effect is particularly strong when:

1. a point is far from the mean
2. in small tanks, where there are fewer tadpoles and so the model is more sceptical of the data (in light of the experience of the other tanks).


### More Than One Type of Cluster {-}

The chapter reintroduces the chimpanzee example. We will add clustering according to actor (the chimp pulling levers) and experimental block. Here's the model:

$$
\begin{aligned}
L_i &\sim \text{Binomial}(1,p_i) \\
\text{logit}(p_i) &= \alpha_{\text{ACTOR}[i]} + \gamma_{\text{BLOCK}[i]} + \beta_{\text{TREATMENT}[i]} \\
\beta_j &\sim \text{Normal}(0,0.5) && \text{for } j = 1 \dots 4 \\
\alpha_j &\sim \text{Normal}(\bar{\alpha},\sigma_\alpha) && \text{for } j = 1 \dots 7\\
\gamma_j &\sim \text{Normal}(0,\sigma_\gamma) && \text{for } j = 1 \dots 6\\
\bar{\alpha} & \sim \text{Normal}(0,1.5) \\
\sigma_\alpha &\sim \text{Exponential}(1) \\
\sigma_\gamma &\sim \text{Exponential}(1)
\end{aligned}
$$

What's happening here? The chapter explains:

> "Each cluster gets its own vector of parameters. For actors, the vector is $\alpha$, and it has length 7, because there are 7 chimpanzees in the sample. For blocks, the vector is $\gamma$, and it has length 6, because there are 6 blocks. Each cluster variable needs its own standard deviation parameter that adapts the amount of pooling across units, be they actors or blocks. These are $\sigma_\alpha$ and $\sigma_\gamma$, respectively. Finally, note that there is only one global mean parameter $\bar{\alpha}$. We can’t identify a separate mean for each varying intercept type, because both intercepts are added to the same linear prediction."


We fit the model and plot the posterior:


![](13-models_memory_files/figure-epub3/unnamed-chunk-6-1.png)<!-- -->

We can see that there is much more variation among actors ($\sigma_\alpha$) than among blocks ($\sigma_\gamma$).

### Divergent Transitions and Non-Centered Priors {-}

Divergent transitions occur quite frequently in multi-level models. The chapter introduces two methods of dealing with these:

1. Increasing Stan's target acceptance rate, which results in a smaller step size.
2. Reparameterisation of the model, to use non-centered priors

We start by increasing the target acceptance rate, to 99% compared to the ulam default 95%: 


```r
set.seed(13) 
m13.4b <- ulam( m13.4 , chains=4 , cores=4 , control=list(adapt_delta=0.99), cmdstan = TRUE )
divergent(m13.4b)
```

Creating a non-centered version of the chimp models requires taking $\bar{\alpha}$, $\sigma_\alpha$ and $\sigma_\gamma$ out of the intercepts:

$$
\begin{aligned}
\alpha_j &\sim \text{Normal}(\bar{\alpha},\sigma_\alpha)\\
\gamma_j &\sim \text{Normal}(0,\sigma_\gamma)
\end{aligned}
$$

Here's the non-centered parameterisation of the model:

$$
\begin{aligned}
L_i &\sim \text{Binomial}(1,p_i) \\
\text{logit}(p_i) &= \bar{\alpha} +  z_{\text{ACTOR}[i]} \sigma_\alpha + x_{\text{BLOCK}[i]} \sigma_\gamma + \beta_{\text{TREATMENT}[i]} \\
\beta_j &\sim \text{Normal}(0,0.5) && \text{for } j = 1 \dots 4 \\
z_j &\sim \text{Normal}(0,1)  \\
x_j &\sim \text{Normal}(0,1) \\
\bar{\alpha} & \sim \text{Normal}(0,1.5) \\
\sigma_\alpha &\sim \text{Exponential}(1) \\
\sigma_\gamma &\sim \text{Exponential}(1)
\end{aligned}
$$

The actor and block intercepts have been standardised, and are now transformed in the linear model instead.

We plot the effective number of parameters of the centred and non-centred models against each other:



![](13-models_memory_files/figure-epub3/unnamed-chunk-9-1.png)<!-- -->

Each point is a parameter. Points above the line suggest the non-centered model performed better.



## Questions

### 13E1 {-}

#### Question {-}


#### Answer {-}



## Further Resources {-}



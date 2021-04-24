# Markov Chain Monte Carlo {#markov_chain}




## Chapter Notes


### Simulating King Markov's Journey (A Metropolis Algorithm) {-}


The chapter opens with an implementation of the Metropolis algorithm, through a parable about the king of a ring of ten islands. Each week, the king decides whether to remain on his current island, or move to a neighbouring island. A proposal island is chosen by flipping a coin - either the next island clockwise or anti-clockwise from the current one. Whether the king moves to the proposal island or stays put depends on a random draw, with the probability weighted by the relative population of the island:


```r
num_weeks <- 1e5 

positions <- rep(0,num_weeks) 

current <- 10 

for ( i in 1:num_weeks ) { 
  
  ## record current position 
  
  positions[i] <- current
  
  ## flip coin to generate proposal 
  
  proposal <- current + sample( c(-1,1) , size=1 )
  
  ## now make sure he loops around the archipelago 
  
  if ( proposal < 1 ) proposal <- 10 
  if ( proposal > 10 ) proposal <- 1

  ## move?

  prob_move <- proposal/current 

  current <- ifelse( runif(1) < prob_move , proposal , current )
}

position_data <- tibble(week = 1:100000, position = positions)
```


Overall time, the proportion of time the king spends on each island is in proportion to its population:

![](09-markov_chain_files/figure-epub3/unnamed-chunk-3-1.png)<!-- -->

The chapter also displays the first 100 weeks so you can see the path that the king takes:

![](09-markov_chain_files/figure-epub3/unnamed-chunk-4-1.png)<!-- -->

Revisit: Return to the Overthinking box on page 276: Overthinking: Hamiltonian Monte Carlo in the raw.

The chapter introduces the ulam tool for fitting Hamiltonian Monte Carlo (HMC) models in Stan. We load the ruggedness data from chapter 8 and fit the interaction model, this time using HMC instead of quadratic approximation.

To save computation, we want to pre-process any variable transformations before passing the model to Stan. It's also good practice to remove columns from the data frame if they will not be included in the model.  


```r
data(rugged)
data_rugged <- as_tibble(rugged)

data_rugged <- data_rugged%>%
  mutate(log_gdp = log(rgdppc_2000))%>%
  filter(!is.na(log_gdp))%>%
  mutate(log_gdp_std = log_gdp / mean(log_gdp),
         rugged_std = rugged / max(rugged),
         cid <- if_else(cont_africa==1,1,2),
         cid = factor(cid))%>%
  select(log_gdp_std, rugged_std,cid)
```

The model in chapter 8, fit using quadratic approximation looks like this:


```r
m8.3 <- quap( alist(
log_gdp_std ~ dnorm( mu , sigma ) , 
mu <- a[cid] + b[cid]*( rugged_std - 0.215 ) , 
a[cid] ~ dnorm( 1 , 0.1 ) , 
b[cid] ~ dnorm( 0 , 0.3 ) , 
sigma ~ dexp( 1 )
) , data=data_rugged )

precis( m8.3 , depth=2 )
```

```
##             mean          sd        5.5%       94.5%
## a[1]   0.8865370 0.015675200  0.86148497  0.91158896
## a[2]   1.0505769 0.009936314  1.03469673  1.06645702
## b[1]   0.1323525 0.074202543  0.01376249  0.25094248
## b[2]  -0.1425916 0.054747840 -0.23008920 -0.05509396
## sigma  0.1094909 0.005934863  0.10000586  0.11897597
```

Here is the same model using ulam:




```r
set.seed(100)
m9.1 <- ulam( alist(
log_gdp_std ~ dnorm( mu , sigma ) ,
mu <- a[cid] + b[cid]*( rugged_std - 0.215 ) ,
a[cid] ~ dnorm( 1 , 0.1 ) ,
b[cid] ~ dnorm( 0 , 0.3 ) ,
sigma ~ dexp( 1 )
) , data=data_rugged , chains=4, cores=4, cmdstan = TRUE )
```


```
##             mean          sd         5.5%       94.5%    n_eff     Rhat4
## a[1]   0.8863900 0.016066653  0.861263755  0.91239339 2385.040 0.9989979
## a[2]   1.0507876 0.010359464  1.034099450  1.06736275 3169.658 0.9983588
## b[1]   0.1308372 0.074861084  0.007588926  0.24926313 2501.290 0.9999101
## b[2]  -0.1421440 0.053609582 -0.231558840 -0.05599294 2814.435 0.9984347
## sigma  0.1115744 0.006327084  0.102049240  0.12232890 2873.048 0.9992689
```

We show the traceplot and trankplot for the model fit, in order to contrast with more pathological plots that will be shown in the next section.


```r
traceplot(m9.1)
```

```
## [1] 1000
## [1] 1
## [1] 1000
```

```r
trankplot(m9.1)
```

![](09-markov_chain_files/figure-epub3/unnamed-chunk-10-1.png)<!-- -->![](09-markov_chain_files/figure-epub3/unnamed-chunk-10-2.png)<!-- -->


The chapter includes an example of a model with very flat priors and very little data, in order to demonstrate how you may be able to tell if you're attempt at model fitting has gone wrong somewhere.




```
## [1] 1000
## [1] 1
## [1] 1000
```

![](09-markov_chain_files/figure-epub3/unnamed-chunk-12-1.png)<!-- -->![](09-markov_chain_files/figure-epub3/unnamed-chunk-12-2.png)<!-- -->

The chain's here are not stationary, and they do not converge to the same region of high probability. They are a warning that something has gone wrong.

In this case we can fix the issue by using even slightly informative priors.


Another way that model fitting can go wrong is with non-identifiable parameters. We saw this in the leg length example in chapter 6. 


## Questions

Revisit.

### 9E1 {-}


## Further Reading {-}


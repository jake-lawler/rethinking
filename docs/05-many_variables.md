# The Many Variables & The Spurious Waffles {#many_variables}




## Chapter Notes

I haven't written chapter notes for the first five chapters.

## Questions

### 5E1 {-}

#### Question {-}

>Which of the linear models below are multiple linear regressions? 
>
(1) µi = α + βxi 
>
(2) µi = βx * xi + βz * zi 
>
(3) µi = α + β(xi − zi)
>
(4) µi = α + βx * xi + βz * zi


#### Answer {-}

Number 4 looks the most like the multiple regressions in the chapter: µ is regressed on both xi and zi with the "intercept" α. (2) is just (4) with the α set to zero, so that counts too.

Number 1 is just a bivariate regression.

Number 3 is interesting. I think this is not really a multiple regression, even though there are two variables. Rather than attempting to determine separately the influence of x and z on µ, you are asserting in the model that they have and equal and opposite impact. I think this is not really what you want a multiple regression to do, but don't feel confident about my answer.


### 5E2 {-}

#### Question {-}

>Write down a multiple regression to evaluate the claim: 
>
Animal diversity is linearly related to latitude, but only after controlling for plant diversity. You just need to write down the model definition.


#### Answer {-}


Oh boy. This question immediately feels like a trap with "animal diversity is *linearly* related to latitude." Surely if I choose to use a multiple linear regression with two variables, and control for one, the only relationships I'll observe will be linear.

I'm going to ignore the "linearly" part of the question from this point. It seems like the claim is that if I naively regress animal diversity on to latitude without accounting for plant diversity, I would find no relationship. I.e. that the relationship between latitude and animal diversity is masked.

If that interpretation is correct, I would start with a bivariate model.


$$
A_i \sim Normal(\mu, \sigma) \\

\mu_i = \alpha + \beta_L*L
$$

Where if the claim is true I would expect to see little relationship. I would then move on to a multiple regression including plant diversity:

$$
A_i \sim Normal(\mu, \sigma) \\

\mu_i = \alpha + \beta_L*L +\beta_P * P
$$

and examine whether it appears as if a relationship has now emerged.


### 5E3 {-}

#### Question {-}

>Write down a multiple regression to evaluate the claim: Neither amount of funding nor size of laboratory is by itself a good predictor of time to PhD degree; but together these variables are both positively associated with time to degree. 
>
Write down the model definition and indicate which side of zero each slope parameter should be on.


#### Answer {-}


$$
T_i \sim Normal(\mu, \sigma) \\

\mu_i = \alpha + \beta_F*F +\beta_S * S
$$

T - time to PhD degree
F - amount of funding
S - size of laboratory


### 5E4 {-}

#### Question {-}


>Suppose you have a single categorical predictor with 4 levels (unique values), labeled A, B, C and D. Let $A_i$ be an indicator variable that is 1 where case i is in category A. Also suppose $B_i$, $C_i$, and $D_i$ for the other categories. 
Now which of the following linear models are inferentially equivalent ways to include the categorical variable in a regression? Models are inferentially equivalent when it’s possible to compute one posterior distribution from the posterior distribution of another model.
>
$$
(1) \mu_i = \alpha + \beta_A A_i + \beta_B B_i + \beta_D D_i  \\
(2) \mu_i = \alpha + \beta_A A_i + \beta_B B_i + \beta_C C_i + \beta_D D_i \\ 
(3) \mu_i = \alpha + \beta_B B_i + \beta_C C_i + \beta_D D_i \\
(4) \mu_i = \alpha_A A_i + α_B B_i + α_C C_i + α_D D_i \\
(5) \mu_i = \alpha_A(1 − B_i − C_i −D_i) + \alpha_B B_i + \alpha_C C_i + \alpha_D D_i
$$

#### Answer {-}

1 is the standard indicator variable approach. Where A, B and D are equal to 0, $\alpha$ is the mean $\mu$ where the predictor is at level C.
2 There is redundancy in two, surely it wouldn't be possible to estimate $\alpha$ - it can take any value and produce the same $\mu$ so long as the appropriate $\beta_x$ adjusts to compensate. I don't know if that means it is not inferentially equivalent though.
3 is clearly equivalent to (1), it doesn't make a difference (except for interpretation) which of the levels you label null.
4 Is equivalent also, just set $\alpha_c$ equal to $\alpha$ from (1).
5 Is an incredibly annoying way to set up your model, but can be pretty easily transformed into (3) with some algebra and relabelling:

$$
\begin{aligned}
\mu_i &= \alpha_A(1 − B_i − C_i −D_i) + \alpha_B B_i + \alpha_C C_i + \alpha_D D_i \\
&= \alpha_A + (\alpha_B-\alpha_A)B_i + (\alpha_C -\alpha_A)C_i +(\alpha_D-\alpha_A)D_i  \\
&= \alpha + \beta_B B_i + \beta_C C_i + \beta_D D_i
\end{aligned}          
$$

## Further Reading {-}

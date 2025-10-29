---
layout: page
title: "Statistical Methods"
---

## Likelihood Building

A likelihood function, parameterised by the model parameters $\vec{\Phi}$, is constructed for a particular data set of independent identically distributed observables $\vec{x}_{d}$ as

$$
  \mathcal{L}(\vec{\Phi}) = \prod_{d}p(\vec{x}_{d};\vec{\theta},\vec{\nu})\prod_{k}p_{k}(y_{k};\nu_{k}),
$$

where $p(\vec{x};\vec{\theta},\vec{\nu})$ is the probability distribution of the observables for the primary analysis, and $p_{k}(\vec{\theta},\vec{\nu})$ are the probability distributions of the auxiliary observables. 

The probability density terms associated with the auxiliary observables, $p_{k}(y_{k};\nu_{k})$, can also be reinterpreted as posterior distributions for the nuisance parameters, $p_{k}(\nu_{k}\|y_{k})$, resulting from the outcome of measurements of, or otherwise justified constraints on, the auxiliary observables $y_{k}$, through the relationship

$$
  p_{k}(\nu_{k}|y_{k}) \propto p_{k}(y_{k};\nu_{k})\pi_{k}(\nu_{k}),
$$

where $\pi_{k}(\nu_{k})$ are the nuisance parameter priors. 

These priors typically take the form of the Gaussian distribution

$$
  \mathcal{N}(y;\nu,\sigma_{\nu}) = \frac{1}{\sigma_{\nu}\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{\nu-y}{\sigma_{\nu}}\right)^{2}},
$$

where $y$ is the measured value of the auxiliary observable, $\nu$ is the expected value of the nuisance parameter, and $\sigma_{\nu}$ is the uncertainty on that expectation.


Each element of $\vec{x}$ are statistically independent from all other elements of $\vec{x}$. 
For example, each element of $\vec{x}$ could be the event counts in different reconstructed final states of some data set (binned), or a continuous observable such as the invariant mass of a pair of final state particles (unbinned). 


In the case of the binned likelihood, $p(\vec{x}_{d};\vec{\theta},\vec{\nu})$ for each bin $d$ is given by a Poisson distribution,

For unbinned likelihoods, the exact PDF, $p(\vec{x}_{d};\vec{\theta},\vec{\nu})$ for each event $d$, is used.
ß

$$
  \mathcal{P}(n;\lambda) =  \lambda^{n}\frac{e^{-\lambda}}{n!},
$$

which models the number of observed events in that bin as a Poisson-distributed random variable with an expected value $\lambda_{d}(\vec{\theta},\vec{\nu})$.


For unbinned likelihoods, the exact PDF, $p(\vec{x}_{d};\vec{\theta},\vec{\nu})$ for each event $d$, is used.
This describes the probability density of the continuous observable $\vec{x}_{d}$ given the parameters $\vec{\theta}$ and nuisance parameters $\vec{\nu}$.
In the case where the total number of observed events is itself a Poisson-distributed observable, the unbinned likelihood can be extended to include a term that accounts for the total expected yield. 
This extended likelihood formulation modifies the standard likelihood by incorporating a Poisson factor for the total number of observed events $n$, given the expected number $\lambda(\vec{\mu}, \vec{\theta})$

$$
\mathcal{L}_{\text{ext}}(\vec{\Phi}) = \mathcal{P}(n;\lambda) \prod_{d}p(\vec{x}_{d};\vec{\theta},\vec{\nu})\prod_{k}p_{k}(y_{k};\nu_{k}).
$$

There is also the possibility of using a hybrid approach, where the likelihood is binned in some observables and unbinned in others.
This is particularly common when using different categories for a different set of unbinned observables. 
In this example, the categorised extended unbinned likelihood takes the form

$$
\mathcal{L}_{\text{hybrid}}(\vec{\Phi}) = \prod_{c}\left( \mathcal{P}(n_c;\lambda_c) \prod_{d}p_{c}(\vec{x}_{c,d};\vec{\theta},\vec{\nu})\right)\prod_{k}p_{k}(y_{k};\nu_{k}),
$$

where the product over $d$ is only over the unbinned observables, and the categories are accounted for with the index $c$ and its equivalent subscript.


It is common to split the contribution of $p(\vec{x}_{d};\vec{\theta},\vec{\nu})$ up into many contributing processes $p$.
This allows for additional primary observables $\mu_p$, that scale the rate of individual processes.
In the binned case, this amounts to a change in the expected yield for that bin, $\lambda_{d}(\vec{\theta}, \vec{\nu})$, to

$$
  \lambda_{d}(\vec{\theta}, \vec{\nu}) = \sum_{p}\mu_{p}(\vec{\theta}, \vec{\nu})\lambda_{d,p}(\vec{\theta}, \vec{\nu}),
$$

where $\lambda_{d,p}(\vec{\theta}, \vec{\nu})$ is the expected yield for bin $d$ from process $p$.

In the unbinned case, the PDF $p(\vec{x}_{d};\vec{\theta},\vec{\nu})$ is modified to

$$
  p(\vec{x}_{d};\vec{\theta},\vec{\nu}) = \sum_{p}\mu_{p}\lambda_{p}(\vec{\theta}, \vec{\nu})p_{p}(\vec{x}_{d};\vec{\theta},\vec{\nu}) / \lambda(\vec{\theta}, \vec{\nu}),
$$

where $\lambda_{p}(\vec{\theta}, \vec{\nu})$ is the expected yield of process $p$, and $p_{p}(\vec{x}_{d};\vec{\theta},\vec{\nu})$ is the PDF of process $p$.
The expected yield $\lambda(\vec{\theta}, \vec{\nu})$ is calculated by

$$
  \lambda(\vec{\theta}, \vec{\nu}) = \sum_{p}\mu_{p}\lambda_{p}(\vec{\theta}, \vec{\nu}).
$$

## Interpolation of yields

It is common practice to factorise the contributions to the expected yields arising from the parameters of interest and those from each individual nuisance parameter. 
This factorisation can be expressed as

$$
  \lambda_{p}(\vec{\theta}, \vec{\nu}) = \lambda_{p}^{\theta}(\vec{\theta}) \sum_{k} \lambda_{p}^{k}(\nu_{k}),
$$

where $\lambda_{p}^{\theta}(\vec{\theta})$ encodes the dependence on the parameters of interest, and each $\lambda_{p}^{k}(\nu_{k})$ represents the modification due to a single nuisance parameter $\nu_k$. 
The term $\lambda_{p}^{\theta}(\vec{\theta})$ is typically known analytically and can be directly specified. 
In contrast, the contributions from the nuisance parameters, $\lambda_{p}^{k}(\nu_{k})$, are generally not available in closed analytic form, but can be estimated from simulation. 
However, evaluating the likelihood by summing over the entire simulation for each likelihood call is computationally prohibitive. 
To address this, the expected yields are interpolated using an asymmetric log-normal model.

This interpolation relies on providing the expected yields at three values of the nuisance parameters: the nominal value ($\nu_{k}=0$) $\sum_{b} \omega_{b}^{0}$, and the values corresponding to positive and negative shifts, $\sum_{b} \omega_{b}^{\pm}$. 
These shifted yields correspond to a deviation of $2q$ in the nuisance parameter, where $q$ is typically set to 0.5.
The interpolated contribution of the nuisance parameter $\nu_k$ to the expected yield is then given by

$$
  \lambda_{p}^{k}(\nu_k) = \exp\left[ F(\nu_k, \ln \kappa^{+}, \ln \kappa^{-}) \right],
$$

with the asymmetry factors defined as

$$
  \kappa^{\pm} = \frac{\sum_{b} \omega_{b}^{\pm}}{\sum_{b} \omega_{b}^{0}}.
$$

The function $F(\nu, \kappa^{+}, \kappa^{-})$ is defined piecewise as

$$
  F(\nu, \delta^{+}, \delta^{-}) =
    \begin{cases}
      \frac{1}{2} \nu \left[ (\delta^{+} - \delta^{-}) + \frac{1}{8} (\delta^{+} + \delta^{-}) J(\bar{\nu}) \right], & \text{if } \|\nu\| < q; \\
      \nu \delta^{+}, & \text{if } \nu \ge q; \\
      -\nu \delta^{-}, & \text{if } \nu \le -q,
    \end{cases}
$$

where $\bar{\nu} = \nu / q$, and the smooth transition function is given by:

$$
  J(\bar{\nu}) = 3\bar{\nu}^5 - 10\bar{\nu}^3 + 15\bar{\nu}.
$$

This formulation ensures that the interpolated yields, along with their first and second derivatives, are continuous with respect to $\nu$ over the entire domain.


{% include mathjax.html %}
---
layout: page
title: "Introduction"
permalink: /
---

**INNFER** (Invertible Neural Networks for Extracting Results) is a framework for performing simulation-based inference in a frequentist statistical setting.  
It enables multidimensional, unbinned likelihood fits by leveraging machine learning techniques to model probability densities.  

At its core, INNFER uses artificial intelligence to learn the probability density function:

$$
p(\vec{x} \mid \vec{\theta}, \vec{\nu})
$$

where:
- $\vec{x}$ = reconstructed variables  
- $\vec{\theta}$ = parameters of interests 
- $\vec{\nu}$ = nuisance parameters  

This approach provides an optimal and statistically rigorous way to perform high-dimensional data analyses while maintaining the interpretability of classical frequentist methods.

The framework is developed at Imperial College London for initial use performing statistical analysis on data collected by the CMS experiment. If you have any questions about the repository contact George Uttley at george.peter.uttley@cern.ch.

Contributors: George Uttley, Nicholas Wardle, Ye He

---

Next: [Setup Guide](setup.md).
# Bayesian Neural Networks Reading List

This is a reading list on Bayesian neural networks. The list is quite opinionated in focusing mainly on the case where priors and posteriors are defined over the neural network weights, and the predictive task is classification. In this setting, Bayesian neural networks might be preferable because they either improve test accuracy, test calibration or both. I first include a list of essential papers, and then organize papers by subject. The aim is to create a guide for new researchers in Bayesian Deep Learning, that will speed up their entry to the field. 

Interested in a more detailed discussion? Check out our recent review paper 🚨["A Primer on Bayesian Neural Networks: Review and Debates"](https://arxiv.org/abs/2309.16314) joint work with [Julyan Arbel (Inria Grenoble Rhône-Alpes)](https://www.julyanarbel.com/), [Mariia Vladimirova (Criteo)](https://sites.google.com/site/mrvladimirova/home), [Vincent Fortuin (Helmholtz AI)](https://fortuin.github.io/)🚨.

## ⚠️ Essential reads



- [[Weight Uncertainty in Neural Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiEqOnA2cv8AhWHTKQEHeATB8sQFnoECAoQAQ&url=http%3A%2F%2Fproceedings.mlr.press%2Fv37%2Fblundell15.pdf&usg=AOvVaw1XvLXExIhW1Sad_feY49ss)]: A main challenge in Bayesian 
neural networks is how to obtain gradients for parametric distributions such as the Gaussian. This is one of the first papers that discusses Variational Inference using the reparametrization trick for realistic neural networks. The reparametrization trick allows obtaining gradients using Monte Carlo sampling from the posterior.

- [[Laplace Redux -- Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)]: The Laplace approximation is one of the few realistic options to perform approximate inference for Bayesian Neural Networks.
Not only does it result in good uncertainty estimates, but it can also be used for model selection and invariance learning.

- [[How Good is the Bayes Posterior in Deep Neural Networks Really?](http://proceedings.mlr.press/v119/wenzel20a/wenzel20a.pdf)]: This paper describes a major criticism of Bayesian Deep Learning, that for the metrics of accuracy and negative log-likelihood, 
a deterministic network is often better than a Bayesian one. At the same time it describes two common tricks for efficient MCMC approximate inference, [Preconditioned Stochastic Gradient Langevin Dynamics](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11835/11805) and [Cyclical Step Sizing](https://arxiv.org/abs/1902.03932).

- [[What Are Bayesian Neural Network Posteriors Really Like?](http://proceedings.mlr.press/v139/izmailov21a/izmailov21a.pdf)]: This paper implements Hamiltonian Monte Carlo (HMC) for approximate inference 
in Bayesian Deep Neural Networks. HMC is considered the gold standard in approximate inference, however it is very computationally intensive.

- [[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj0jMP75sv8AhXWT6QEHeaLBxkQFnoECCkQAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F2017%2Ffile%2F9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf&usg=AOvVaw1zcxDvvpYRZlrPzKo7zzZO)]: This paper implements deep neural network ensembles. 
This is a Frequentist alternative to Bayesian Neural Networks. It is one of the most common baselines for Bayesian Neural Networks, and frequently outperforms them.

- [[The Bayesian Learning Rule](https://arxiv.org/abs/2107.04562)]: Many machine-learning algorithms are specific instances of a single algorithm called the Bayesian learning rule. The rule, derived from Bayesian principles, yields a wide-range of algorithms from fields such as optimization, deep learning, and graphical models. 

## 🔥 Interesting recent reads!
- [SAM as an Optimal Relaxation of Bayes](https://openreview.net/pdf?id=k4fevFqSQcX)
- [Robustness to corruption in pre-trained Bayesian neural networks](https://openreview.net/pdf?id=kUI41mY8bHl)
- [Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift](https://arxiv.org/pdf/2306.12306)
- [Collapsed Inference for Bayesian Deep Learning](https://arxiv.org/pdf/2306.09686.pdf)
- [The Memory Perturbation Equation: Understanding Model's Sensitivity to Data](https://arxiv.org/pdf/2310.19273)
- [Learning Layer-wise Equivariances Automatically using Gradients](https://arxiv.org/pdf/2310.06131)

## Approximate Inference
The Bayesian paradigm consists of choosing a prior over the model parameters, evaluating the data likelihood and the estimating the posterior over the model parameters. This can be done analytically only in simple cases (Gaussian likelihood, prior and posterior). For more complex and interesting cases we have to resort to approximate inference.

### Variational Inference 
++ Computationally efficient --Explores a single mode of the loss

- [Keeping the neural networks simple by minimizing the
description length of the weights.](https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/pdf/10.1145/168304.168306&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=10176669309393884854&ei=5h_FY9ffEYaymgHM6YaoBw&scisig=AAGBfm26p-N1UN5egv0YIQzRamLrWqemLw)
- [Ensemble learning in Bayesian neural networks.](https://scholar.google.com/scholar_url?url=https://seunghan96.github.io/assets/pdf/BNN/paper/05.Ensemble%2520Learning%2520in%2520Bayesian%2520Neural%2520Networks.pdf&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=8533198369198161660&ei=mCDFY8SzGeyVy9YP4KGr0Ao&scisig=AAGBfm0FmcAORkwGf9QYkeIvoo43mrwT7g)
- [Practical variational inference for neural networks.](https://scholar.google.com/scholar_url?url=https://proceedings.neurips.cc/paper/4329-practical-variational-inference-for-neural-networks&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=16673382953830986184&ei=ySDFY87wO42Sy9YP7pSx8AU&scisig=AAGBfm0ZKmVC22U1ZNQl0DJ8TB4yhTtdHg)
- [Auto-encoding variational Bayes.](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/1312.6114&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=10486756931164834716&ei=7yDFY6O5CoaymgHM6YaoBw&scisig=AAGBfm21twq9Fdq-mroZkKJFO98cQ8uMwA)
- [Weight uncertainty in neural
networks.](https://scholar.google.com/scholar_url?url=https://proceedings.mlr.press/v37/blundell15.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=6370453062994389837&ei=fCHFY9XHGYaymgHM6YaoBw&scisig=AAGBfm0f-bYr_T9kRXccr37odGA0gBNmxw)
- [Fast and scalable
Bayesian deep learning by weight-perturbation in ADAM.](https://scholar.google.com/scholar_url?url=https://proceedings.mlr.press/v80/khan18a.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=11374390410783252644&ei=qyHFY_eLJuyVy9YP4KGr0Ao&scisig=AAGBfm3jb7CoJkqSGjjq_YU8zaXfLPpwXw)
- [Vprop: Variational inference using rmsprop.](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/1712.01038&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=14528683780372901925&ei=1CHFY9WpD7nGsQL-6pDoCg&scisig=AAGBfm3ed7938rmWVJBD0tmlo7ApYYvcCg)
- [Structured and efficient variational deep learning with matrix
Gaussian posteriors](https://scholar.google.com/scholar_url?url=https://proceedings.mlr.press/v48/louizos16.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=6832261610013648671&ei=BCLFY46pGeyVy9YP4KGr0Ao&scisig=AAGBfm3Dk70i6rAzP1RCkkKazyU6CfPj8g)
- [Slang: Fast structured
covariance approximations for bayesian deep learning with natural gradient.](https://scholar.google.com/scholar_url?url=https://proceedings.neurips.cc/paper/2018/hash/d3157f2f0212a80a5d042c127522a2d5-Abstract.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=16145055537497825367&ei=MSLFY5r2OYbcmwGmsojQCQ&scisig=AAGBfm0J9CD0VPhaaxHcXvMeZ9OEBGJ-MA)
- [Structured second-order methods via natural-gradient descent.](https://arxiv.org/pdf/2107.10884)
### Laplace approximation
++ Computationally efficient --Explores a single mode of the loss

- [A practical Bayesian framework for backpropagation networks.](https://scholar.google.com/scholar_url?url=https://direct.mit.edu/neco/article/4/3/448/5654&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=4883107376501501717&ei=FyPFY5bkLJOSy9YPxYCN6AI&scisig=AAGBfm2uQ4xCZIUjhFInqshLOyN4qk6WcQ)
- [A scalable Laplace approximation for neural networks.](https://scholar.google.com/scholar_url?url=https://discovery.ucl.ac.uk/id/eprint/10080902/&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=3068639073703398000&ei=OSPFY_bIDuOSy9YP2ZKSqA8&scisig=AAGBfm2rxtCiHqTNTtypVHoU1PFVqUK5Tg)
- [Laplace Redux -- Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)
- [Improving predictions of bayesian neural nets via local
linearization. ](https://scholar.google.com/scholar_url?url=https://proceedings.mlr.press/v130/immer21a.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=7363952428677779165&ei=hiPFY-6WJOOSy9YP2ZKSqA8&scisig=AAGBfm0WKGryBtL6Te5RTwNBZpYPZq5mCQ)
- [Adapting the linearised laplace model evidence for modern deep learning. ](https://scholar.google.com/scholar_url?url=https://proceedings.mlr.press/v162/antoran22a.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=3588379385157935270&ei=pCPFY7T8M6PGsQKV5YqwCg&scisig=AAGBfm2I5Sto1dfSHydrDzjtQxf4_eorNw)
- [ Bayesian Deep Learning via Subnetwork Inference ](https://arxiv.org/pdf/2010.14689)
### Sampling methods
++ Explore multiple modes --Computationally expensive

- [What Are Bayesian Neural Network Posteriors Really Like?](http://proceedings.mlr.press/v139/izmailov21a/izmailov21a.pdf)
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj0jMP75sv8AhXWT6QEHeaLBxkQFnoECCkQAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F2017%2Ffile%2F9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf&usg=AOvVaw1zcxDvvpYRZlrPzKo7zzZO)
- [Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks.](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11835/11805)
- [Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning.](https://arxiv.org/abs/1902.03932)
- [Bayesian learning via stochastic gradient Langevin dynamics.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiB_Pqj78v8AhWQVKQEHXspC9YQFnoECAsQAQ&url=https%3A%2F%2Fwww.stats.ox.ac.uk%2F~teh%2Fresearch%2Fcompstats%2FWelTeh2011a.pdf&usg=AOvVaw2Eomq_YCJE9-E1E8x33fsM)
- [A complete recipe for stochastic gradient MCMC.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjn_Lrh78v8AhXhVKQEHZGoBDsQFnoECAsQAQ&url=http%3A%2F%2Fpapers.neurips.cc%2Fpaper%2F5891-a-complete-recipe-for-stochastic-gradient-mcmc.pdf&usg=AOvVaw1_xxogcCifNSVbrKX3W283)

### Deep Ensembles
++ Explore multiple modes, Computationally competitive --Memory expensive

- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/pdf/1612.01474.pdf)
- [Multi-Symmetry Ensembles: Improving Diversity and Generalization via Opposing Symmetries](https://arxiv.org/pdf/2303.02484.pdf)
- [Hyperparameter Ensembles](https://arxiv.org/pdf/2006.13570.pdf)
- [Agree to Disagree: Diversity through Disagreement for Better Transferability](https://openreview.net/pdf?id=K7CbYQbyYhY)
- [Feature Space Particle Inference for Neural Network Ensembles](https://proceedings.mlr.press/v162/yashima22a/yashima22a.pdf)
- [Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/pdf/1912.02757.pdf)
- [DICE: Diversity in Deep Ensembles via Conditional Redundancy Adversarial Estimation](https://arxiv.org/pdf/2101.05544)
- [Deep Ensembles Work, But Are They Necessary?](https://arxiv.org/pdf/2202.06985.pdf)
- [Combining Diverse Feature Priors](https://arxiv.org/pdf/2110.08220.pdf)

## Performance Certificates
Bayesian neural networks are compatible with two approaches to model selection that eschew validation and test sets and can (in principle) give guarantees on out-of-sample
performance simply by using the training set.

### Marginal Likelihood
The marginal likelihood is a purely Bayesian approach to model selection.

- [Scalable marginal
likelihood estimation for model selection in deep learning](https://arxiv.org/abs/2104.04975)
- [Bayesian model selection,
the marginal likelihood, and generalization.](https://arxiv.org/abs/2202.11678)
- [A bayesian perspective on training
speed and model selection.](https://arxiv.org/abs/2010.14499)
- [Speedy performance
estimation for neural architecture search.](https://arxiv.org/abs/2006.04492)
### PAC-Bayes
PAC-Bayes bounds give high-probability Frequentist guarantees on out-of-sample performance.

- [Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data.](https://arxiv.org/abs/1703.11008)
- [On the role of data in PAC-Bayes bounds.](https://arxiv.org/abs/2006.10929)
- [PAC-Bayesian theory meets Bayesian inference.](https://arxiv.org/abs/1605.08636)
- [PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization.](https://arxiv.org/abs/2211.13609)
- [Non-Vacuous Generalization Bounds at the ImageNet Scale: A PAC-Bayesian Compression Approach.](https://arxiv.org/abs/1804.05862)
- [Sharpness-Aware Minimization for Efficiently Improving Generalization.](https://arxiv.org/abs/2010.01412)

## Benchmarking
Bayesian Neural Networks aim to improve test accuracy and more commonly *test calibration*. While Bayesian Neural Networks are often
tested on standard computer vision datasets such as CIFAR10/100 with test accuracy as the test metric, dedicated datasets and metrics can also be used.

### Datasets
- [Benchmarking bayesian deep learning on diabetic retinopathy detection tasks.](https://arxiv.org/abs/2211.12717)
- [Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift.](https://arxiv.org/abs/1906.02530)
- [Uncertainty Baselines: Benchmarks for uncertainty & robustness in deep learning.](https://arxiv.org/abs/2106.04015)
- [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://wilds.stanford.edu/)
### Metrics
- [Expected Calibration Error (ECE).](https://www.jstor.org/stable/2987588)
- [Thresholded Adaptive Calibration Error (TACE).](https://arxiv.org/abs/1904.01685)
- [Pitfalls of in-domain uncertainty
estimation and ensembling in deep learning. ](https://arxiv.org/abs/2002.06470)
- [A New Vector Partition of the Probability Score (Brier score and its decomposition)](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml#container-56956-item-56959)

## Review papers
- [Hands-on Bayesian Neural Networks--a Tutorial for Deep Learning Users.](https://arxiv.org/abs/2007.06823)
- [A review of uncertainty quantification in deep learning: Techniques, applications and challenges.](https://arxiv.org/abs/2011.06225)
- [Bayesian neural networks: An introduction and survey](https://arxiv.org/abs/2006.12024)

<h2> :memo: Citation </h2>

Did you find this reading list helpful? Consider citing our review paper on your scientific publications using the following **BibTeX** citation:

```bibtex
@article{arbel2023primer,
  title={A Primer on Bayesian Neural Networks: Review and Debates},
  author={Arbel, Julyan and Pitas, Konstantinos and Vladimirova, Mariia and Fortuin, Vincent},
  journal={arXiv preprint arXiv:2309.16314},
  year={2023}
}
```

When citing this repository on any other medium, please use the following citation:

```
A Primer on Bayesian Neural Networks: Review and Debates by Julyan Arbel, Konstantinos Pitas, Mariia Vladimirova and Vincent Fortuin
```

# Bayesian Neural Networks Reading List

This is a reading list for Bayesian neural networks. While Bayesian Neural Networks are used for a variety of tasks, and in a variety of contexts, this reading list focuses on the task of uncertainty estimation. I first include a list of essential papers, and then organize papers by subject. The aim is to create a guide for new researchers in Bayesian Deep Learning, that will speed up their entry to the field.

## Essential reads



- [[Weight Uncertainty in Neural Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiEqOnA2cv8AhWHTKQEHeATB8sQFnoECAoQAQ&url=http%3A%2F%2Fproceedings.mlr.press%2Fv37%2Fblundell15.pdf&usg=AOvVaw1XvLXExIhW1Sad_feY49ss)]: A main challenge in Bayesian 
neural networks is how to obtain gradients for parametric distributions such as the Gaussian. This is one of the first papers that discusses Variational Inference using the reparametrization trick for realistic neural networks. The reparametrization trick allows obtaining gradients using Monte Carlo sampling from the posterior.

- [[Laplace Redux -- Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)]: The Laplace approximation is one of the few realistic options to perform approximate inference for Bayesian Neural Networks.
Not only does it result in good uncertainty estimates, but it can also be used for model selection and invariance learning.

- [[How Good is the Bayes Posterior in Deep Neural Networks Really?](http://proceedings.mlr.press/v119/wenzel20a/wenzel20a.pdf)]: This paper describes a major criticism of Bayesian Deep Learning, that for the metrics of accuracy and negative log-likelihood, 
a deterministic network is often better than a Bayesian one. At the same time it describes two common tricks for efficient MCMC approximate inference, [Preconditioned Stochastic Gradient Langevin Dynamics](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11835/11805) and [Cyclical Step Sizing](https://arxiv.org/abs/1902.03932).

- [[What Are Bayesian Neural Network Posteriors Really Like?](http://proceedings.mlr.press/v139/izmailov21a/izmailov21a.pdf)]: This paper implements Hamiltonian Monte Carlo (HMC) for approximate inference 
in Bayesian Deep Neural Networks. HMC is considered the gold standard in approximate inference, however it is very computationally intensive.

- [[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj0jMP75sv8AhXWT6QEHeaLBxkQFnoECCkQAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F2017%2Ffile%2F9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf&usg=AOvVaw1zcxDvvpYRZlrPzKo7zzZO)]: This paper implements deep neural network ensembles. 
This is a Frequentist alternative to Bayesian Neural Networks. It is one of the most common baselines for Bayesian Neural Networks.

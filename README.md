# EGO4D_VQ2D

This repository contains the Facebook AI Research's (FAIR) EGO4D 2023 Visual Query 2D Localization challenge submission for the team "Hakuna Matata". The pipeline relies on a Bayesian approach and uses the original Siamese Head complemented with the BEiT transformer. The repository has been primarily built on the original repository and the steps for getting started can be found [here](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D).

The steps for execution or testing for a video are similar to the original baseline, however, we have provided a single consolidated bash file for 
1) downloading most of the dependencies
2) randomly sampling from the train, val and test set
3) testing on the sampled videos

Our submitted report can be viewed online [here](https://arxiv.org/abs/2305.17611)

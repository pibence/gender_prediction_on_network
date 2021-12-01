# Demography estimation
## Goal of the project 
This project uses Dong et al.(2014): Inferring User Demographics and Social Strategies in Mobile Social Networks article and recreates the plots presented in the article by using the Slovakian facebook's data.
source: https://snap.stanford.edu/data/soc-pokec.html
source of the article https://ericdongyx.github.io/papers/KDD14-Dong-et-al-WhoAmI-demographic-prediction.pdf

The project also compares prediction methods for genders of the nodes in the network. The following methods are presented:
* based on the neighbors of a node, the predicted value is the gender that is represented in a higher proportion - this model is used as benchmark.
* Using the triads the users are in a logistic regression and decision tree are being used to calculate how gender composition of friendship triangles determine the gender of the given node. These methods are also used to predict the gender on the test set.

## Results
Both triad-based models outperform the benchmark with similar results 
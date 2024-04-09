
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

# pycv: Python Cross Validation Library

The Python Cross Validation Library (`pycv`) assembles a set of cross validation methods to mitigate dataset shift.

Dataset shift corresponds to a scenario where the training and test sets have different distributions and encompass several representations (i.e., covariate shift, prior probability
shift, concept shift, internal covariate shift). An example of dataset shift (namely covariate shift) is depicted in Figure 1, where one class concentrates low feature values in training
and high feature values in testing.

![alt text](https://github.com/DiogoApostolo/pyCV/blob/main/images/datasetShift.png?raw=true)

This library currently includes 4 Cross Validation Algorithms: SCV, DBSCV, DOBSCV and MSSCV

#### SCV
SCV is an improvement over the basic CV which guarantees that the training and testing sets have the same percentages of samples per class as in the original dataset so that the prior probability shift is avoided, however this algorithm does not actively mitigate covariate shift.


#### DBSCV
Introduced in [1](https://doi.org/10.1080/095281300146272), DBSCV is a CV variant for addressing covariate shift. This method attempts to separate the data into folds by attributing to each fold a similar observation to the one attributed to a previous fold (Figure 2). In such a way, the distribution of each fold will be more similar when compared to SCV.

![alt text](https://github.com/DiogoApostolo/pyCV/blob/main/images/DBSCV_example.png?raw=true "Employee Data title")
*Figure 1: Example of DBSCV for two folds: for each class (blue and red), a starting sample (0 and 1) is chosen and assigned to the first fold, then the closest examples (2 and 3) are chosen and assigned to the next fold. This process is repeated until there are no samples left.*

#### DOBSCV
DOBSCV is an optimized version of DBSCV  proposed in [2](https://pubmed.ncbi.nlm.nih.gov/24807526/). While both algorithms are similar in their goal to reduce covariate shift by distributing samples of the same class as evenly as possible between folds, DOBSCV is less sensitive to random choices since, after assigning a sample to each of the $k$ folds, it picks a new random sample to restart the process (Figure 3). 

![alt text](https://github.com/DiogoApostolo/pyCV/blob/main/images/DOBSCV_example.png?raw=true)


#### MSSCV
MSSCV can be considered a baseline [2](https://pubmed.ncbi.nlm.nih.gov/24807526/), corresponding to the opposite version of DBSCV. Instead of assigning the closest sample to the next fold, it assigns the most distant (Figure 4). Each fold will be as different as possible, which may cause an increase in covariate shift but also provide more variability of samples.

![alt text](https://github.com/DiogoApostolo/pyCV/blob/main/images/MSSCV_example.png?raw=true)

## Usage Example:

The `dataset` folder contains some datasets with binary and multi-class problems. All datasets are numerical and have no missing values. The `complexity.py` module implements the complexity measures.
To run the measures, the `Complexity` class is instantiated and the results may be obtained as follows:

```python
from pycol_complexity import complexity
complexity = complexity.Complexity("dataset/61_iris.arff",distance_func="default",file_type="arff")

# Feature Overlap
print(complexity.F1())
print(complexity.F1v())
print(complexity.F2())
# (...)

# Instance Overlap
print(complexity.R_value())
print(complexity.deg_overlap())
print(complexity.CM())
# (...)

# Structural Overlap
print(complexity.N1())
print(complexity.T1())
print(complexity.Clust())
# (...)

# Multiresolution Overlap
print(complexity.MRCA())
print(complexity.C1())
print(complexity.purity())
# (...)
```

## Developer notes:
To submit bugs and feature requests, report at [project issues](https://github.com/DiogoApostolo/pyCV/issues).

## Licence:
The project is licensed under the MIT License - see the [License](https://github.com/DiogoApostolo/pycol/blob/main/LICENCE) file for details.



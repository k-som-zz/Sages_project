# Breast cancer Wisconsin diagnostic data set - via Kaggle:

https://www.kaggle.com/uciml/breast-cancer-wisconsin-data



#### Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34]. 


Attribute Information:

=>  Diagnosis (M = malignant, B = benign))

Ten real-valued features are computed for each cell nucleus:

* radius (mean of distances from center to points on the perimeter) 
* texture (standard deviation of gray-scale values) 
* perimeter 
* area 
* smoothness (local variation in radius lengths) 
* compactness (perimeter^2 / area - 1.0) 
* concavity (severity of concave portions of the contour) 
* concave points (number of concave portions of the contour) 
* symmetry 
* fractal dimension ("coastline approximation" - 1)


#### Limitations: super small data set!


Table of Contents:
1. Data set preparation

    1.1 Read the data set
    1.2 Remove ID column
    1.3 Binary 'diagnosis' input
    1.4 Missing values
    1.5 Class distribution

2. Testing algorithms

    2.1 Withouth dimensionality reduction

        2.1.1 Comparison

    2.2 Correlation Coefficient Score

        2.2.1 Comparison

    2.3 Voting classifier

        2.3.1 Voting hard 2.3.2 Voting soft

            2.3.3 Comparison

    2.4 Linear SVC + SelectFromModel

        2.4.1 Comparison

    2.5 Linear SVC + RFECV

        2.5.1 Comparison

    2.6 Tree-based feature classifier

        2.6.1 Comparison



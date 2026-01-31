![](asserts/logo-deepdms.png)

  <h1 align="center">Deep Learning enhanced Ultrasensitive Eu-Based MOF Luminescence Senser for Clenbuterol Visible Recognition</h1>

  <p align="center">Lan Duo*, Yuze Hao*, Chuanbao Jiao and Xiaomin Kang†</p>

  <p align="center">College of Chemistry and Chemical Engineering, Inner Mongolia University, Hohhot 010021, China.</p>


This project develops a comprehensive framework for the rapid detection of Clenbuterol (Ractopamine) concentrations in wastewater. By leveraging image processing (RGB feature extraction) and various machine learning/deep learning algorithms, the system achieves precise quantitative analysis and prediction.

## Table of Contents
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Clustering Models](#clustering-models)
- [Multivariate Regression](#multivariate-regression)
- [Deep Learning Models](#deep-learning-models)
    - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
    - [ResNet with Multi-Head Attention](#resnet-with-multi-head-attention)
- [Summary of Findings](#summary-of-findings)

---

## Data Preprocessing
The initial dataset consists of RGB values extracted from images via a mobile application. To understand the data distribution, statistical analysis including violin plots and box plots were generated for the R, G, B, and C features.

![figure1](asserts/figure1.png)  
*Figure 1: (a-d) Violin plots showing attribute distribution; (e-h) Box plots showing medians, quartiles, and outliers.*

## Feature Engineering
Due to the high variance in RGB attributes which could negatively impact model training, data standardization was performed using `scikit-learn`. The values were scaled from the [50, 230] range to near 0, ensuring all features are on the same magnitude while preserving the underlying distribution.

![figure2](asserts/figure2.png)  
*Figure 2: Scatter plot matrix of RGB features and statistical distribution of labels.*

![figure3](asserts/figure3.png)  
*Figure 3: 4D scatter plot where X, Y, Z axes represent RGB values and color indicates concentration.*

## Clustering Models
To improve detection accuracy over single-signal comparison, unsupervised learning models were employed:
* **K-means & PCA:** Achieved a coefficient of determination above 0.6.
* **Hierarchical Clustering Analysis (HCA):** Used to identify samples with similar concentration characteristics.
* **Silhouette Coefficient:** Reached approximately 0.65, indicating well-defined clusters.

![figure4](asserts/figure4.png)  
*Figure 4: (a) HCA dendrogram; (b) Gap statistic for optimal clusters; (c) PCA clustering results.*

![figure5](asserts/figure5.png)  
*Figure 5: Silhouette coefficient plot.*

## Multivariate Regression
A Multivariate Linear Regression model was built to quantify Clenbuterol concentration.
* **Pearson Correlation:** Calculated to understand the linear relationship between features.
* **Assumptions Testing:** Verified through residual linearity and normality tests (Q-Q plots).

![figure6](asserts/figure6.png)  
*Figure 6: Pearson correlation heatmap between RGB values and concentration.*

![figure7](asserts/figure7.png)  
*Figure 7: Scatter plot matrix showing linear relationships between variables.*

![figure8](asserts/figure8.png)  
*Figure 8: Training results of multivariate linear regression models with 8 different alpha values.*

![figure9](asserts/figure9.png)  
*Figure 9: Linear relationship between independent variables and residuals.*

![figure10](asserts/figure10.png)  
*Figure 10: Normal Q-Q plot for residual diagnostics.*

## Deep Learning Models

### Multi-Layer Perceptron (MLP)
A Feed-forward Neural Network was implemented using a 4:1 train-test split. The model achieved a coefficient of determination above 0.9 on both sets, demonstrating high precision.

![figure11](asserts/figure11.png)  
*Figure 11: Fitting curves for true vs. predicted values in MLP training and testing sets.*

### ResNet with Multi-Head Attention
A ResNet architecture was adapted for this detection task. In addition to high accuracy (R² > 0.9), **Permutation Importance** was calculated using PyTorch to interpret the model.

![figure12](asserts/figure12.png)  
*Figure 12: ResNet fitting curves for true vs. predicted values.*

![figure13](asserts/figure13.png)  
*Figure 13: Feature importance ranking showing the significance of B and R values over G.*

## Summary of Findings
This project demonstrates that combining fluorescence image RGB analysis with machine learning provides a rapid, low-cost method for Clenbuterol detection. While clustering offers a rough range, deep learning models (MLP and ResNet) provide accurate quantitative predictions, with feature importance analysis further optimizing the detection process.

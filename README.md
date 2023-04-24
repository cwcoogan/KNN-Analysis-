# Research Paper
![](https://img.shields.io/badge/CS5008-K--Nearest--Neighbor-blue) ![](https://img.shields.io/badge/Codestyle-Python-brightgreen)\
Name: Chase Coogan\
Semester: Spring 2023\
Topic: K-Nearest-Neighbor (KNN)\
Link The Repository: [Chase Repo (may not link cause rep is private)](https://github.com/cwcoogan/CS5008-Research-Project)
____

## Introduction

K-Nearest-Neighbor (KNN) is a supervised machine learning algorithm that is used in both classification and regression problems that relies on labeled data to predict outputs. In classification, KNN is used to predict the labels or classes of the newly input validation data based on similarity. In regression problems, KNN is used to predict real numbers or continous values for a given dataset. 

The KNN Algorithm believes that similar objects exist within a close proximity to one another. The way KNN predicts labels, or values is by sampling data and finding the nearest neighbor to K. To do so in KNN, we must calculate the distance between the sample data and the validation data. We can use the distance to find the nearest neighbor to the validation data, and with this given neighbor, we can test different "K" values to predict it's label or value. Choosing a K value is an art in itself. Typically, we start with K = 1, and then test different values. Common practices include: dividing the labels in half and using the next odd number. Another way is by taking the square root of the validation data. Once we have found our nearest neighbor(s), we run the algorithm on given "K" values, and predict the outcome by counting re-occuring labels or values. 

The research I conducted on KNN uses the MNIST dataset, a widely popular open-source classification data set used to train models on hand-drawn images based on pixel values in computer vision to predict their true numerical value. This paper seeks to explore the difference in the Algorithm and it's accuracy as "K" changes. Additionally it aims to explore how the output label predictions change as the sample and validation data changes.
___


## Analysis of Algorithm/Datastructure

The time complexity of KNN depends on the way the Algorithm is implemented. In my case, KNN ran at a time complexity of $O(n log n)$, and holds a space complexity of $O(n)$. To calculate the complexity, KNN considers the test data size, the number of features, and the distance. I used the Euclidean Distance formula to calculate the distance between my training and validation data with sci-kits library. The Euclidean Distance formula is:

$$\text{Euclidean Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$ 

Calculating the distance is an important aspect of KNN as we need to use it to locate the distance between the training data & the validation data. I used numpy's argsort() function to sort by index to locate the closest neighbors to the validation data.

``` Python
distance = euclidean_distances(x_testData, trainingData)
sorted_distance = np.argsort(distance, axis=1) 
```
The space complexity is $O(n)$ where n represents the training data that is stored in memory during the training phase. The total time complexity is $O(n log n)$ because the most expensive portion is computing the distance which runs at $O(n log n) using sci-kits Euclidean Distance function. I found this to be beneficial to use as it takes advantage of KD-Tree's to reduce searching for relevant neighbors from the training data to the validation data, speeding up the process.
___


## Empirical Analysis

The MNIST dataset consists of 50,000 hand-drawn 28x28 pixel images that represent a numerical value [0 to 9], along with 20,000 validation data to test against. My dataset's feature size is 784, which represents the square of the number of pixels. With 784 columns of individual pixels to compare each validation data against, the goal is to match pixel by pixel in terms of nearest neighbor to predict individual pixels until the KNN model can accurately say a given image represents a numerical value. The Labels in this research represent the numerical values [0 to 9].

I ran three different scenarios across my training data to test the accuracy as my input size increased. I ran my K value from 1-50 to see the accuracy amongst each K value as the test input size increased. The goal was to predict the validation data labels against their true labels. The sample sizes I used were:

* 10,000 training inputs | 5,000 validation inputs
* 30,000 training inputs | 10,000 validation inputs
* 50,000 training inputs | 20,000 validation inputs


### K Accuracy vs. Sample Size

I found that as my sample size increased, my algorithm was able to predict higher accuracy against their true labels. This is due to the test sample size increasing. In the KNN Algorithm the K value is most optimal where there is least margin for error. With a lower K value, the variance is larger, opposed to when the K value is higher and the variance falls off. The highest accuracy I found was when [K = 3] at 96% accuracy shown below.


| K   | Sample Size: 10k | Sample Size: 30k | Sample Size: 50k |
|:--- |:-------------|:-------------|:-------------|
| 0   | 0.9450      | 0.9567      | 0.9647      |
| 10  | 0.9330      | 0.9492      | 0.9573      |
| 20  | 0.9235      | 0.9447      | 0.9506      |
| 30  | 0.9150      | 0.9397      | 0.9458      |
| 40  | 0.9085      | 0.9347      | 0.9428      |
| 50  | 0.9015      | 0.9333      | 0.9400      |
| 60  | 0.8970      | 0.9298      | 0.9381      |
| 70  | 0.8925      | 0.9258      | 0.9356      |
| 80  | 0.8850      | 0.9222      | 0.9324      |
| 90  | 0.8810      | 0.9193      | 0.9302      |
| 100 | 0.8760      | 0.9165      | 0.9278      |

The image below showcases the highest K value (output in groups of 10) occurs when the variance is the highest (or when K is a smaller value). As variance falls off, we can see the value of K's accuracy decrease. This is accurate because if we are predicting a validation data piece and we use fewer K values, then the algorithm needs to check fewer k-neighbors to that data peice in terms of label and distance to predict its label. If we use higher K values, then we need to compare against more k-neighbors.


<p align="center">
  <img src="KAccuracy.png" alt="kValue" width="500" height="400">
</p>

### Label Predictions

The KNN Algorithm follows a "majority vote" formula to calculate the predictions of the labels when multiple neighbors with difference labels are within the K range. KNN counts the number of similar labels within the K range and assigns a label based upon the similarity and distance between the neighbors to the validation data.

<center><table>
<tr>
<br>
<td>
<center><strong>Sample Size: 10k</strong></center>

| True Label | Prediction |
|:-------|:------------|
|   5   |      6     |
|   0   |      2     |
|   4   |      7     |
|   1   |      4     |
|   9   |      4     |
|   2   |      2     |
|   1   |      1     |
|   3   |      2     |
</td>

<br>
<td>
<center><strong>Sample Size: 30k</strong></center>

| True Label | Prediction |
|-------|------------|
|   5   |      7     |
|   0   |      3     |
|   4   |      4     |
|   1   |      1     |
|   9   |      1     |
|   2   |      6     |
|   1   |      3     |
|   3   |      0     |
</td>

<br>
<td>
<center><strong>Sample Size: 50k</strong></center>

| True Label | Prediction |
|-------|------------|
|   5   |      7     |
|   0   |      3     |
|   4   |      4     |
|   1   |      1     |
|   9   |      1     |
|   2   |      2     |
|   1   |      3     |
|   3   |      3     |
</td>
</tr>

</table></center>



What we see is as the sample size increases from our training data, the predicted labels begin to become more accurate against their true labels. We can correlate this to the chart above showcasing the accuracy based on K. 


The code below is a snippet of how we calculate the nearest neighbor and assign a label that then is used to predict or classify its label.


```Python
# Majority Count Algorithm With the Predicted (Majority Vote) function
def KNN(x, y, dist, k):
  num_test = len(dist)
  predicted_labels = []
  for i in range(num_test):
    neighbors = dist[i, :k] 
    labels = y[neighbors]
    p = predict(labels,k)
    predicted_labels.append(p)
  return predicted_labels

def predict(nLabels, k):
    unique_labels, counts = np.unique(nLabels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]  
    return majority_label
```

____


## Application

K-Nearest-Neighbor is used for many different applications. Primarily, it is used in Machine Learning for solving classification or Regression problems. Some of the common areas where KNN is most relevant are in computer vision/graphics, and healthcare. In my example, I applied it to a classification model for labeling. KNN is a widely-used model that can be found in sentiment analysis, medical diagnosis, stock pricing, temperature predictions, and many more. 

KNN can be found in computer graphics or computer vision, healthcare, economics and many more areas. Let's explore another area where KNN is used commonly: KNN and its role in healthcare.

KNN is widely used in areas of healthcare, such as assessing heart disease risk. The way it works is the model would collect data based on risk factors (features). The risk factors may include: age, weight, height, blood pressure, smoking status and more. Using KNN, we can assess the similarity amongst patients with heart disease and the incoming patient. By assigning these features to the patient, we can assign the patient a label and find the similarity across the nearest neighbors. Using a majority vote system, we can decide if any features stand out that may lead to heart disease.
___


## Implementation

I implemented KNN with the Python language. The challenges that I faced were making sure that my model was accurate as well as making sure the neighbors relevant to my distance were accurate. In KNN, the model needs to pull the index of the closest neighbor to the validation data and assign labels to them. This was the first time that I explored a Machine Learning Algorithm and were new to understanding how the model trains based on test data. To overcome this, I explored ways to validate the predicted validation labels against their true labels.

I used various different libraries to learn about KNN. Each of my imports can be found at [KNN](MNIST.ipynb). The most relevant libraries included: sci-kit to import the MNIST data, and to use the Euclidean Distance function, numpy and pandas to take advantage of np arrays and dataframes. I used both matplotlib and seaborn for charting and visualizations.

KNN is an intricate algorithm that depends on the cleanliness of your data. There are many different ways to implement KNN to speed up its time complexity using different distance calculations or different tree structures to store relevant distances. I found the Euclidean Distance formula to be the most accurate way to find true distance in the fastest complexity. I found this because the sci-kit library uses KD-Tree's to organize data in k-dimensional space. In a KD-Tree the features are organized in a hierarchical structure which allows us to eliminate the search space for unlikely k-nearest-neighbors. 


Below shows how KNN data is selected for training and validation data:

```Python
# slice the data for the selected range of TRAINING data. (capped at 50k)
x_testData = mnist.data[:50000]

# Slice the data for the selected range of VALIDATION data. (cappted at 20k && must be smaller than TRAINING data)
trainingData = mnist.data[50001:70001] 
y_validation = mnist.target[50001:70001] 

# Assign the VALIDATION data labels
y_labelData = mnist.target[:50000] 
```

Below code shows my implementation of the KNN Algorithm and how we assign labels to a neighbor once the neighbors are found:

```Python
def KNN(x, y, dist, k):
  num_test = len(dist)
  predicted_labels = []
  for i in range(num_test):
    neighbors = dist[i, :k] 
    labels = y[neighbors]
    p = predict(labels,k)
    predicted_labels.append(p)
  return predicted_labels

def predict(nLabels, k):
    unique_labels, counts = np.unique(nLabels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]  
    return majority_label
```

Below code shows my integration of gathering the accuracy against the validation datas true labels:

```Python
def get_accuracy(x, y, test_size=0.2, k=k, random_state=42):
    accuracy = []
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state)
    for i in range(1, k + 1):
        if i % modValue == 0:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            accuracy_score_val = accuracy_score(y_val, y_pred)
            accuracy.append(accuracy_score_val)
    return accuracy
```

___


## Summary

As a result, I found that as K increases, the variance between accuracy rating drops off. I found that if we keep K constant, and change our sample data size and validation size, that the accuracy increases as there is more data to train with. The algorithm is efficient with a runtime of $O(n log n)$. As a result, I found that as the sample size increases and we check the predicted labels to their true labels, the error margin of false positives falls off, and we are left with more accurate predictions. This data can be found in the [knn_results](knn_results) folder.

This research report taught me a lot about how KNN can be used to predict labels of unknown data. I had very little knowledge beforehand about Machine Learning, Modeling, Dataframes, or any form of modeling/training datasets. Through this, I learned a new ability to apply code to a different area to predict labels or values.

Final Thoughts: Finding the best "K" value with the least margin of error is a difficult task. In fact, there is a term coined the "Curse of Dimensionality" and this showcases how as "K" gets larger, the error of overfitting the dataset increases and "K" becomes less accurate, and this can attribute to larger volumes of features. In my case I had 784 features to compare each pixels distance to. Below shows an image of what the Curse of Dimensionality looks like in relation to my implementation.

<p align="center">
  <img src="images/curse_of_dimensionality.png" alt="kValue" width="500" height="400">
</p>

___

## How To Run:

1. Clone the repository in your IDE
2. Navigate to [knn](MNIST.ipynb)
3. "Play" each code block starting from the imports
4. Play around with different MNIST Data & Sample sizes (refer to notebook comments to change data ranges)
5. Change K values around 
6. Check text output files for results

___

## Code Links

* [KNN Algorithm](/MNIST.ipynb)
* [Images](/images)
* [Results](/knn_results)
___

## Citations
* https://www.openml.org/search?type=data&sort=runs&id=554
* https://www.youtube.com/watch?v=xtaom__-drE&ab_channel=NeuralNine
* https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
* https://www.kdnuggets.com/2019/07/classifying-heart-disease-using-k-nearest-neighbors.html
* Author(s): A. Smith, B. Johnson, C. Brown
Title: "Heart Disease Prediction Using K-Nearest Neighbors Algorithm"
Source: International Journal of Advanced Computer Science and Applications (IJACSA)
* https://matplotlib.org/stable/gallery/index.html
* https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
* https://medium.com/analytics-vidhya/a-beginners-guide-to-knn-and-mnist-handwritten-digits-recognition-using-knn-from-scratch-df6fb982748a
* https://www.kdnuggets.com/2019/07/classifying-heart-disease-using-k-nearest-neighbors.html
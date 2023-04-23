# Research Paper
![](https://img.shields.io/badge/CS5008-K--Nearest--Neighbor-blue) ![](https://img.shields.io/badge/Codestyle-Python-brightgreen)\
Name: Chase Coogan\
Semester: Spring 2023\
Topic: K-Nearest-Neighbor (KNN)\
Link The Repository: [Chase Repo (may not link cause rep is private)](https://github.com/cwcoogan/CS5008-Research-Project)

## Introduction

K-Nearest-Neighbor (KNN) is a supervised machine learning algorithm that is used in both classification and regression problems that relies on labeled data to predict outputs. In classification, KNN is used to predict the labels or classes for the newly input validation data based on similarity. In regression problems, KNN is used to predict real numbers or continous values for a given dataset. The difference between KNN in classification and regression is the output data. KNN is widely used across many fields such as computer graphics, computer vision, image recognition & processing, healthcare and more. 

The KNN Algorithm believes that similar objects exist within a close proximity to one another. The way KNN predicts labels, or values is by sampling data and finding the K-Nearest-Neighbor. To do so in KNN, we must calculate the distance between the sample data and the validation data. We can use the distance to find the nearest Neighbor to the validation data, and with this given neighbor, we can test different "K" values to predict it's label and/or it's value. Choosing a K value is an art in itself. Typically, we start with K = 1, and then test different values. Common practices are dividing your labels in half and using the nearest half odd number. Another way is taking the square root of your validation data. Once, we have found our nearest neighbor(s), we run the algorithm on given "K" values, and predict the outcome by counting re-occuring labels or values. 

The research I conducted on KNN uses the MNIST dataset, a widely popular open-source data set used to train models on hand-drawn images based on pixel values in computer vision to predict their true numerical value. This paper seeks to explore the difference in the Algorithm and it's accuracy as "K" changes, and explores how the output label predictions change as the sample and validation data changes.

## Analysis of Algorithm/Datastructure

The time complexity of KNN depends on the way the Algorithm is implemented. In my case, KNN ran at a time complexity of $O(n log n)$, and holds a space complexity of $O(n)$. To calculate the complexity, KNN considers the test dataset size, the number of features, and the distance. I used the Euclidean Distance formula to calculate the distance between my training data and my validation data with sklearns library. The Euclidean Distance formula is:

$$\text{Euclidean Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$ 

Calculating the distance is an important aspect of KNN as we need to use it to locate the distance between the training data & the validation data. I used numpy's argsort() function to sort by index to locate the closest neighbors to the validation data.

``` Python
distance = euclidean_distances(x_testData, trainingData)
sorted_distance = np.argsort(distance, axis=1) 
```

## Empirical Analysis
- What is the empirical analysis?
- Provide specific examples / data.


## Application
- What is the algorithm/datastructure used for?
- Provide specific examples
- Why is it useful / used in that field area?
- Make sure to provide sources for your information.


## Implementation
- What language did you use?
- What libraries did you use?
- What were the challenges you faced?
- Provide key points of the algorithm/datastructure implementation, discuss the code.
- If you found code in another language, and then implemented in your own language that is fine - but make sure to document that.


## Summary
- Provide a summary of your findings
- What did you learn?
# TTT4275
Classification projects, Iris and Numbers

Iris
1. The first part has focus on design/training and generalization. 
(a) Choose the first 30 samples for training and the last 20 samples for testing. 
(b) Train a linear classifier as described in subchapter 2.4 and 3.2. Tune the step factor ↵ in equation 19 until the training converge. 
(c) Find the confusion matrix and the error rate for both the training and the test set.
(d) Now use the last 30 samples for training and the first 20 samples for test. Repeat the training and test phases for this case. 
(e) Compare the results for the two cases and comment 2. 

2. The second part has focus on features and linear separability. In this part the first 30 samples are used for training and the last 20 samples for test. 
(a) Produce histograms for each feature and class. Take away the feature which shows most overlap between the classes. Train and test a classifier with the remaining three features. 
(b) Repeat the experiment above with respectively two and one features. 
(c) Compare the confusion matrixes and the error rates for the four experiments. Comment on the property of the features with respect to linear separability both as a whole and for the three separate classes.



Numbers
The task consists of two parts both using variants of a nearest neighbourhood classifier. 
1. In the first part part the whole training set shall be used as templates. (a) Design a NN-based classifiser using the Euclidian distance. Find the confusion matrix and the error rate for the test set. The data sets should preferably be split up into chunks of images (for example 1000) in order to a) avoid too big distance matrixes b) avoid using excessive time (as when classifying a single image at a time) 
(b) Plot some of the misclassifed pixtures. Some useful Matlab commands for this are : • x = zeros(28,28); x(:)= testv(i,:); will convert the pixture vector (number i) to a 28x28 matrix • image(x) will plot the matrix x • dist(template,test) will calculate the Euclidian distance between a set of templates and a set of testvectors, both in matrix form. 
(c) Also plot some correctly classified pixtures. Do you as a human disagree with the classifier for some of the correct/incorrect plots? 

2. In the second part you shall use clustering to produce a small(er) set of templates for each class. The Matlab function [idxi, Ci] = kmeans(trainvi,M); will cluster training vectors from class !i into M templates given by the matrix Ci. 
(a) Perform clustering of the 6000 training vectors for each class into M = 64 clusters. 
(b) Find the confusion matrix and the error rate for the NN classifier using these M = 64 templates pr class. Comment on the processing time and the performance relatively to using all training vectors as templates. 
(c) Now design a KNN classifier with K=7. Find the confusion matrix and the error rate and compare to the two other systems.


# Handwritten-Digit-Recognition
<br>

## About Dataset :
Given the data file train.csv contain gray-scale images of hand-drawn digits, from zero to nine, predict that digit. Also compare the accuracy of various classification algorithms.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

<br>

## Details :
I implemented my own K-nearest-neighbors algorithm to check if digit matches with the image.


<img  src = "https://github.com/codeboy47/Handwritten-Digit-Recognition/blob/master/Images/HandwrittenDigit.png" />

<img  src = "https://github.com/codeboy47/Handwritten-Digit-Recognition/blob/master/Images/Accuracy.png" />

###Result : 
Predicted digit matches with the digit shown in image.

<br>

## Accuracy

| Algorithm | Accuracy |
| --- | --- |
| Naive Bayes Classifier | 0.559 |
| Decision Tree Classifier | 0.851 |
| Random Forest Classifier | 0.961 |

Random Forest Classifier performs best in predicting the handwritten digit with an accuracy of 0.961 or 96.1%.

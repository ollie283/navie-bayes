# Text classiﬁcation

## A program written in python that performs Naive Bayes Classiﬁcation.

**Data:** The folder has 3 ﬁles: ```sampleTrain.txt```, ```sampleTrain.vocab.txt``` and ```sampleTest.txt```.
sampleTrain.txt is the training data for building a classiﬁer. 
The vocabulary of the training data is in ```sampleTrain vocab.txt```. 
The classiﬁer should be ﬁnally run and evaluated on test data in ```sampleTest.txt```. The second column in the sampleTrain.txt and sampleTest.txt ﬁles gives the gold standard true class for each document. The ﬁrst column of these ﬁles is the document id, the third column gives the words in the document. The columns are separated by tab spaces.

There are 2 classes in the data 0 and 1.

**Task:** Build a Naive Bayes classiﬁer using the document words as features. It should compute a model given some training data and be able to predict classes on a new test set. For this project, use ```sampleTrain.txt``` for training a model and the model should be used to predict classes for documents in ```sampleTest.txt```. Use Laplace smoothing for feature likelihoods. There is **no** need for UNK token. The dataset has been simpliﬁed so that the test corpus only contains words seen during training (so no need for UNK). There is also no need to smooth the prior probabilities.

**Code:** Should run without any arguments. It should read ﬁles in the same directory. Absolute paths must not be used. It should print values in the following format:
```
    Prior probabilities
    class 0 = 
    class 1 =

    Feature likelihoods 
            great    sad    boring ... 
    class 0 
    class 1

    Predictions on test data 
    d5 = 
    d6 = 
    d7 = 
    d8 = 
    d9 = 
    d10 =

    Accuracy on test data =
```
The features in the feature likelihood table (great, sad, boring, ...) can be printed in any order.		

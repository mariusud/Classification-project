# Classification-project
Classification of Flowers and handwritten numbers, for the course TT4275


## Part 1 - Iris recognizition
For this task, a linear classifier was trained on 30 samples and tested on 20 with focus on design and testing.
Confusion matrix and error rate was found for both train/test sets to compare performance.

### Problem 1
With 30 samples training and 20 for testing we get   
correct guesses:  96.66666666666667 %  
Minimum mean-squared error obtained:  0.06688564151844724  
confusion matrix:  
 [[20.  0.  0.]  
 [ 0. 18.  0.]  
 [ 0.  2. 20.]]  
With last 30 samples training and first 20 for testing we get  
correct guesses:  98.33333333333333 %  
Minimum mean-squared error obtained:  0.06893705268438184  
confusion matrix:  
 [[20.  0.  0.]  
 [ 0. 19.  0.]  
 [ 0.  1. 20.]]  


![alt text](https://raw.githubusercontent.com/mariusud/Classification-project/master/mse_values.png)
### Problem 2

Removed 1 feature  
correct guesses:  95.0 %  
Minimum mean-squared error obtained:  0.07818070062584248  
confusion matrix:    
 [[20.  0.  0.]  
 [ 0. 18.  1.]  
 [ 0.  2. 19.]]  

Removed 2 features  
correct guesses:  93.33333333333333 %  
Minimum mean-squared error obtained:  0.08511926507021324  
confusion matrix:  
 [[20.  0.  0.]  
 [ 0. 17.  1.]  
 [ 0.  3. 19.]]  

Removed 3 features  
correct guesses:  33.333333333333336 %  
Minimum mean-squared error obtained:  0.21179482409985068  
confusion matrix:  
 [[ 0.  0.  0.]  
 [ 0.  0.  0.]  
 [20. 20. 20.]]  
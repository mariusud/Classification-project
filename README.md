# Classification-project
Classification of Flowers and handwritten numbers, for the course TTT4275


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


![alt text](https://raw.githubusercontent.com/mariusud/Classification-project/master/figures/mse_values.png)
### Problem 2
![alt text](https://raw.githubusercontent.com/mariusud/Classification-project/master/figures/histogram1.png)
![alt text](https://raw.githubusercontent.com/mariusud/Classification-project/master/figures/histogram2.png )
![alt text](https://raw.githubusercontent.com/mariusud/Classification-project/master/figures/histogram3.png )

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



 ## Part 2 - Vowel recognition

### Problem 1

Training set confusion matrix   
[[65.  0.  0.  5.  0.  0.  0.  0.  0.  0.  0.  0.]   
 [ 0. 66.  4.  0.  0.  0.  0.  0.  0.  0.  1.  0.]   
 [ 0.  4. 63.  0.  0.  0.  0.  0.  0.  0.  2.  0.]   
 [ 5.  0.  0. 65.  0.  0.  0.  0.  0.  1.  0.  0.]   
 [ 0.  0.  0.  0. 70.  0.  0.  0.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0. 68.  0.  3.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  1. 70.  0.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  1.  0. 67.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0. 69.  0.  0.  2.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0. 67.  0.  2.]   
 [ 0.  0.  3.  0.  0.  0.  0.  0.  0.  1. 67.  0.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  0. 66.]]   
   
 Testing set confusion matrix   
[[60.  0.  0.  1.  0.  1.  0.  0.  0.  0.  0.  0.]   
 [ 3. 58.  8.  0.  0.  0.  0.  0.  0.  0.  0.  0.]   
 [ 0.  8. 52.  0.  0.  0.  0.  0.  1.  0.  1.  0.]   
 [ 6.  1.  0. 66.  0.  0.  0.  0.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0. 69.  0.  0.  0.  0.  0.  1.  0.]   
 [ 0.  0.  0.  0.  0. 67.  0.  5.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  0. 68.  2.  0.  0.  0.  1.]   
 [ 0.  0.  0.  0.  0.  1.  0. 62.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0. 65.  0.  0.  3.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0. 67.  2.  1.]   
 [ 0.  2.  9.  2.  0.  0.  0.  0.  0.  1. 65.  0.]   
 [ 0.  0.  0.  0.  0.  0.  1.  0.  3.  1.  0. 64.]]   
Model accuracy for training set:  0.955952380952381   
Model accuracy for testing set:  0.9214975845410628   

#### For the diagonal model

Training set confusion matrix   
[[46.  0.  0.  5.  0.  0.  0.  0.  0.  0.  0.  0.]   
 [ 0. 50. 13.  0.  0.  0.  0.  0.  0.  0.  7.  0.]   
 [ 0.  9. 56.  0.  0.  0.  0.  0.  1.  0.  5.  0.]   
 [24.  0.  0. 64.  1.  0.  1.  0.  0.  2.  5.  0.]   
 [ 0.  0.  0.  1. 68.  0.  0.  0.  0.  0.  0.  1.]   
 [ 0.  0.  0.  0.  1. 37.  3.  4.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0. 29. 66.  1.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  4.  0. 65.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0. 63.  0.  0.  9.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0. 56.  7.  6.]   
 [ 0. 11.  1.  0.  0.  0.  0.  0.  3. 10. 46.  0.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0.  3.  2.  0. 54.]]   

  Testing set confusion matrix   
[[49.  0.  0.  5.  2.  0.  0.  0.  0.  0.  0.  0.]   
 [ 2. 58. 19.  0.  0.  0.  0.  0.  0.  0.  1.  0.]   
 [ 0.  8. 41.  0.  0.  0.  0.  0.  0.  0.  3.  0.]   
 [18.  1.  0. 56.  0.  1.  0.  0.  0.  1.  1.  0.]   
 [ 0.  0.  1.  0. 67.  0.  0.  0.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0. 51.  1.  9.  0.  0.  0.  0.]   
 [ 0.  0.  0.  2.  0. 11. 66.  1.  0.  0.  0.  2.]   
 [ 0.  0.  0.  0.  0.  6.  1. 59.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  0.  0.  0. 60.  0.  0.  5.]   
 [ 0.  0.  0.  1.  0.  0.  0.  0.  1. 58.  8.  5.]   
 [ 0.  2.  8.  5.  0.  0.  0.  0.  2.  4. 56.  0.]   
 [ 0.  0.  0.  0.  0.  0.  1.  0.  6.  6.  0. 57.]]   
Model accuracy for training set:  0.7988095238095239   
Model accuracy for testing set:  0.8188405797101449   


 ### Problem 2
GMM using 2 gaussians:      
Estimated error rate for testing set:  29.830917874396135 %        
confusion matrix:      
 [[64.  0.  0.  2.  3.  0.  0.  0.  0.  0.  0.  0.]        
 [ 2. 59.  5.  1.  1.  0.  0.  0.  0.  0.  1.  0.]      
 [ 0. 27. 37.  0.  1.  0.  0.  0.  0.  0.  4.  0.]  
 [13.  0.  0. 51.  0.  0.  4.  0.  0.  0.  1.  0.]      
 [ 9.  0.  0. 12. 44.  4.  0.  0.  0.  0.  0.  0.]    
 [ 0.  0.  0.  1.  0. 25.  1. 42.  0.  0.  0.  0.]   
 [ 0.  0.  0.  0.  0.  2. 47. 20.  0.  0.  0.  0.]    
 [ 0.  0.  0.  0.  0.  6.  0. 63.  0.  0.  0.  0.]   
 [ 0.  0.  1.  0.  0.  0.  0.  0. 58.  2.  4.  4.]    
 [ 0.  0.  0. 13.  0.  0.  0.  0.  0. 46. 10.  0.]   
 [ 0.  3.  3. 12.  0.  0.  0.  0.  0.  0. 51.  0.]    
 [ 0.  0.  0.  2.  1.  4.  3.  0. 12. 11.  0. 36.]]  
Model accuracy for training set:  0.9154761904761904    
Model accuracy for testing set:  0.7016908212560387   

GMM using 3 gaussians:     
Estimated error rate for testing set:  31.28019323671497 %    
confusion matrix:    
 [[59.  0.  0.  5.  0.  2.  0.  3.  0.  0.  0.  0.]   
 [ 0. 59.  6.  3.  0.  0.  0.  0.  0.  0.  1.  0.]     
 [ 0. 34. 29.  0.  1.  0.  0.  0.  0.  0.  5.  0.]    
 [24.  0.  0. 41.  0.  0.  3.  0.  0.  0.  1.  0.]    
 [ 6.  0.  0.  3. 54.  6.  0.  0.  0.  0.  0.  0.]      
 [ 0.  0.  0.  1.  0. 26.  2. 40.  0.  0.  0.  0.]    
 [ 1.  0.  0.  1.  0.  5. 50. 12.  0.  0.  0.  0.]     
 [ 0.  0.  0.  0.  0.  8.  0. 61.  0.  0.  0.  0.]   
 [ 0.  0.  1.  0.  0.  0.  0.  0. 57.  3.  4.  4.]     
 [ 0.  0.  0. 11.  0.  0.  0.  0.  0. 47. 10.  1.] 
 [ 0.  2.  2. 12.  0.  0.  0.  0.  0.  1. 52.  0.]      
 [ 1.  0.  0.  0.  1.  5.  2.  0. 17.  9.  0. 34.]]    
Model accuracy for training set:  0.9130952380952381   
Model accuracy for testing set:  0.6871980676328503  
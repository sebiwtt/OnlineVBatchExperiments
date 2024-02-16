# Results
Accuracy: 0.9933774834437085
TP: 6.666666666666667, FP: 0.6666666666666666, TN: 1343.3333333333333, FN: 8.333333333333334
Recall: 0.4444444444444445, Precision: 0.9259259259259259, F1 Score: 0.597041847041847

## Additional Results 
- Computed in the BatchARIMA.ipynb
- Measured scores before and after the concept drift

### Before CD
- Accuracy: 0.9984375
- TP: 5, FP: 0, TN: 634, FN: 1
- Recall: 0.8333333333333334, Precision: 1.0, F1 Score: 0.9090909090909091
### After CD
- Accuracy: 0.9916434540389972
- TP: 3, FP: 0, TN: 709, FN: 6
- Recall: 0.3333333333333333, Precision: 1.0, F1 Score: 0.5
### After CD using threshold from before CD
- Accuracy: 0.04317548746518106
- TP: 3, FP: 681, TN: 28, FN: 6
- Recall: 0.3333333333333333, Precision: 0.0043859649122807015, F1 Score: 0.008658008658008658
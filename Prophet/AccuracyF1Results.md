# Results
Accuracy: 0.9580721118469464
TP: 2.52, FP: 44.55, TN: 1299.5, FN: 12.43
Recall: 0.16838095238095235, Precision: 0.5930691022931889, F1 Score: 0.17720685581177129

## Additional Results 
- Computed in the Prophet.ipynb
- Measured scores before and after the concept drift

### Before CD
- Accuracy: 0.99375
- TP: 3, FP: 2, TN: 633, FN: 2
- Recall: 0.6, Precision: 0.6, F1 Score: 0.6

### After CD
- Accuracy: 0.9888579387186629
- TP: 2, FP: 0, TN: 708, FN: 8
- Recall: 0.2, Precision: 1.0, F1 Score: 0.3333333333333333

### After CD using threshold from before CD
- Accuracy: 0.6434540389972145
- TP: 3, FP: 249, TN: 459, FN: 7
- Recall: 0.3, Precision: 0.011904761904761904, F1 Score: 0.022900763358778626
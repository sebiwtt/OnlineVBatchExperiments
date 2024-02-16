# Time
## PAD
- 200 ms ± 533 µs per loop 
## ARIMA
- 1min 14s ± 559 ms per loop
## Prophet
- 466 ms ± 4.54 ms per loop 

# MAE and MSE
## PAD
- MAE: 0.19119518007573544
- MSE: 0.06132180083036618

## ARIMA
- MAE: 0.8727022616227608
- MSE: 1.2415427363956855

## Prophet
- MAE: 0.9140117595760887
- MSE: 1.3170267382015617

# F1 Score
## PAD
- F1: 0.7589498928434161

## ARIMA
- F1: 0.597041847041847

## Prophet
- F1: 0.17720685581177129

# F1 Score before and after CD
## PAD
- F1 Before: 0.8888888888888888
- F1 After: 0.782608695652174
- F1 After (no change in threshold): 0.7272727272727273

## ARIMA
- F1 Before: 0.9090909090909091
- F1 After: 0.5
- F1 After (no change in threshold): 0.008658008658008658

## Prophet
- F1 Before: 0.6
- F1 After:  0.3333333333333333
- F1 After (no change in threshold): 0.022900763358778626
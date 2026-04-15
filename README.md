# mmllm
mmllm
## EXP-0 the initial toy experiment with Hyena on 5mC and 5hmC tracks, using the same 10k regions as in the original Hyena toy experiment. The goal is to see if Hyena can learn to predict the 5mC track from the 5hmC track, which are known to be correlated but not identical.
### Note:0414 I have tried to tune the initial param, confirming the crosshyena with long_mixer='conv' and filter_len=4. Then add the Hyena layer. The r2 in 3000 data dataset is around 0.4.\
## Note: 0415 I have run the model from 1000 to 70000 data points. The r2 is around 0.5 and pearson r is 0.7 for 70000 data points, which is pretty good for this toy experiment. The model is still underfitting, and I will try to tune the hyperparameters and train for more epochs to see if I can get better performance.

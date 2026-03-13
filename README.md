import satisfaction_score_predict.py once in the system

Predict() do calc the weighted product [0, 2.5, 5] with the probs which are calculated by phobert model. Then use predict() to predict the score and update it to overall score of user or product (which is average of all rating score).

EX: if probs = [0.1, 0.2, 0.7], then score = 0*0.1 + 2.5*0.2 + 5*0.7 = 4.75.
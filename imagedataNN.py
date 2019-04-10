from KerasNNArchitecture import NeuralNetwork
import pandas as pd, numpy as np, tensorflow as tf
from sklearn.preprocessing import StandardScaler

test = NeuralNetwork()


bio_data = pd.read_csv('C:/Users/Alex/Documents/School/Machine Learning/Project/output_nn/61016P2D221.csv')

'''
Try loading all the csv data instead of just one person

Training set mix of  
'''
bio_scores = bio_data.iloc[:, :].values
count = 0
for i in range(len(bio_scores)):
    if not pd.isnull(bio_scores[i][22]):
        count = count + 1

X = np.ones((count, 19), dtype=float)
y = np.ones((count, 3), dtype=float)
offset = 0
for i in range(len(bio_scores)):
    if not pd.isnull(bio_scores[i][22]):
        for j in range(1, 20):
            X[i - offset][j - 1] = bio_scores[i][j]

        y[i - offset][0] = bio_scores[i][21]
        y[i - offset][1] = bio_scores[i][22]
        y[i - offset][2] = bio_scores[i][23]  #dividing by 100 (our scale) gives higher accuracies
    else:
        offset = offset + 1

'''
scaler = StandardScaler()
print(scaler.fit(X))
StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.transform(X[:][:])
'''

acc = 0
test.num_epochs = 30
test.num_hidden_layers = 3
test.hidden_units = 64
trainSize = int(len(X) * .8)
test.load_data(X, y, trainSize, len(X) - trainSize)

test.hidden_act_type= tf.nn.relu
test.output_nodes = 3
test.model_optimizer = "rmsprop"
test.model_loss = "mean_absolute_error"
count = 0
prevAcc = -1
maxAcc = -1
bestEpochNum = 0

while acc < .95 and count < 10:
    prevAcc = acc
    loss, acc, model = test.evaluate()
    test.num_epochs = test.num_epochs + 3

    print("Predictions on test set :\n{}".format(np.transpose(model.predict(X[trainSize:]))))

    print("List of y values: \n{}".format(np.transpose(y[trainSize:])))
    print(acc)
    count = count + 1
    print(count)
    if acc > maxAcc:
        maxAcc = acc
        bestEpochNum = test.num_epochs
print("Best accuracy found to be {} with epochs num of {}".format(maxAcc, bestEpochNum))
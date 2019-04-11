from KerasNNArchitecture import NeuralNetwork
import pandas as pd, numpy as np, tensorflow as tf, os
from sklearn.preprocessing import MinMaxScaler

test = NeuralNetwork()


file_list = os.listdir('C:/Users/Alex/Documents/School/Machine Learning/Project/output_nn/')
bio_data = pd.read_csv('C:/Users/Alex/Documents/School/Machine Learning/Project/output_nn/{}'.format(file_list[0]))
bio_scores = bio_data.iloc[:, :].values

for i in range(1, len(file_list)):
    bio_data = pd.read_csv('C:/Users/Alex/Documents/School/Machine Learning/Project/output_nn/{}'.format(file_list[i]))
    bio_scores = np.concatenate((bio_scores, bio_data.iloc[:, :].values), axis=0)

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


# # might not need this
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y)

acc = 0
test.num_epochs = 45
test.num_hidden_layers = 20
test.hidden_units = 35
trainSize = int(len(X) * .8)
test.load_data(X, y, trainSize, len(X) - trainSize)

test.hidden_act_type= tf.nn.relu
test.output_nodes = 3
test.model_optimizer = "rmsprop"        # radam, rmsprop
test.model_loss = "mean_squared_error"    #mean_absolute_error
count = 0
prevAcc = -1
maxAcc = -1
bestEpochNum = 0

while acc < .95 and count < 1:     # Not having luck with this method
    prevAcc = acc
    loss, acc, model = test.evaluate()
    test.num_hidden_layers = test.num_hidden_layers + 1
    predictions = model.predict(X[trainSize:])
    print("Predictions on test set :\n{}".format(predictions))

    print("List of y values: \n{}".format((y[trainSize:])))
    print(acc)
    count = count + 1
    print(count)
    if acc > maxAcc:
        maxAcc = acc
        bestEpochNum = test.num_epochs
print("Best accuracy found to be {} with epochs num of {}".format(maxAcc, bestEpochNum))

# Curiousity of ranges, max, mins
# max1 = 0
# min1 = 100
# avg = 0
# for i in range(len(predictions)):
#     for j in range(3):
#         if predictions[i][j] < min1:
#             min1 = predictions[i][j]
#         if predictions[i][j] > max1:
#             max1 = predictions[i][j]
#         avg = avg + predictions[i][j]
#
# print("PREDICTIONS:\nmax is {} and min is {}".format(max1, min1))
# print("range is {} and avg is {}".format(max1 - min1, avg / (3 * len(predictions))))
#
#
# max1 = 0
# min1 = 100
# avg = 0
# numInPredictionRange = 0
# for i in range(len(y)):
#     for j in range(3):
#         if y[i][j] < min1:
#             min1 = y[i][j]
#         if y[i][j] > max1:
#             max1 = y[i][j]
#         avg = avg + y[i][j]
#         if .52576 > y[i][j] > .46236586:
#             numInPredictionRange = numInPredictionRange + 1
#
# print("\nY VALUES:\nmax is {} and min is {}".format(max1, min1))
# print("range is {} and avg is {}".format(max1 - min1, avg / (3 * len(y))))
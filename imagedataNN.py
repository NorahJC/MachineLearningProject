from KerasNNArchitecture import NeuralNetwork
import pandas as pd, numpy as np, tensorflow as tf, os, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

emotion_detection_ann = NeuralNetwork()

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


# does not affect our results
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y)

acc = 0
emotion_detection_ann.num_epochs = 45
emotion_detection_ann.num_hidden_layers = 5
emotion_detection_ann.hidden_units = 35
trainSize = int(len(X) * .8)
emotion_detection_ann.load_data(X, y)

emotion_detection_ann.hidden_act_type= tf.nn.relu
emotion_detection_ann.output_nodes = 3
emotion_detection_ann.model_optimizer = "rmsprop"        # radam, rmsprop
emotion_detection_ann.model_loss = "mean_squared_error"    #mean_absolute_error
count = 0
prevAcc = -1
maxAcc = -1
bestEpochNum = 0

while acc < .95 and count < 1:     # Not having luck with this method
    prevAcc = acc
    loss, acc, model, history = emotion_detection_ann.evaluate()
    emotion_detection_ann.num_hidden_layers = emotion_detection_ann.num_hidden_layers + 1
    predictions = model.predict(X[trainSize:])
    print("Predictions on test set :\n{}".format(predictions))

    print("List of y values: \n{}".format((y[trainSize:])))
    print(acc)
    count = count + 1
    print(count)
    if acc > maxAcc:
        maxAcc = acc
        bestEpochNum = emotion_detection_ann.num_epochs
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




# Plot training & validation accuracy values
# https://keras.io/visualization/   based on visualization logic from here
fig, ax = plt.subplots(nrows=2, ncols=1)
print(history.history.keys())

ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title("Model Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend(['Train', 'Test'], loc='lower left')

# Plot training & validation loss values
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Test'], loc='lower left')
plt.tight_layout()
plt.show()

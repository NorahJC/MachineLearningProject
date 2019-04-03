from KerasNNArchitecture import NeuralNetwork
import pandas as pd, numpy as np, tensorflow as tf

'''
def __init__(self, hidden_units=64, hiddent_act_type=tf.nn.relu, num_epochs=64, num_hidden_layers=6,
                 model_optimizer='Nadam', model_loss='sparse_categorical_crossentropy'):
'''
test = NeuralNetwork()

bio_data = pd.read_csv('C:/Users/Alex/Documents/School/Machine Learning/Project/output_nn/61016P1D220.csv')
bio_scores = bio_data.iloc[:, :].values
count = 0
for i in range(len(bio_scores)):
    if not pd.isnull(bio_scores[i][22]):
        count = count + 1

X = np.ones((count, 21), dtype=float)
y = np.ones((count, 1), dtype=float)
offset = 0
for i in range(len(bio_scores)):

    if not pd.isnull(bio_scores[i][22]):
        for j in range(21):
            X[i - offset][j] = bio_scores[i][j]
        y[i - offset][0] = bio_scores[i][21] / 100
    else:
        offset = offset + 1
print(np.shape(y))
print(np.shape(X))
print(y[402])
acc = 0
test.num_epochs = 16
test.num_hidden_layers = 16
test.hidden_units = 20
test.load_data(X, y, 300, 100)
count = 0
while acc < .95:
    loss, acc, model = test.evaluate()
    test.hidden_units = test.hidden_units + 1
    listTest = np.sum(model.predict(X[302:402]), axis=1).tolist()
    print(model.predict(X[302:402]))
    print(listTest)
    print(acc)
    count = count + 1
print(count)


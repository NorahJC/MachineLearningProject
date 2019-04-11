import tensorflow as tf
'''
To DO - 
Convert into python class with structure
__init__ :
    load data from constructor
    setup vars from constructor

save()
plot()    
'''


class NeuralNetwork(object):
    hidden_units = 0
    hidden_act_type = ""
    num_epochs = 0
    num_hidden_layers = 0
    x_train, x_test, y_train, y_test = 0,0,0,0
    model_optimizer, model_loss = ' ', ' '
    output_act_type = ""
    output_nodes = 0

    # constructor
    def __init__(self, hidden_units=64, hiddent_act_type=tf.nn.relu, num_epochs=64, num_hidden_layers=6,
                 model_optimizer='Nadam', model_loss='sparse_categorical_crossentropy',
                 output_act_type=tf.nn.sigmoid, output_nodes = 100):
        self.hidden_units = hidden_units
        self.hidden_act_type = hiddent_act_type
        self.num_epochs = num_epochs
        self.num_hidden_layers = num_hidden_layers
        self.model_optimizer = model_optimizer
        self.output_act_type = output_act_type
        self.output_nodes = output_nodes
        self.model_loss = model_loss

    def load_data(self, X, y, train_size, test_size):
        self.x_train = X[:train_size]
        self.x_test = X[train_size:train_size + test_size]
        self.y_train = y[:train_size]
        self.y_test = y[train_size:train_size + test_size]

    def evaluate(self):
        model = tf.keras.models.Sequential()

        # input layer
        model.add(tf.keras.layers.Dense(units=32, input_shape=(19,)))

        # 3 hidden layers with relu activation
        for i in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.hidden_units, activation=self.hidden_act_type))
            model.add(tf.keras.layers.Dropout(.22
                                              ))

        # output layer (To DO - Change this to something better)
        model.add(tf.keras.layers.Dense(self.output_nodes))

        # soft plus max test = .51
        # softmax = .51
        # sigmoid = .51

        '''
        https://keras.io/activations/
        https://keras.io/optimizers/    --documentation on supported optimizers  
        https://keras.io/losses/        --documentation on supported loss functions
        Make optimizer and loss function changeable
        '''
        model.compile(optimizer=self.model_optimizer,
                      loss=self.model_loss,
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=self.num_epochs)

        loss, acc = model.evaluate(self.x_test, self.y_test)
        print('test set loss: {} \ntest set accuracy {}'.format(loss, acc))
        return loss, acc, model

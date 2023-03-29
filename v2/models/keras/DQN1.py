import time
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import *
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


def relu_bn(inputs):
    relu1 = relu(inputs)
    bn = BatchNormalization()(relu1)
    return bn


def residual_block(x, filters, kernel_size=3):
    y = Conv2D(kernel_size=kernel_size,
               strides=(1),
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = relu(out)

    return out


def value_head(input):
    conv1 = Conv2D(kernel_size=1,
                   strides=1,
                   filters=1,
                   padding="same")(input)

    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)

    flat = Flatten()(bn1_relu)

    dense1 = Dense(256)(flat)
    dn_relu = relu(dense1)

    dense2 = Dense(256)(dn_relu)

    return dense2


def policy_head(input):
    conv1 = Conv2D(kernel_size=2,
                   strides=1,
                   filters=1,
                   padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)
    flat = Flatten()(bn1_relu)
    return flat


class Connect4NNet:
    def __init__(self, game, lr=0.001, num_channels=128, num_residual_layers=20):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.lr = lr
        self.num_channels = num_channels
        self.num_residual_layers = num_residual_layers

        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        inputs = Reshape((self.board_x, self.board_y, 1))(self.input_boards)

        bn1 = BatchNormalization()(inputs)
        conv1 = Conv2D(self.num_channels, kernel_size=3, strides=1, padding="same")(bn1)
        t = relu_bn(conv1)

        for i in range(self.num_residual_layers):
            t = residual_block(t, filters=self.num_channels)

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(policy_head(t))
        self.v = Dense(1, activation='tanh', name='v')(value_head(t))

        self.calculate_loss()

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=[self.loss_pi, self.loss_v], optimizer=Adam(self.lr))

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1, ]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)


class NNetWrapper:
    def __init__(self, game, batch_size=64, num_epochs=10):
        self.nnet = Connect4NNet(game)
        self.nnet.model.summary()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs],
                            batch_size=self.batch_size, epochs=self.num_epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=False)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

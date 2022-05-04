import argparse 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout, BatchNormalization
import os
import utils.data_util as data_util
import pickle

def generate_simple_lstm_model(input_shape, num_classes, num_units = 256):
    """
    input_shape = (embedded_word_length, max_seq_length)
    """
    inputs = tf.keras.Input(shape = input_shape)
    
    outs = CuDNNLSTM(units = num_units)(inputs) 
    outs = BatchNormalization(axis = 1)(outs)
    outs = Dense(num_classes, activation='softmax')(outs)
    
    model = Model(inputs, outs)
    return model 


def train_and_save(model, train_X, train_y, val_X, val_y, 
    num_epochs = 500, batch_size = 128, patience = 100, lr = 2e-4, to_save_file = "mdl.best.h5"):
    """
    """
    optimizer = tf.keras.optimizers.Adam(lr = lr)
    model.compile(
        loss = tf.keras.losses.categorical_crossentropy,
        optimizer = optimizer, 
        metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_acc` is no longer improving
            monitor = "val_acc",
            mode = 'max', 
            min_delta = 0, 
            # "no longer improving" being further defined as "for at least 100 epochs"
            patience = patience, 
            verbose = 1), 
        tf.keras.callbacks.ModelCheckpoint(
            to_save_file,
            monitor = 'val_acc', 
            mode = 'max', 
            verbose = 1, 
            #save_weights_only =True,
            save_best_only = True)
        ]  

    model.fit(train_X, train_y,
        batch_size = batch_size,
        epochs = num_epochs,
        verbose = 1,
        callbacks = callbacks,
        validation_data = (val_X, val_y))

    return model 
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-train", "--train_datafile", type = str)
parser.add_argument("-test", "--test_datafile", type = str)
parser.add_argument("-dst", "--dest", type = str)

args = parser.parse_args()

with open(args.train_datafile, 'rb') as f:
    train_X, train_y = pickle.load(f)

with open(args.test_datafile, 'rb') as f:
    test_X, test_y = pickle.load(f)


input_shape = train_X.shape[1:]
num_classes = 3
num_units = 256 #128

batch_size = 128
num_epochs = 500
patience = 100
lr = 2e-4

print (input_shape)
model = generate_simple_lstm_model(input_shape, num_classes, num_units = num_units)
model.summary()

os.makedirs(args.dest, exist_ok = True)
to_save_file = os.path.join(args.dest, "tweets.sa.mdl.best.h5")
print ("weights will be saved in {}".format(to_save_file))
train_and_save(model, 
    train_X, data_util.format_label(train_y, num_classes), 
    test_X, data_util.format_label(test_y, num_classes), 
    num_epochs = num_epochs, batch_size = batch_size, patience = patience, lr = lr, to_save_file = to_save_file)

saved_model = tf.keras.models.load_model(to_save_file)

# evaluate the model
_, train_acc = saved_model.evaluate(train_X, data_util.format_label(train_y, num_classes), verbose=0)
_, test_acc = saved_model.evaluate(test_X, data_util.format_label(test_y, num_classes), verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


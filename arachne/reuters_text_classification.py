"""
Train an LSTM-based model to categorise the newswires in the reuters dataset
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout, Embedding
import os
import utils.data_util as data_util
import numpy as np

def build_simple_ReusterLSTM_model(input_shape, num_labels, num_units = 128, ver = 1):
    """
    """
    inputs = tf.keras.Input(shape = input_shape)
    outs = Dropout(rate = 0.3)(outs)
    outs = CuDNNLSTM(units = num_units)(outs)  
    outs = tf.keras.layers.BatchNormalization()(outs)
    outs = Dropout(rate = 0.5)(outs)

    if ver == 1:
        outs = Dense(np.max([num_units*2, num_labels*2]), activation = 'relu')(outs)
        outs = tf.keras.layers.BatchNormalization()(outs)
        outs = Dropout(rate = 0.3)(outs)
    elif ver == 2:
        outs = Dense(np.max([num_units*4, num_labels*4]), activation = 'relu')(outs)
        outs = tf.keras.layers.BatchNormalization()(outs)
        outs = Dropout(rate = 0.5)(outs)
        outs = Dense(num_labels*2, activation = 'relu')(outs)
        outs = tf.keras.layers.BatchNormalization()(outs)
        outs = Dropout(rate = 0.5)(outs)

    outs = Dense(num_labels, activation = 'softmax')(outs)

    model = Model(inputs = inputs, outputs = outs)
    model.summary()
    sys.exit()
    return model 


def train_and_save_model(
    model, destfile, 
    train_X, train_y, val_X, val_y, 
    lr = 2e-4, num_epoch = 500, patience = 100, batch_size = 128):
    """
    """
    model.summary()
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
            destfile,
            monitor = 'val_acc', 
            mode = 'max', 
            verbose = 1, 
            save_weights_only =True,
            save_best_only = True)
        ]   

    print (train_X.shape, train_y.shape)
    print (val_X.shape, val_y.shape)

    # or batch_size = 128
    model.fit(train_X, train_y, 
        epochs = num_epoch, 
        batch_size = batch_size, 
        callbacks = callbacks, 
        verbose = 1,
        validation_data = (val_X, val_y))


if __name__ == "__main__":
    import argparse
    import pickle 

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train_datafile", type = str)
    parser.add_argument("-test", "--test_datafile", type = str)
    parser.add_argument("-dst", "--dest", type = str, 
        default = "data/models/lstm/reuters/lstm_dnn_dnn/temp")
    parser.add_argument("-only_best_eval", action = "store_true", 
        help = "if given, execute only the evaluation of $dest/reuter_lstm.h5")    
    parser.add_argument("-num_units", type = int, default = 128)

    args = parser.parse_args()

    ver = 1
    # get data
    with open(args.train_datafile, 'rb') as f:
        train_X, train_y = pickle.load(f)

    with open(args.test_datafile, 'rb') as f:
        test_X, test_y = pickle.load(f)
 
    print (train_X.shape, train_y.shape)
    print (test_X.shape, test_y.shape)

    # parameter setting
    num_labels = 46
    input_shape = train_X.shape[1:] # (300, 50  or 100 or 300)) (1st axis: time steps, 2nd axis: the number of features)
    batch_size = 128
    num_epoch = 500 #0
    patience = 100
    lr = 2e-4

    os.makedirs(args.dest, exist_ok=True)
    checkpoint_path = os.path.join(args.dest, "cp.best.ckpt") 

    model = build_simple_ReusterLSTM_model(input_shape, num_labels, num_units = args.num_units, ver = ver)
    if not args.only_best_eval:
        train_and_save_model(
            model, checkpoint_path, 
            train_X, data_util.format_label(train_y, num_labels), 
            test_X, data_util.format_label(test_y, num_labels), 
            lr = lr, num_epoch = num_epoch, patience = patience, batch_size = batch_size)

        score, acc = model.evaluate(test_X, data_util.format_label(test_y, num_labels), batch_size = batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

    destfile = os.path.join(args.dest, "reuter_lstm.h5")
    tf.keras.models.save_model(model, destfile)

    print ("evaluate with the best weights")
    model.load_weights(checkpoint_path)
    score, acc = model.evaluate(test_X, data_util.format_label(test_y, num_labels), batch_size = batch_size)
    train_score, train_acc = model.evaluate(train_X, data_util.format_label(train_y, num_labels), batch_size = batch_size)
    print ("For best:")
    print('\tTrain score:', train_score)
    print('\tTrain accuracy:', train_acc)
    print('\tTest score:', score)
    print('\tTest accuracy:', acc)
    best_mdl_destfile = os.path.join(args.dest, "reuter_lstm_best.h5")
    tf.keras.models.save_model(model, best_mdl_destfile)




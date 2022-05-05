"""
referenced:
    - https://github.com/anjanatiha/Twitter-US-Airline-Sentiment/blob/master/sentiment-analysis-with-lstm-cnn.ipynb
    - https://www.kaggle.com/loaiabdalslam/air-line-tweet-sentiment-lstm-glove
"""
import pandas as pd 
import os
import pandas as pd 
import argparse
import numpy as np
import time  
from tqdm import tqdm
# Deep Learing Preprocessing - Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

def drop_airline_mentioning(text, airlines):
    """
    tweet text preprocessing. 
    """
    text = text.lower()
    for airline in airlines:
        text = text.replace("@{}".format(airline.lower()), "")
        text = text.replace(airline, "")
    return text

def convert_to_int(label): 
    """
    """
    int_labels = {'positive':0,'neutral':1,'negative':2}
    return int_labels[label]


def decode_text(sequence, reverse_word_index):
    """
    # decode sequence to a list of words
    """
    return ' '.join([reverse_word_index.get(i, '?') for i in sequence])

def embedd_vectors(word2vec, sequence, reverse_word_index, num_features = 100):
    """
    num_features: the length of an embedded vector. 
    """
    assert num_features in [50,100], num_features # based on what I have currently

    embedded_vector = np.zeros((len(sequence), num_features))
    
    for i, word_idx in enumerate(sequence):
        word = decode_text([word_idx], reverse_word_index)
        embedded_value = word2vec.get(word)
        if embedded_value is not None:
            embedded_vector[i] = embedded_value
        else: # unknown word 
            embedded_vector[i] = word2vec.get('<OOV>')
            
    return embedded_vector

def preprocess_data(target_data, 
    glove_word2vec, 
    max_words, 
    max_seq_length, 
    reverse_word_index, 
    num_words = None,
    ratio_to_keep = None, 
    drop_airline_mention = False):
    """
    max_words -> the number of the most frequent words to keep
    max_seq_length -> the maximum length of a sequence. i.e., the maximum number of words in a sequence from the start
    """
    assert 'text' in target_data.columns and 'airline_sentiment' in target_data.columns, target_data.columns 
    assert num_words is not None or ratio_to_keep is not None
    
    if drop_airline_mention: # do I need this ...??? 
        t1 = time.time()
        airlines = [airline.lower() for airline in target_data.airline.values]
        airlines = [airline.lower() for airline in target_data.airline.values]
        target_data.text = target_data.text.apply(drop_airline_mentioning, args = (airlines,)).values[:]
        t2 = time.time()
        print ("time for droping airline mention: {}".format(t2 - t1))
    
    # tokenize & convert texts to sequences 
    if num_words is not None:
        num_words_to_keep = num_words + 1 # +1 for <OOV>
    else:
        num_words_to_keep = int(max_words * ratio_to_keep) + 1 # +1 for <OOV>
        assert num_words_to_keep > 0, num_words_to_keep
        
    print ("top {} most frequent words will be kept".format(num_words_to_keep))
    
    tokenizer = Tokenizer(num_words = num_words_to_keep, oov_token = "<OOV>") # <OOV> will have an index of num_words 
    tokenizer.fit_on_texts(target_data.text.values)
    sequences_of_texts = tokenizer.texts_to_sequences(target_data.text.values)
    
    # padd seq #
    padded_sequences_of_texts = sequence.pad_sequences(sequences_of_texts, maxlen = max_seq_length)
    
    # glove-based embedding #
    embedded_vector_arr = np.zeros(tuple(list(padded_sequences_of_texts.shape) + [max_seq_length]))
    for i, seq in enumerate(tqdm(padded_sequences_of_texts)):
        embedded_vector_arr[i] = embedd_vectors(
            glove_word2vec, seq, reverse_word_index, num_features = max_seq_length)
        
    return embedded_vector_arr


def gen_ref_dict(embedded_vector, labels, tweet_ids):
    """
    k = a string of embeded vector
    """
    embedded_vector_d2 = {}
    for i,vector in enumerate(tqdm(embedded_vector)):
        k = ",".join([str(v) for v in vector.flatten()])
        k += "," + str(labels[i])
        try:
            _ = embedded_vector_d2[k]
        except Exception:
            embedded_vector_d2[k] = []
        # i = index to the embedded vector, s
        embedded_vector_d2[k].append([i, tweet_ids[i]]) 
    return embedded_vector_d2


def get_tweet_id(target_data_vector, target_label, ref_d):
    """
    return (idx_to_complete_d, matching tweet id)
    """
    k = ",".join([str(v) for v in target_data_vector.flatten()])
    k += "," + str(target_label)
    return ref_d[k]


def get_matching_confidences(tweets_df, X_arr, y_arr, ref_d):
    """
    X_arr -> an array of embedded vector
    y_arr -> an array of labels
    ref_d -> the output of gen_ref_dict
    return (list): a list of matching confidence values
    """
    matching_confidences = []
    num = len(y_arr)
    for idx in tqdm(range(num)):
        matching_tweet_ids = get_tweet_id(X_arr[idx], y_arr[idx], ref_d)
        confs = []
        for _, matching_tweet_id in matching_tweet_ids:
            conf = tweets_df.loc[
                tweets_df.tweet_id == matching_tweet_id].airline_sentiment_confidence.values[0]
            confs.append(conf)
        matching_confidences.append(np.mean(confs))
    return matching_confidences


def get_confs_of_target_patched_and_not_patched(
    confidences, predictions, 
    target_label, pred_label, 
    by_mean = False):
    """
    true == target_label & pred == pred_label 
        - patched: new_pred == target_label
        - not_patched: new_pred == pred_label
    by_mean: take the average of all confidence values of a single iteration
    """
    patched_confs = np.array([]); not_patched_confs = np.array([])
    if isinstance(confidences, list):
        confidences = np.array(confidences)
        
    for i, preds in predictions.items():
        target_preds = preds.loc[
            (preds.true == target_label) & (preds.pred == pred_label)]
        indices_to_patched = target_preds.loc[
            target_preds.new_pred == target_label].index.values
        indices_to_not_patched = target_preds.loc[
            target_preds.new_pred != target_label].index.values
        
        if by_mean:
            if len(indices_to_patched) > 0:
                patched_confs = np.append(
                    patched_confs, np.mean(confidences[indices_to_patched]))
            if len(indices_to_not_patched) > 0:
                not_patched_confs = np.append(
                    not_patched_confs, np.mean(confidences[indices_to_not_patched]))
        else:
            patched_confs = np.append(
                patched_confs, confidences[indices_to_patched])
            not_patched_confs = np.append(
                not_patched_confs, confidences[indices_to_not_patched])
            
    return patched_confs, not_patched_confs


def eval_confidence(tweets_df, X, y, predcs, ref_d):
    """
    evaluate the confidence of patched and not-patched inputs
    """
    data_confidences = get_matching_confidences(tweets_df, X, y, ref_d)
    confs_patched = np.array([]); confs_not_patched = np.array([])
    for l in [0,1,2]: # pos, neu, neg
        vs_patched, vs_not_patched = get_confs_of_target_patched_and_not_patched(
            np.array(data_confidences), predcs, l, l)
        confs_patched = np.append(confs_patched, vs_patched)
        confs_not_patched = np.append(confs_not_patched, vs_not_patched)
    return {'patched':confs_patched, 'not_patched':confs_not_patched}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_file", "-glove", type = str, help = "glove.6B.100d.txt")
    parser.add_argument("--dest", "-dest", type = str, default = ".")
    parser.add_argument("--tweets_file", "-tweet", type=str, help = "Tweets.csv")

    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    # read tweets
    tweets_df = pd.read_csv(args.tweets_file)

    # drop airline mentioning
    print ('dropping airline mentioning')
    airlines = [airline.lower() for airline in tweets_df.airline.values]
    tweets_df.text = tweets_df.text.apply(drop_airline_mentioning, args = (airlines,)).values[:]

    # get and set the maximum number of words and reverse_word_index
    print ("tokenise")
    tokenizer = Tokenizer(lower = False)
    tokenizer.fit_on_texts(tweets_df.text.values)
    #print (tokenizer.__dict__.keys())
    reverse_word_index = tokenizer.index_word
    max_words = len(reverse_word_index)
    print ('max words: {}'.format(max_words))

    # get GloVe word2vec
    word2vec_100d = {}
    with open(args.glove_file) as f:
        for line in f.readlines():
            fields = line.split()
            word2vec_100d[fields[0]] = np.float32(np.asarray(fields[1:]))

    # set parameters related to glove-w2v embedding and also the w2v for the given glove
    max_seq_length = 100 
    word2vec = word2vec_100d 
    word2vec_arr = np.array(list(word2vec.values()))
    mean_word2v = np.mean(word2vec_arr, axis = 0)
    #set the mean of word2v to represent OOV words
    word2vec['<OOV>'] = mean_word2v

    # convert sentiments to integer: pos -> 0, neutral -> 1, negative -> 2
    print ("converting text sentiments to integer")
    int_sentiment_labels = tweets_df.airline_sentiment.apply(convert_to_int).values

    num_words = 5000
    ratio_to_keep = None

    embedded_vector_arr = preprocess_data(
        tweets_df, word2vec, 
        max_words, max_seq_length, 
        reverse_word_index,
        num_words = num_words,
        ratio_to_keep = ratio_to_keep, 
        drop_airline_mention = True) 

    # dump embedded vector & int_sentiment labels
    dumped_file = os.path.join(args.dest, "entire_data.pkl")
    with open(dumped_file, 'wb') as f:
        import pickle
        pickle.dump(
            (embedded_vector_arr, 
            int_sentiment_labels, 
            tweets_df.tweet_id.values), f)
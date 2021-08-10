import os, sys
import utils.data_util as data_util
import numpy as np

def generate_base_mdl(mdl_path, X, indices_to_target = None, target_all = True):
    from tensorflow.keras.models import load_model, Model 
    from gen_frame_graph import build_k_frame_model
    from run_localise import get_target_weights

    mdl = load_model(mdl_path)
    target_weights = get_target_weights(mdl, mdl_path, 
        indices_to_target = indices_to_target, target_all = target_all)

    k_fn_mdl, _, _  = build_k_frame_model(mdl, X, indices_to_target)
    return k_fn_mdl, target_weights


def random_sample_weights(target_ws, indices_to_target, num_sample = 1):
    """
    """
    indices = []
    for idx in indices_to_target:
        target_w = target_ws[idx][0]
        curr_indices = list(np.ndindex(target_w.shape))
        curr_indices = list(zip([idx]*len(curr_indices), curr_indices))
        
        indices.extend(curr_indices)

    selected_neural_weights = np.random.choice(indices, num_sample, replace = False)
    return selected_neural_weights


def tweak_weights(k_fn_mdl, target_weights, ys, selected_neural_weights):
    """
    """
    import tensorflow as tf

    # initial prediction
    indices_to_tls = sorted(list(target_weights.keys()))
    init_predictions, _ = k_fn_mdl([target_weights[idx][0] for idx in indices_to_tls] + [ys])
    init_corr_predictions = np.argmax(init_predictions, axis = 1)
    init_corr_predictions = init_corr_predictions == np.argmax(ys, axis = 1)

    indices_to_sel_w_tls = np.asarray([vs[0] for vs in selected_neural_weights])
    indices_to_uniq_sel_w_tls = np.unique(indices_to_sel_w_tls)
    indices_to_uniq_sel_w_tls.sort()
    indices_to_sel_ws = np.asarray([vs[1] for vs in selected_neural_weights]) 

    num_inputs = len(ys)
    prev_corr_predictons = init_corr_predictions
    by = 0.1 # starting from here
    while True:
        deltas_as_lst = []
        deltas_of_snws = {"layer":[], "w_idx":[], "value":[]}
        # update
        for idx_to_tl in indices_to_tls:
            init_weight, _ = target_weights[idx_to_tl]
            if idx_to_tl not in indices_to_uniq_sel_w_tls:
                deltas_as_lst.append(init_weight)
            else:
                w_stdev = np.std(init_weight)
                w_mean = np.mean(init_weight)
                local_indices_to_sel_nws = list(zip(*np.where(indices_to_sel_w_tls == idx_to_tl))) 
                curr_indices_to_sel_nws = indices_to_sel_ws[local_indices_to_sel_nws]

                delta = by * w_stdev * np.random.rand(init_weight.shape) 
                for idx in curr_indices_to_sel_nws:
                    init_weight[tuple(idx)] += delta[tuple(idx)]
                    deltas_of_snws['layer'].append(idx_to_tl)
                    deltas_of_snws['w_idx'].append(idx)
                    deltas_of_snws['value'].append(delta[tuple(idx)])

                deltas_as_lst.append(init_weight)

        aft_predictions, _ = k_fn_mdl(deltas_as_lst + [ys])
        aft_corr_predictions = np.argmax(aft_predictions, axis = 1)
        aft_corr_predictions = aft_corr_predictions == np.argmax(ys, axis = 1)

        # check whehter the accuracy decreases
        num_prev_corr = np.sum(prev_corr_predictons); num_aft_corr = np.sum(aft_corr_predictions)
        if num_prev_corr > num_aft_corr:
            print ("Accuracy has been decreased: {} -> {}".format(num_prev_corr/num_inputs, num_aft_corr/num_inputs))
            #return {idx_to_tl:[deltas_as_lst[idx_to_tl], idx_to_tl in indices_to_uniq_sel_w_tls] 
            #    for idx_to_tl in indices_to_tls}, num_aft_corr
            return deltas_of_snws, num_aft_corr
        else:
            if num_prev_corr > num_aft_corr: # fix 
                print ("Has been improved instead: {} -> {}".format(num_prev_corr/num_inputs, num_aft_corr/num_inputs))
            else: # num_prev == num_aft_corr (nothing has been changed)
                by += 0.1
                if by > 3:
                    print ("Out of the initial distribution: {}".format(by))
                elif by > 4.5:
                    print ("Out of the limit")
                    return None, None


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("-datadir", type = str)
    parser.add_argument("-dest", type = str)
    parser.add_argument("-which_data", type = str, help = "fashion_mnist, cifar10, GTSRB")
    parser.add_argument("-model_path", type = str)
    parser.add_argument("-target_all", type = int, default = 1)
    parser.add_argument("-seed", type = int, default = 0)

    args = parser.parse_args()

    np.random.seed(args.seed)

    train_data, test_data = data_util.load_data(args.which_data, args.datadir)

    k_fn_mdl, target_weights = generate_base_mdl(args.model_path, train_data[0], 
        indices_to_target = None, target_all = bool(args.target_all))

    num_sample = 1
    indices_to_target_layers = list(target_weights.keys())
    selected_neural_weights = random_sample_weights(target_weights, indices_to_target_layers, num_sample = num_sample)

    deltas_of_snws, num_aft_corr = tweak_weights(
        k_fn_mdl, target_weights, train_data[1], selected_neural_weights)
    
    print ("Changed Accuracy: {}".format(num_aft_corr/len(train_data[1])))
    deltas_of_snws = pd.DataFrame.from_dict(deltas_of_snws)
    print (deltas_of_snws)

    dest = os.path.join(args.dest, "{}".format(args.which_data))
    destfile = os.path.join(dest, "faulty_nws.{}.pkl".format(args.seed))
    print ("Saved to {}".format(destfile))
    
    deltas_of_snws.to_pickle(destfile)
    
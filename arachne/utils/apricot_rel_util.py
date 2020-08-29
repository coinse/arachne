"""
Apricot related methods
"""

def get_weights(model_name, start_idx = 0):
	"""
	model = keras model
	"""
	from tensorflow.keras.models import load_model
	model = load_model(model_name, compile=False)

	kernel_and_bias_pairs = []
	ws = model.get_weights()
	for i, w in enumerate(ws):
		if i >= 2*start_idx: # since we are looking at the pairs
			print (w.shape)
			if i % 2 == 0: # kernel
				kernel_and_bias_pairs.append([w])
			else: # bias
				kernel_and_bias_pairs[-1].append(w)
		else:
			continue
	
	return kernel_and_bias_pairs


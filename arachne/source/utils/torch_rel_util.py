"""
Torch relevant method
"""

def get_weights(model_name, start_idx = 0):
	"""
	model = torch model
	"""
	import torch

	m = torch.load(model_name)
	tensor_names = list(m.keys())
	
	kernel_and_bias_pairs = []
	for i,tname in enumerate(tensor_names):
		w = m[tname]
		w = w.cpu().numpy()
		if i >= 2*start_idx: # since we are looking at the pairs
			if i % 2 == 0: # kernel
				kernel_and_bias_pairs.append([w.T])
			else: # bias
				kernel_and_bias_pairs[-1].append(w)
		else:
			continue
	
	return kernel_and_bias_pairs

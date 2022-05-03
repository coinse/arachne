import pandas as pd 

def read_and_add_flag(filename):
	"""
	"""
	df = pd.read_csv(filename)
	df['flag'] = df.true == df.pred
	return df

def combine_init_aft_predcs(init_pred_df, aft_pred_df):
	"""
	"""
	# combine
	combined_df = pd.DataFrame(data = {
		'true':init_pred_df.true.values, 
		'pred':init_pred_df.pred.values,
		'new_pred':aft_pred_df.pred.values,
		'init_flag':init_pred_df.flag})

	return combined_df

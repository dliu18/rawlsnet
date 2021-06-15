import rawlsnet 
import numpy as np


data =\
{
	'adv_soc_pos_val': 'RecentPromotion',\
	'control_val': 'WorkLifeBalance',\
	'sensitive_val': ['Gender'],\
	'justified_val': ['Education'],\
	'control_input_val': [],\
	'bayes_net_exists': 'yes',\
	'cpt_exists': 'no',\
	'bn_edge_list': ["Gender", "JobSatisfaction", 
					 "Gender", "WorkLifeBalance",
					 "Gender", "RecentPromotion",
					 "Education", "WorkLifeBalance", 
					 "Education", "RecentPromotion",
					 "JobSatisfaction", "WorkLifeBalance",
					 "JobSatisfaction", "RecentPromotion",
					 "WorkLifeBalance", "RecentPromotion"],\
	'cpt_values': 'no',\
	'hashed_file': "ibm_hr",\
	'file_path': "./data/ibm_hr_feo.csv",\
	'out_path': "figs",\
	'manual_constraints': {'A': [], 'B': []},
	"constraint_val": None
}

if __name__ == "__main__": 
	rawlsnet.perturb_bayes(data)
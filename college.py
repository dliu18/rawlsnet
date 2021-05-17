import utils 
import numpy as np

talent_cpd = np.array([[0.5, 0.5]]).T
ses_cpd = np.array([[0.8, 0.2]]).T
test_cpd = np.array([[0.9, 0.1],
                   [0.1, 0.9],
                   [0.75, .25],
                   [0.05, 0.95]]).T
college_cpd = np.array([[0.9, 0.1],      # !SES, !Test
                       [0.4, 0.6],       # !SES, Test
                       [0.6, 0.4],       # SES, !Test
                       [0.15, 0.85]]).T
job_cpd = np.array([[0.9, 0.1],
                   [0.1, 0.9],
                   [0.75, .25],
                   [0.05, 0.95]]).T

data =\
{
	'adv_soc_pos_val': 'Job',\
	'control_val': 'College',\
	'sensitive_val': ['SES'],\
	'justified_val': ['Talent'],\
	'control_input_val': [],\
	'bayes_net_exists': 'yes',\
	'cpt_exists': 'yes',\
	'bn_edge_list': ["Talent", "Test", "SES", "Test", "Test", "College", "SES", "College", "College", "Job", "SES", "Job"],\
	'cpt_values': [\
					{"variable": "SES", "variable_card": "2", "evidence": [], "evidence_card": "", "values": ses_cpd},\
					{"variable": "Talent", "variable_card": "2", "evidence": [], "evidence_card": "", "values": talent_cpd},\
					{"variable": "Test", "variable_card": "2", "evidence": ["SES", "Talent"], "evidence_card": [2, 2], "values": test_cpd},\
					{"variable": "College", "variable_card": "2", "evidence": ["SES", "Test"], "evidence_card": [2,2], "values": college_cpd},\
					{"variable": "Job", "variable_card": "2", "evidence": ["SES", "College"], "evidence_card": [2,2], "values": job_cpd}\
				   ],\
	'hashed_file': "college",\
	'file_path': "",\
	'out_path': "figs",\
	'manual_constraints': {'A': [[0, 0, 1, 0], [0, 0, 0, 1]], 'B': [0.1, 0.85]}
}

if __name__ == "__main__": 
	utils.perturb_bayes(data)
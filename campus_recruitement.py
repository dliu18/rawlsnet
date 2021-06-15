import rawlsnet 
import numpy as np


data =\
{
	'adv_soc_pos_val': 'Salary',\
	'control_val': 'Internship',\
	'sensitive_val': ['Gender'],\
	'justified_val': ['SchoolPercent'],\
	'control_input_val': [],\
	'bayes_net_exists': 'yes',\
	'cpt_exists': 'no',\
	'bn_edge_list': ["Gender", "HighSchoolPercent", 
					 "Gender", "DegreePercent",
					 "Gender", "Internship",
					 "Gender", "Salary",
					 "SchoolPercent", "Internship", 
					 "SchoolPercent", "DegreePercent",
					 "SchoolPercent", "HighSchoolPercent",
					 "SchoolPercent", "EmploymentTest",
					 "DegreePercent", "Internship",
					 "HighSchoolPercent", "Internship",
					 "Internship", "EmploymentTest",
					 "Internship", "Salary",
					 "EmploymentTest", "Salary"],\
	'cpt_values': 'no',\
	'hashed_file': "campus_recruitement",\
	'file_path': "./data/campus_recruitement.csv",\
	'out_path': "figs",\
	'manual_constraints': {'A': [], 'B': []},
	"constraint_val": None
}

if __name__ == "__main__": 
	rawlsnet.perturb_bayes(data)
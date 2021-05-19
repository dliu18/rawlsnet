# RAWLSNET: Altering Bayesian Networks to Encode Rawlsian Fair Equality of Opportunity 
David Liu, Zohair Shafi, Will Fleisher, Tina Eliass-Rad, Scott Alfeld

Presented at AIES'21

[Full Paper](https://arxiv.org/abs/2104.03909)

# Reproducibility Instructions
The core implementation of RAWLSNET is included in the `rawlsnet.py` file. The parent method in this file is `perturb_bayes` which then calls all of the necessary helper functions. 

The file `college.py` reproduces Figure 2 from the paper -- the core example from the paper. Future examples and experiments can be created by writing a new main function and calling `perturb_bayes` accordingly. Users may find the comment header for `pertub_bayes` helpful. For convenience the documentation is also included below: 

```python
def perturb_bayes(data):
	# Takes as input a dictionary containing : 
	# 1. Advantageous Social Position - String
	# 2. Control Variable - String
	# 3. Sensitive Variables - List of Strings
	# 4. Justified Variables - List Of Strings
	# 5. Control Input Variable - List Of Strings 
	# 6. Bayes Net Exists - "yes" / "no"
	# 7. Edge List - Continous List Of Nodes (node i and i+1 form an edge)
	# 8. CPT Values - Numpy Array Of CPT Values
	# 9. Hashed File Name - String [Optional]
	# 10. Constraint Value - Dictionary 
	# 11. Manual Constraints - list
	# 12. CPT Exists - "yes" / "no"
	# 13. File Path - Path to csv file
	# 14. Out Path - Path to write HTML plots

	# data = 
	# {

	# 	'adv_soc_pos_val' = 'Job',
	# 	'control_val' = 'College',
	# 	'sensitive_val' = ['SES'],
	# 	'justified_val' = ['Talent'],
	# 	'control_input_val' = [],
	# 	'bayes_net_exists' = 'yes',
	# 	'cpt_exists' = 'yes',
	# 	'bn_edge_list' = ["SES", "College", "SES", "Job", "Test", "College", "College", "Job"],
	# 	'cpt_values' = [
	# 					{variable: "SES", variable_card: "2", evidence: [], evidence_card: "", values: ["0.8", "0.2"]}
	# 					{variable: "Test", variable_card: "2", evidence: [], evidence_card: "", values: ["0.5", "0.5"]}
	# 					{variable: "College", variable_card: "2", evidence: ["SES", "Test"], evidence_card: "2,2", values: ["0.8", "0.4", "0.3", "0.1", "0.2", "0.6", "0.7", "0.9"]}
	# 					{variable: "Job", variable_card: "2", evidence: ["SES", "College"], evidence_card: "2,2", values: ["0.9", "0.4", "0.3", "0.1", "0.1", "0.6", "0.7", "0.9"]}
	# 				   ]
	# 	'hashed_file' = "",
	# 	'file_path' = "/somewhere/data.csv",
	# 	'out_path' = "/thatplace/results/",
	# 	'constraint_val' = "0.5",
	# 	'manual_constraints' = {'A': [[0, 0, 1, 0], [0, 0, 0, 1]], 'B': [0.1, 0.85]}
	# }
```

After executing the main function, all figures are exported into the `/figs` directory.

All of the python dependencies are specified in `requirements.txt`.

# Implementation Notes
The implementation of RAWLSNET released here allows for categorical query variables (e.g. job) but all other variables must be binary. From a methodological perspective, categorical variables are not an issue; it simply requires additional implementation. To support categorical variables, the `generate_binary_truth` table function in `utils.py` should be replaced by a helper function that generates all permutations of the provided variables. 

# Contact 
Please direct all questions and comments to David Liu at liu.davi@northeastern.edu 
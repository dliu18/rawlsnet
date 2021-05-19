from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import cvxpy as cp

import copy

import numpy as np
import pandas as pd

import utils.bn
import utils.visualize
import utils.util

############################## Main Functions ##############################

def perturb_bayes(data):

	'''
		Takes as input a dictionary containing : 
		1. Advantageous Social Position - String
		2. Control Variable - String
		3. Sensitive Variables - List of Strings
		4. Justified Variables - List Of Strings
		5. Control Input Variable - List Of Strings 
		6. Bayes Net Exists - "yes" / "no"
		7. Edge List - Continous List Of Nodes (node i and i+1 form an edge)
		8. CPT Values - Numpy Array Of CPT Values
		9. Hashed File Name - String [Optional]
		10. Constraint Value - Dictionary 
		11. Manual Constraints - list
		12. CPT Exists - "yes" / "no"
		13. File Path - Path to csv file
		14. Out Path - Path to write HTML plots

		data = 
		{

			'adv_soc_pos_val' = 'Job',
			'control_val' = 'College',
			'sensitive_val' = ['SES'],
			'justified_val' = ['Talent'],
			'control_input_val' = [],
			'bayes_net_exists' = 'yes',
			'cpt_exists' = 'yes',
			'bn_edge_list' = ["SES", "College", "SES", "Job", "Test", "College", "College", "Job"],
			'cpt_values' = [
							{variable: "SES", variable_card: "2", evidence: [], evidence_card: "", values: ["0.8", "0.2"]}
							{variable: "Test", variable_card: "2", evidence: [], evidence_card: "", values: ["0.5", "0.5"]}
							{variable: "College", variable_card: "2", evidence: ["SES", "Test"], evidence_card: "2,2", values: ["0.8", "0.4", "0.3", "0.1", "0.2", "0.6", "0.7", "0.9"]}
							{variable: "Job", variable_card: "2", evidence: ["SES", "College"], evidence_card: "2,2", values: ["0.9", "0.4", "0.3", "0.1", "0.1", "0.6", "0.7", "0.9"]}
						   ]
			'hashed_file' = "",
			'file_path' = "/somewhere/data.csv",
			'out_path' = "/thatplace/results/",
			'constraint_val' = "0.5",
			'manual_constraints' = {'A': [[0, 0, 1, 0], [0, 0, 0, 1]], 'B': [0.1, 0.85]}

		}
	'''

	try: 

		# Read in variables from data dict
		adv_soc_pos_val = data['adv_soc_pos_val']
		control_val = data['control_val']
		sensitive_val = data['sensitive_val']
		justified_val = data['justified_val']
		control_input_val = data['control_input_val']
		bayes_net_exists = data['bayes_net_exists']
		bn_edge_list = data['bn_edge_list']
		cpt_values = data['cpt_values']
		hashed_file = data['hashed_file']
		file_path = data['file_path']
		out_path = data['out_path']

		try:
			constraint_val = float(data['constraint_val'])
		except: 
			constraint_val = None

		try: 
			cpt_exists = data['cpt_exists']
		except:
			cpt_exists = "no"

		try:
			manual_constraints = data['manual_constraints']
		except:
			manual_constraints = {"A": [], "B": []}

		cardinalities = {}

		# try: 
		# 	csv_file_name = file_path #'./user_data/' + hashed_file + ".csv" 
		# 	working_data = pd.read_csv(csv_file_name)
		
		# except Exception as e: 
		# 	print ("Error | Could Not Read CSV File | ", e)
		# 	traceback.print_exc();
		# 	return {"error_exists" : True}

		# Bayes Net Exists 
		if bayes_net_exists == "yes":

			# Create Model From Edge List
			if bn_edge_list != []:

				edge_list = []
				idx = 0
				while idx < len(bn_edge_list):
					edge_list.append(tuple(bn_edge_list[idx : idx + 2]))
					idx = idx + 2
				
				model = BayesianModel(edge_list)

			else: 
				print ("No Edge List Provided")
				return {'error_exists' : True};


			# Bayes Net Exists, But No CPT
			if cpt_exists == "no":

				# Learn CPT From Data
				model.fit(data = working_data, 
						  estimator = MaximumLikelihoodEstimator)

				cardinalities = {}
				for node in working_data.columns:
					
					cardinalities[node] = working_data[node].value_counts().values.shape[0]

			# Bayes Net And CPT Both Exist 
			if cpt_exists == "yes":

				# cpt_values structure : 
				# [
				# 	{variable: "SES", variable_card: "2", evidence: [], evidence_card: "", values: ["0.8", "0.2"]}
				# 	{variable: "Test", variable_card: "2", evidence: [], evidence_card: "", values: ["0.5", "0.5"]}
				# 	{variable: "College", variable_card: "2", evidence: ["SES", "Test"], evidence_card: "2,2", values: ["0.8", "0.4", "0.3", "0.1", "0.2", "0.6", "0.7", "0.9"]}
				# 	{variable: "Job", variable_card: "2", evidence: ["SES", "College"], evidence_card: "2,2", values: ["0.9", "0.4", "0.3", "0.1", "0.1", "0.6", "0.7", "0.9"]}
				# ]

				# Add CPTs To Model
				for i in range(len(cpt_values)):

					variable = cpt_values[i]['variable']
					variable_card = int(cpt_values[i]['variable_card'])
					cardinalities[variable] = variable_card
					
					evidence = cpt_values[i]['evidence']
					evidence_card = cpt_values[i]['evidence_card']
					if type(evidence_card) == type([]):
						if type(evidence_card[0]) != type(0):
							evidence_card = [int(x) for x in evidence_card]
					else:
						if evidence_card == '':
							evidence_card = []
						else: 
							evidence_card = evidence_card.strip(' ').strip(',').replace(' ', '').split(',')
							evidence_card = [int(x) for x in evidence_card]
					
					values = cpt_values[i]['values']
					# if evidence_card == []:
					# 	values = [x.replace('<br>', '') for x in values]
					# 	values = np.array(values).reshape([variable_card, -1])

					# else: 
					# 	values = np.array(values).reshape([variable_card, np.product(evidence_card)])


					model.add_cpds(TabularCPD(variable = variable,
											 variable_card = variable_card, 
											 values = values,
											 evidence = evidence,
											 evidence_card = evidence_card))

				# Verify CPTs Make Sense
				for i in model.get_cpds():
					print(i)

			# Form Constraints To Pass In 
			node_list = [] + justified_val 
			for node in sensitive_val:
				if node not in node_list: 
					node_list.append(node)

			card_list = []
			for node in node_list:
				card_list.append(cardinalities[node])
				
			inter_constraints = []
			for row in utils.util.generate_binary_truth_table(card_list):
				inter_dict = {}
				for i in range(len(node_list)):
					inter_dict[node_list[i]] = row[i]
				inter_constraints.append(inter_dict)
				
			constraints = []
			remaining_nodes = list(set(node_list).difference(set(justified_val)))
			card_modulus = 1
			for node in remaining_nodes:
				card_modulus = card_modulus * cardinalities[node]

			idx = 0
			while idx < len(inter_constraints):
				constraints.append(inter_constraints[idx : idx + card_modulus])
				idx = idx + card_modulus


			y_variables = list(set(model.nodes).difference(set(sensitive_val + justified_val + [adv_soc_pos_val])))
			y_card = [cardinalities[y] for y in y_variables]
			# Form Final Input
			user_input = {
				"query_node": adv_soc_pos_val,
				"control_variable": control_val,
				"y_variables": y_variables,
				"constraints": constraints,
				'y_card' : y_card,
				"manual_constraints" : manual_constraints
			}


			updated_cpd, objective = optimize_control_cpd(model, user_input, cardinalities, B_feasibility = constraint_val)
			feasibility_cpd, _ = optimize_control_cpd(model, user_input, cardinalities, B_feasibility = constraint_val)

			
			updated_cpd = np.hstack((1 - updated_cpd, updated_cpd))
			feasibility_cpd = np.hstack((1 - feasibility_cpd, feasibility_cpd))

			# Get Old CPT For Control Variable
			old_cpd = model.get_cpds(control_val).values.ravel()

			# print ("Old CPT : ")
			# print (old_cpd.shape)
			# print (old_cpd)

			if type(hashed_file) == type(None) or hashed_file == "":
				hashed_file = "output"
			if out_path[-1] == '/':
				hashed_file = out_path + hashed_file 
			else: 
				hashed_file = out_path + "/" + hashed_file

			# Get old inference object 
			inference_old = VariableElimination(model)
			utils.visualize.get_feo_plot(model = model, 
						 inference = inference_old,
						 important_nodes = node_list, 
						 query_node = adv_soc_pos_val,
						 control_val = control_val,
						 cardinalities = cardinalities,
						 hashed_file = hashed_file + "_old",
						 title = 'Existing')

			# Update CPT 
			utils.bn.update_cpd(model = model,
					   node = control_val,
					   update_vector = updated_cpd)

			# Get new inference object
			inference_new = VariableElimination(model)
			utils.visualize.get_feo_plot(model = model, 
						 inference = inference_new,
						 important_nodes = node_list,
						 query_node = adv_soc_pos_val, 
						 control_val = control_val,
						 cardinalities = cardinalities,
						 hashed_file = hashed_file + "_optimized", 
						 title = 'Updated')


			# Update CPT 
			utils.bn.update_cpd(model = model,
					   node = control_val,
					   update_vector = feasibility_cpd)

			# Get new inference object
			inference_new = VariableElimination(model)
			utils.visualize.get_feo_plot(model = model, 
						 inference = inference_new,
						 important_nodes = node_list,
						 query_node = adv_soc_pos_val, 
						 control_val = control_val,
						 cardinalities = cardinalities,
						 hashed_file = hashed_file + "_feasibility", 
						 title = 'Updated with Feasibility Constraints')
			new_cpd = model.get_cpds(control_val).values.ravel()

			parents = utils.bn.get_parents(model, control_val)
			parent_card = [cardinalities[control_val]] + [cardinalities[node] for node in parents]
			truth_table = utils.util.generate_binary_truth_table(parent_card)
			truth_table = pd.DataFrame(truth_table)
			truth_table.columns = [control_val] + parents
			truth_table['Existing Values'] = old_cpd
			truth_table['FEO Compliant Values'] = new_cpd

			print (truth_table)



			return {'error_exists' : False,
					'truth_table' : truth_table,}

		# Bayes Net Does Not Exist, Learn Structure
		else: 

			# Try Constraint Model 
			# contraint_object = PC(working_data)
			# constraint_model = contraint_object.estimate()
			# constraint_model = BayesianModel(constraint_model.edges)
			
			# Try Best Of The Exhaustive Search
			if len(cardinalities) > 6:
				print ("Cannot ennumerate all possible structures for graph with >6 nodes. Please provide a Bayes Net structure.")
				return {'error_exists' : True,
				        'model_edge_list' : None}

			exhaustive = ExhaustiveSearch(working_data)
			best_estimate = exhaustive.estimate()
			ex_model = BayesianModel(best_estimate.edges())
			
			# Also Get Top 3 Models 
			all_scores = exhaustive.all_scores()
			model_list = []
			check_limit = 100000
			model_limit = 4

			for idx, (score, model) in enumerate(all_scores): 
				if idx < check_limit and len(model_list) < model_limit:
					if utils.bn.check_model_correctness(model, control_input_val, sensitive_val, control_val):
						model_list.append(model)

			model_list = [ex_model] + model_list
			print ("Potential Models Generated")

			model_score = [np.abs(K2Score(working_data).score(model)) for model in model_list]

			model_score, model_list = (list(t) for t in zip(*sorted(zip(model_score, model_list), reverse = False)))

			model_edge_list = []
			for model in model_list:
				model_edge_list.append(list(model.edges))

			return {'error_exists' : False,
					'model_edge_list' : model_edge_list,}

	except Exception as e: 

		print ("Error | ", e)
		traceback.print_exc() 
		return {'error_exists' : True}

############################## Helper Functions ##############################
	
def z_1(model, inference, query_node, y, e):
	
	'''
	Computes intermediate variable Z_1 defined in the paper (equation 11)
	
	Inputs :
		model : pgmpy Bayesian model object
		inference : Object of type VariableElimination instantiated on the model
		query_node : String, name of node
		y : Dictionary containing values for each of the y variables
		e : Dictionary containing values for each of the evidence variables
		
	Outputs : 
		Real values number
		
	'''
	
	# parents is a list of node names
	parents = utils.bn.get_parents(model, query_node)
	
	evidence = {}
	# Each p is a node
	# p = SES or p = College
	for p in parents: 
		try:
			evidence[p] = y[p]
		except: 
			evidence[p] = e[p]
			
	return inference.query([query_node], evidence = evidence, joint = False)[query_node].values[-1]
		  
def z_2_3(model, inference, variable_list, y, e, control_variable = None):
	
	'''
	Computes intermediate variable Z_2 defined in the paper (equation 12)
	
	Inputs : 
		model : pgmpy Bayesian model object
		inference : Object of type VariableElimination instantiated on the model
		variable_list : List of nodes
		y : Dictionary containing values for each of the y variables
		e : Dictionary containing values for each of the evidence variables
		control_variable : String, name of node
		
	Outputs : 
		Real valued number
	
	'''
	
	product = 1
	
	# Used For z_3
	if control_variable != None: 
		variable_list.pop(variable_list.index(control_variable))
	
	for node in variable_list: 
		
		parents = utils.bn.get_parents(model, node)
		evidence = {}
		for p in parents:
			try:
				evidence[p] = y[p]
			except: 
				evidence[p] = e[p]
		
		try:
			node_value = y[node]
		except: 
			node_value = e[node]
			
		product = product * inference.query([node], evidence = evidence, joint = False)[node].values[node_value]
		
	return product
	
def compute_coefficients(model, inference, query_node, control_variable, y_variables, y_card, e):
	
	'''
	Computes equation 12 defined in the paper
	
	Inputs : 
		model : pgmpy Bayesian model object
		inference : Object of type VariableElimination instantiated on the model
		query_node : String, name of node
		control_variable : String, name of node
		y_variables : List of y variable names 
		e : Dictionary containing values for each of the evidence variables
		
	Outputs : 
		Dictionary of coefficients and values 
		
	Usage : 
		compute_coefficients(model = model,
							 inference = VariableElimination(model),
							 query_node = "Job",
							 control_variable = "College",
							 y_variables = ['Test', 'College'],
							 e = {'SES' : 0, 'Talent' : 0})
							 
		Out: {'f': 0.03200000000000001,
			  'Test_1_College_0_SES_0_Talent_0': 0.008000000000000002,
			  'Test_0_College_1_SES_0_Talent_0': 0.12800000000000003,
			  'Test_1_College_1_SES_0_Talent_0': 0.03200000000000001}
		
	'''
	
	# Generate List Of Dictionaries For y
	y_list = []
	# truth_table = utils.util.generate_binary_truth_table(num_variables = 2)
	truth_table = utils.util.generate_binary_truth_table(num_variables = y_card)


	for row in range(truth_table.shape[0]):
		y_dict = {}
		for idx, y_var in enumerate(y_variables):
			y_dict[y_var] = truth_table[row][idx]
		y_list.append(y_dict)
		
	# Y List : [{'Test': 0, 'College': 0},
	#           {'Test': 1, 'College': 0},
	#           {'Test': 0, 'College': 1},
	#           {'Test': 1, 'College': 1}]
	
	# Get a list of keys from the dictionary
	evidence_variables = list(e)
	
	# Calculate Alpha
	alpha = 1
	for ev in evidence_variables:
		# This assumes evidence variables are always root nodes - is this correct?
		alpha = alpha * utils.bn.get_variables(model = model, node = ev)[e[ev]]
	alpha = 1 / alpha    
	
	coefficient_dict = {}
	intercept = 0
	for itr in range(len(y_list)):
		
		y = y_list[itr]
		y_var_copy = copy.deepcopy(y_variables)
		
		coefficient = \
		z_1(model, inference, query_node, y, e) \
		* \
		z_2_3(model, inference, evidence_variables, y, e, control_variable = None) \
		* \
		z_2_3(model, inference, y_var_copy, y, e, control_variable = control_variable)
		
		coeff_name = ""
		for parent in utils.bn.get_parents(model, control_variable): 
			parent_value = y[parent] if parent in y else e[parent]
			coeff_name = coeff_name + parent + "_" + str(parent_value) + "_"
		coeff_name = coeff_name.strip('_')
		
		if coeff_name not in coefficient_dict:
			coefficient_dict[coeff_name] = 0
			
		if y[control_variable]:
			coefficient_dict[coeff_name] += coefficient * alpha
		else:
			coefficient_dict[coeff_name] -= coefficient * alpha
			intercept += coefficient*alpha
		
	return (coefficient_dict, intercept)



def enum_all_coefficients(model, control_variable, cardinalities):
	parents = utils.bn.get_parents(model, control_variable)
	parent_card = [cardinalities[x] for x in parents]
	truth_table = utils.util.generate_binary_truth_table(parent_card)
	coefficient_dict = {}
	for row in range(truth_table.shape[0]):
		coefficient = ""
		for idx, y_var in enumerate(parents):
			coefficient += y_var + "_" + str(truth_table[row][idx]) + "_"
		coefficient_dict[coefficient.strip("_")] = 0
	return coefficient_dict 

def process_coefficients(model, control_variable, coefficients, cardinalities):
	all_enumerations = enum_all_coefficients(model, control_variable, cardinalities)
	for coef in coefficients:
		all_enumerations[coef] = coefficients[coef]
	print(all_enumerations)
	return np.array(list(all_enumerations.values()))

def get_feasibility_coefficients(model, control_variable, inference, cardinalities):
	parents = utils.bn.get_parents(model, control_variable)
	ancestors = utils.bn.get_ancestors_minus_parents(model, control_variable)
	joint_distribution = inference.query(parents+ancestors, joint = True)
	feasibility_coefficients = {}
	
	parents_truth_table = []
	coefficient_list = []
	truth_table = utils.util.generate_binary_truth_table(num_variables = [cardinalities[x] for x in parents])
	for row in range(truth_table.shape[0]):
		parents_dict = {}
		coefficient = ""
		for idx, parent in enumerate(parents):
			parents_dict[parent] = truth_table[row][idx]
			coefficient+= parent + "_" + str(truth_table[row][idx]) + "_"
		parents_truth_table.append(parents_dict)
		coefficient_list.append(coefficient.strip("_"))
		
	for i in range(len(parents_truth_table)):
		coef_name = coefficient_list[i]
		parent_values = parents_truth_table[i]
		feasibility_coefficients[coef_name] = sum(joint_distribution.reduce(\
			list(parent_values.items()),\
			 inplace=False).values)
	return feasibility_coefficients

def optimize_control_cpd(model, user_input, cardinalities, B_feasibility = None):
	A = [] #coefficients for linear solver
	B = [] #intercepts
	inference = VariableElimination(model)
	for constraint in user_input["constraints"]:
		coef_set_1, intercept_1 = compute_coefficients(
				model = model,
				inference = inference,
				query_node = user_input["query_node"],
				control_variable = user_input["control_variable"],
				y_variables = user_input["y_variables"],
				y_card = user_input['y_card'],
				e = constraint[0])
		coef_set_1 = process_coefficients(model, user_input["control_variable"], coef_set_1, cardinalities)
		coef_set_2, intercept_2 = compute_coefficients(
				model = model,
				inference = inference,
				query_node = user_input["query_node"],
				control_variable = user_input["control_variable"],
				y_variables = user_input["y_variables"],
				y_card = user_input['y_card'],
				e = constraint[1])
		coef_set_2 = process_coefficients(model, user_input["control_variable"], coef_set_2, cardinalities)
		A.append(list(coef_set_1 - coef_set_2))
		B.append(intercept_2-intercept_1)

	for row in user_input['manual_constraints']["A"]:
		A.append(row)
	for row in user_input['manual_constraints']["B"]:
		B.append(row)

	# print (len(coef_set_1))
	
	A = np.array(A)
	# print ("Constraint Shapes")
	# print (A.shape)

	# Construct the problem.
	x = cp.Variable(A.shape[1])
	objective = cp.Minimize(cp.sum_squares(A * x - B))
	
	# Feasibility constraint 
	if type(B_feasibility) != type(None):

		feasibility_coef = get_feasibility_coefficients(model, user_input["control_variable"], inference, cardinalities)
		#feasibility_coef = {'SES_0_Test_0': 1, 'SES_0_Test_1': 1}
		A_feasibility = np.array(process_coefficients(model, user_input["control_variable"], feasibility_coef, cardinalities))
		B_feasibility = np.array([B_feasibility])
		print(feasibility_coef)
		constraints = [0 <= x, x <= 1, A_feasibility * x <= B_feasibility]

	else:
		constraints = [0 <= x, x <= 1]

	prob = cp.Problem(objective, constraints)

	# The optimal objective value is returned by `prob.solve()`.
	result = prob.solve()
	# The optimal value for x is stored in `x.value`.
	# print(x.value)
	# print(1-x.value)
	# print ("####")
	# print (result)

	return x.value, result


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score
from pgmpy.estimators import BayesianEstimator, ExhaustiveSearch, PC

import numpy as np
import networkx as nx

def get_heirarchy(model, node, imp_nodes):
    
    heirarchy = []
    to_check = []
    
    inter = get_parents(model, node)

    for n in inter: 
        if n not in imp_nodes:
            heirarchy.append(n)
            to_check.append(n)
    
    while to_check != []:
        
        h, c = get_heirarchy(model, to_check.pop(0), imp_nodes)
        heirarchy = heirarchy + h
        to_check = to_check + c
    
    return heirarchy, to_check
        

def get_variables(model, node):
	
	'''
	Returns CPD of a node
	
	Inputs : 
		model : pgmpy Bayesian model object
		node : String, name of node
	
	Outputs : 
		1D array of CPD values 
		
	Usage : 
		get_variables(model, "College")
		Out: array([0.9, 0.6, 0.7, 0.1, 0.1, 0.4, 0.3, 0.9])
		
	'''
	
	return np.ravel(model.get_cpds(node).values)

def get_parents(model, node):
	
	'''
	Returns list of immediate parents of a node
	
	Inputs :
		model : pgmpy Bayesian model object
		node : String, name of node
		
	Outputs : 
		List of parents of given node
		
	Usage : 
	   get_parents(model, "College")
	   Out: ['SES', 'Test']
	'''
	
	return model.get_cpds(node).scope()[1:]

def get_ancestors_minus_parents(model, node):
	'''
	Implements breadth-first search using get_parents to determine all ancestors for a node in model
	'''
	
	ancestors = []
	current_level = get_parents(model, node)
	next_level = []
	while len(current_level) > 0:
		for n in current_level:
			n_parents = get_parents(model, n)
			for parent in n_parents:
				if parent not in ancestors:
					ancestors.append(parent)
					next_level.append(parent)
		current_level = next_level
		next_level = []
	return ancestors            

def update_cpd(model, node, update_vector):
	
	'''
	Take in a vector and update the model with the updated values
	
	Inputs : 
		model : pgmpy Bayesian model object
		node : String, name of node
		update_vector : Vector to update CPD table with
		
	Outputs : 
		None
		
	Usage : 
		update_cpd(model = model, 
				   node = 'College', 
				   update_vector = update_vector)
	'''
	
	evidence = model.get_cpds(node).scope()[1:]
	variable_card = model.get_cpds(node).cardinality[0]
	update_vector = update_vector.reshape(variable_card, -1)
	evidence_card = [model.get_cpds(e).cardinality[0] for e in evidence]
	
	cpd = TabularCPD(variable = node,
					 variable_card = variable_card, 
					 values = update_vector,
					 evidence = evidence, 
					 evidence_card = evidence_card)
	
	model.add_cpds(cpd)

def check_model_correctness(model, control_input_val, sensitive_val, control_val):
	
	select_model = True

	node_count = {node : 0 for node in list(model.nodes)}

	for edge in model.edges: 
		
		node_count[edge[0]] = node_count[edge[0]] + 1
		node_count[edge[1]] = node_count[edge[1]] + 1

		# Sensitive Variables Need To Be Root Nodes
		if edge[1] in sensitive_val:
			select_model = False
	
	# Ensure Control Node Inputs
	for node in control_input_val:
		
		if (node, control_val) not in model.edges:
			select_model = False

	# Ensure No Disconnected Nodes
	for node, count in node_count.items():
		if count == 0: 
			select_model = False

	if len(list(nx.algorithms.connected_components(model.to_undirected()))) > 1:
		select_model = False
			
	return select_model
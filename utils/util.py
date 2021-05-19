import numpy as np
import itertools

def generate_binary_truth_table(num_variables):
	
	'''
	 Returns a binary truth table
	
	 Inputs : 
		 num_variables : Number of variables to consider in constructing the table.
						 Can be a single number or a list of cardinalities.
						 If input is a single number, a cardinality of 2 is assumed.
	
	 Outputs : 
		 Truth table
		
	 Usage : 
		 generate_binary_truth_table(num_variables = 2)
		 Out: array([[0, 0],
					 [1, 0],
					 [0, 1],
					 [1, 1]])
	 '''

	if type(num_variables) == type([]):
		
		if len(num_variables) == 1:
			num_variables = [2] * num_variables[0]
			
	else:
		 num_variables = [2] * int(num_variables)
		
	permute_list = []
	for card in num_variables: 
		permute_list.append([x for x in range(card)])

	truth_table = []
	for r in itertools.product(*permute_list): 
		truth_table.append(list(r))

	return np.array(truth_table)


def dict_to_string(in_dict):
    
    string = ""
    
    for k, v in in_dict.items():
        
        string = string + str(k) + " = " + str(v) + " | "
        
    return string.strip(' | ')
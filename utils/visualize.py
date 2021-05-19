import plotly.graph_objects as go

import utils.util
import utils.bn

def get_feo_plot(model, inference, important_nodes, query_node, control_val, cardinalities, hashed_file, title):
    
    imp_card = [cardinalities[n] for n in important_nodes]
    
    imp_tt = utils.util.generate_binary_truth_table(imp_card)
    evidence_list = []

    for row in range(imp_tt.shape[0]):
        evidence = {}
        for idx, node in enumerate(important_nodes):

            evidence[node] = imp_tt[row][idx]
        evidence_list.append(evidence)



    heirarchy, _ = utils.bn.get_heirarchy(model, query_node, important_nodes)
    heirarchy = [query_node] + heirarchy
    heirarchy.reverse()


    plot = {}
    for node in heirarchy:

        try:
            plot[node]
        except: 
            plot[node] = {}

        for ev in evidence_list:
            plot[node][utils.util.dict_to_string(ev)] = inference.query([node], evidence = ev, joint = False)[node].values[1]
            
            
    fig = go.Figure()
    keys = list(plot[heirarchy[0]].keys())


    for k in range(len(keys)):
        value_list = []
        for i in range(len(heirarchy)):
            value_list.append(plot[heirarchy[i]][keys[k]])

        fig.add_trace(go.Scatter(x = ['Start'] + heirarchy, y = [0.0] + value_list, name = keys[k]))


    fig.update_layout(title = "Probabilities Per Stage - " + title, 
                      xaxis_title = 'Stages', 
                      yaxis_title = 'Probabilities')
    
    fig.write_html(hashed_file + ".html")
import networkx as nx
import pandas as pd

def create_graph_with_attributes(nodes:pd.DataFrame, edgelist:pd.DataFrame) -> nx.Graph:
    '''
    This function takes the nodes and edgelist as input and creates a networkx
    graph from the edgelist. Then it appends all the relevant info on the nodes
    that are given in the nodes dataframe and adds them to the graph's nodes as
    attributes. The nodes dataframe should contain a user_id column that is the
    id of a node and it is not joined as attribute.
    '''

    # creating graph from edgelist
    graph = nx.from_pandas_edgelist(edgelist)
    
    # joining attributes to the graph's nodes
    for column in nodes.columns:
        if column != 'user_id':
            nx.set_node_attributes(graph, dict(zip(nodes.user_id, nodes[column])), column)
    
    return graph

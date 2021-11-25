import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx


def creating_plot_df(graph) -> pd.DataFrame:
    '''
    This function takes the graph as input and returns a dataframe
    containing infomration on the nodes such as degree, age, gender.
    '''
    plot_df = pd.DataFrame(dict(
        degree = dict(graph.degree()),
        age =dict(nx.get_node_attributes(graph, 'age')),
        gender = dict(nx.get_node_attributes(graph, 'gender'))))

    return plot_df


def creating_extended_df_plot(plot_df, graph):
    '''
    This function calculates three more attributes to each node:
    * neighbor connectivity
    * clustering coefficient
    * embeddedness.
    The reason for separation is the high calculation time.
    It returns the extended dataframe for plotting.
    '''

    # adding user_id column to plot_df dataframe
    plot_df = plot_df.reset_index().rename(columns={'index':'user_id'})

    # defining functions to be applied on the dataframe. They are using
    # the plot_df as given inside the function as that is the df the function
    # is called on.
    def avg_neighbors_degree(df):
        '''
        The only goal of this function is to be called in an apply structure
        on the plot_df dataframe. It calculates the avg degree of the neighbors.
        '''
        # listing neighbors for each node
        neighbors = set(graph.neighbors(df['user_id']))
        # calculating the average degree of the node's neighbors
        avg_degree = plot_df[plot_df['user_id'].isin(neighbors)].degree.mean()

        return avg_degree

    def embeddedness(df):
        '''
        The only goal of this function is to be called in an apply structure
        on the plot_df dataframe. It calculates the embeddedness of the node. 
        See formula in Dong et al. (2014) Inferring User Demographics and Social Strategies
        in Mobile Social Networks article page 3.
        '''
        # listing neighbors for each node
        neighbors_u = set(graph.neighbors(df['user_id']))
        # Iterating through the neighbors of the node and calculate the ratio of common
        # friends to union of friends of the two user.
        total = 0
        for neighbor in neighbors_u:
            total += len(neighbors_u.intersection(set(graph.neighbors(neighbor)))) / len(neighbors_u.union(set(graph.neighbors(neighbor))))

        return total / len(neighbors_u)

    # adding the three attributes to plot_df

    plot_df = plot_df.assign(neighbor_conn = plot_df.apply(avg_neighbors_degree, axis=1))
    plot_df = plot_df.assign(clustering_coeff = plot_df.user_id.map(nx.clustering(graph, plot_df.user_id)))
    plot_df = plot_df.assign(embeddedness = plot_df.apply(embeddedness, axis=1))
    
    return plot_df


def plot_graph_features(graph, feature):
    '''
    The function takes the graph as input and based on the feature argument
    it returs plots. 
    :param feature str: possible values are:
    * degree --> returns the degree distribution of the graph on a log-log plot
    * descriptive --> returns a plot with 4 subplots, namely: age~degree, 
    age~neighbor connectivity, age~triadic_closure, age~embeddedness.
    '''

    # TODO add logging module, print status of function by tqdm. 
    # getting plot_df
    plot_df = creating_plot_df(graph)

    if feature == 'degree':
        #plotting degree distribution on a log-log plot

        fig, ax = plt.subplots(figsize=(8,5))

        ax.scatter(x=np.log(plot_df.degree.value_counts().index), y=np.log(plot_df.degree.value_counts()), marker = 'x', color='b')
        # setting design
        ax.set_ylabel('degree (log)', size=14)
        ax.set_xlabel('frequency (log)', size=14)
        ax.set_title('Degree distribution on a log-log scale', size=18)
        ax.tick_params(labelsize=12)
        
        return fig

    elif feature == 'descriptive':

        # getting extended dataframe
        extended_df = creating_extended_df_plot(plot_df, graph)
        #listing the names of subplots to iterate through them
        plot_names = ['degree', 'neighbor_conn', 'clustering_coeff', 'embeddedness']
        # grouping the data by age and gender and calculating the mean
        grouped_df = extended_df.groupby(['gender', 'age']).mean().reset_index()

        # creating plot with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16,10))

        for i, subplot in enumerate(plot_names):
            for gender in range(0,2):
                axes.reshape(-1)[i].scatter(x=grouped_df[grouped_df.gender==gender].age, y=grouped_df[grouped_df.gender==gender][subplot], marker='x')
        
            # setting design
            axes.reshape(-1)[i].set_ylabel(subplot, size=14)
            axes.reshape(-1)[i].set_xlabel('age', size=14)
            axes.reshape(-1)[i].set_title(f'Average {subplot} over the ages', size=18)
            axes.reshape(-1)[i].tick_params(labelsize=12)
            axes.reshape(-1)[i].legend(set(grouped_df.gender.replace({0:'female', 1:'male'})), fontsize=12)

        return fig
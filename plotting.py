import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import logging
logging.basicConfig(level=logging.INFO)


def creating_plot_df(graph) -> pd.DataFrame:
    '''
    This function takes the graph as input and returns a dataframe
    containing infomration on the nodes such as degree, age, gender.
    '''
    # TODO rename plot_df to node_df
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
    # TODO rename plot_df to node_df

    # adding user_id column to plot_df dataframe
    plot_df = plot_df.reset_index().rename(columns={'index':'user_id'})

    # defining functions to be applied on the dataframe. They are using
    # the graph given inside the function as that is the df the function
    # is called on.

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

    plot_df = plot_df.assign(neighbor_conn = plot_df.user_id.map(nx.average_neighbor_degree(graph)))
    logging.info("neighbor connectivity added")
    plot_df = plot_df.assign(clustering_coeff = plot_df.user_id.map(nx.clustering(graph)))
    logging.info("clustering coefficient added")
    plot_df = plot_df.assign(embeddedness = plot_df.apply(embeddedness, axis=1))
    logging.info("embeddedness info added")
    
    return plot_df


def plot_graph_features(graph, feature):
    '''
    The function takes the graph as input and based on the feature argument
    it returs plots. 
    :param feature str: possible values are:
    * degree --> returns the degree distribution of the graph on a log-log plot
    * agedist --> returns a distribution plot for ages in 5-year bins.
    * descriptive --> returns a plot with 4 subplots, namely: age~degree, 
    age~neighbor connectivity, age~triadic_closure, age~embeddedness.
    '''

    # TODO combine df creation into a single function and make the df as the input
    # of this function instead of the graph and calling df creation inside.
    # TODO add layout feature to avoid overlapping in 4 subplots
    # TODO correct distplaying figures -- hide twice showings.
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

    elif feature == 'agedist':

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(data=plot_df[['age', 'gender']], x='age', hue='gender', bins=np.arange(0, 45, 5) + 15,  ax=ax)
        ax.legend(plot_df.gender.replace({0:'female', 1:'male'}), fontsize=12)
        # setting design
        ax.set_ylabel('Count', size=14)
        ax.set_xlabel('age', size=14)
        ax.set_title('Age distribution for men and women', size=18)

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


def plot_heatmap_demog_dist(nodes, edgelist):
    '''
    This function plots the demographic distribution of friends.
    TODO update description
    TODO make gender map global'''


    # using edgelist and joining relevant data on it
    edgelist_extended = edgelist.join(nodes.set_index('user_id'), on='source').rename(columns={'age': 'age_1', 'gender': 'gender_1'})
    edgelist_extended = edgelist_extended.join(nodes.set_index('user_id'), on='target').rename(columns={'age': 'age_2', 'gender': 'gender_2'})

    # setting male's age to negative
    edgelist_v2 = edgelist_extended.copy(deep=True)
    edgelist_v2.update(edgelist_v2[edgelist_v2.gender_2 ==1].age_2.apply(lambda x: -x))
    
    # creating plot
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    gender_map = {0:'female', 1:'male'}

    for i, gender in enumerate(gender_map.keys()):
        # filtering for the given gender
        df = edgelist_v2[edgelist_v2.gender_1 ==gender]
        # creating gouped dataframe with counts of relations
        grouped_df = df.groupby(['age_1', 'gender_1', 'age_2', 'gender_2']).count()
        # creating pivot table from grouped df to show the ages on the two axes,
        # the count is the vaule.
        pivot_1 = grouped_df.pivot_table(index="age_2", columns="age_1", values="source").fillna(0)
        # normalizing pivot table by sums in columns
        pivot_2 = pivot_1.apply(lambda x: x/x.sum())
        # plotting
        sns.heatmap(pivot_2, ax=axes[i], cmap='Spectral_r')
        axes[i].invert_yaxis()
        axes[i].set_ylabel('Demographic distribution of friends', size=14)
        axes[i].set_xlabel(f'age ({gender_map[gender])}', size=14)
        fig.suptitle('Age distribution of friends among men and women', size = 18)
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import logging
logging.basicConfig(level=logging.INFO)

GENDER_MAP = {0:'female', 1:'male'}


def creating_node_df(graph:nx.Graph) -> pd.DataFrame:
    '''
    This function takes the graph as input and returns a dataframe
    containing infomration on the nodes such as degree, age, gender.
    '''

    node_df = pd.DataFrame(dict(
        degree = dict(graph.degree()),
        age =dict(nx.get_node_attributes(graph, 'age')),
        gender = dict(nx.get_node_attributes(graph, 'gender'))))

    return node_df


def creating_extended_node_df(node_df:pd.DataFrame, graph:pd.DataFrame) -> pd.DataFrame:
    '''
    This function calculates three more attributes to each node:
    * neighbor connectivity
    * clustering coefficient
    * embeddedness.
    The reason for separation is the high calculation time.
    It returns the extended dataframe for plotting.
    '''

    # adding user_id column to node_df dataframe
    node_df = node_df.reset_index().rename(columns={'index':'user_id'})

    # defining functions to be applied on the dataframe. It is using
    # the graph given inside the function.

    def embeddedness(df):
        '''
        The only goal of this function is to be called in an apply structure
        on the node_df dataframe. It calculates the embeddedness of the node. 
        See formula in Dong et al.(2014) Inferring User Demographics and Social 
        Strategies in Mobile Social Networks article page 3.
        '''
        # listing neighbors for each node
        neighbors_u = set(graph.neighbors(df['user_id']))
        # Iterating through the neighbors of the node and calculate the ratio of 
        # common friends to union of friends of the two user.
        total = 0
        for neighbor in neighbors_u:
            total += len(neighbors_u.intersection(set(graph.neighbors(neighbor)))) / \
            len(neighbors_u.union(set(graph.neighbors(neighbor))))

        return total / len(neighbors_u)

    # adding the three attributes to node_df
    node_df = node_df.assign(neighbor_conn = node_df.user_id.map(nx.average_neighbor_degree(graph)))
    logging.info("neighbor connectivity added")
    node_df = node_df.assign(clustering_coeff = node_df.user_id.map(nx.clustering(graph)))
    logging.info("clustering coefficient added")
    node_df = node_df.assign(embeddedness = node_df.apply(embeddedness, axis=1))
    logging.info("embeddedness info added")
    
    return node_df


def plot_graph_features(graph:nx.Graph, feature:str):
    '''
    The function takes the graph as input and based on the feature argument
    it creates plots. 
    :param feature str: possible values are the following.
    * degree --> returns the degree distribution of the graph on a log-log plot
    * agedist --> returns a distribution plot for ages in 5-year bins.
    * descriptive --> returns a plot with 4 subplots, namely: age~degree, 
    age~neighbor connectivity, age~triadic_closure, age~embeddedness.
    '''

    # getting node_df
    plot_df = creating_node_df(graph)

    if feature == 'degree':
        #plotting degree distribution on a log-log plot
        fig, ax = plt.subplots(figsize=(8,5))

        ax.scatter(x=np.log(plot_df.degree.value_counts().index),\
        y=np.log(plot_df.degree.value_counts()), marker='x', color='b')
        # setting design
        ax.set_ylabel('degree (log)', size=14)
        ax.set_xlabel('frequency (log)', size=14)
        ax.set_title('Degree distribution on a log-log scale', size=18)
        ax.tick_params(labelsize=12)

    elif feature == 'agedist':

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(data=plot_df[['age', 'gender']], x='age', hue='gender', bins=np.arange(0, 45, 5) + 15,  ax=ax)
        ax.legend(plot_df.gender.replace(GENDER_MAP), fontsize=12)
        # setting design
        ax.set_ylabel('Count', size=14)
        ax.set_xlabel('age', size=14)
        ax.set_title('Age distribution for men and women', size=18)

    elif feature == 'descriptive':

        # getting extended dataframe
        extended_df = creating_extended_node_df(plot_df, graph)
        #listing the names of subplots to iterate through them
        subplot_names = ['degree', 'neighbor_conn', 'clustering_coeff', 'embeddedness']
        # grouping the data by age and gender and calculating the mean
        grouped_df = extended_df.groupby(['gender', 'age']).mean().reset_index()

        # creating plot with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16,10))

        for i, subplot in enumerate(subplot_names):
            # plotting both gender's series on the same subplot
            for gender in range(0,2):
                axes.reshape(-1)[i].scatter(x=grouped_df[grouped_df.gender == gender]['age'], \
                y=grouped_df[grouped_df.gender == gender][subplot], marker='x')
        
            # setting design
            axes.reshape(-1)[i].set_ylabel(subplot, size=14)
            axes.reshape(-1)[i].set_xlabel('age', size=14)
            axes.reshape(-1)[i].set_title(f'Average {subplot} over the ages', size=18)
            axes.reshape(-1)[i].tick_params(labelsize=12)
            axes.reshape(-1)[i].legend(set(grouped_df.gender.replace(GENDER_MAP)), fontsize=12)
        
        fig.tight_layout()
    
def extending_edgelist(nodes:pd.DataFrame, edgelist:pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes the nodes df and edgelist df as input and returns
    an extended version of the edgelist df with the node info (age, gender) 
    for both nodes joined on it. 
    '''

    edgelist_extended = edgelist.join(nodes.set_index('user_id'), on='source') \
    .rename(columns={'age': 'age_1', 'gender': 'gender_1'})
    edgelist_extended = edgelist_extended.join(nodes.set_index('user_id'), on='target') \
    .rename(columns={'age': 'age_2', 'gender': 'gender_2'})

    return(edgelist_extended)


def plot_heatmap_demog_dist(nodes:pd.DataFrame, edgelist:pd.DataFrame):
    '''
    This function takes the nodes and edgelist as input and plots the 
    demographic distribution of friends. Negative number is set for the males
    to create a new dimension on the charts.
    '''

    # using edgelist and joining relevant data on it
    edgelist_extended = extending_edgelist(nodes, edgelist)

    # setting male's age to negative
    edgelist_v2 = edgelist_extended.copy(deep=True)
    edgelist_v2.update(edgelist_v2[edgelist_v2.gender_2 == 1].age_2.apply(lambda x: -x))
    
    # creating plot
    fig, axes = plt.subplots(1, 2, figsize=(16,8))

    for i, gender in enumerate(GENDER_MAP.keys()):
        # filtering for the given gender
        df = edgelist_v2[edgelist_v2.gender_1 == gender]
        # creating gouped dataframe with counts of relations
        grouped_df = df.groupby(['age_1', 'gender_1', 'age_2', 'gender_2']).count()
        # creating pivot table from grouped df to show the ages on the two axes,
        # the count is the vaule.
        pivot_1 = grouped_df.pivot_table(index="age_2", columns="age_1", values="source", aggfunc=np.sum) \
        .fillna(0)
        # normalizing pivot table by sums in columns
        pivot_2 = pivot_1.apply(lambda x: x/x.sum())
        # plotting
        sns.heatmap(pivot_2, ax=axes[i], cmap='Spectral_r')
        axes[i].invert_yaxis()
        axes[i].set_ylabel('Demographic distribution of friends', size=14)
        axes[i].set_xlabel(f'age ({GENDER_MAP[gender]})', size=14)
        fig.suptitle('Age distribution of friends among men and women', size = 18)


def filtering_edgelist(edgelist:pd.DataFrame, column1:str, column2:str, \
    filter1:int, filter2:int) -> pd.DataFrame:
    '''
    This function is used to filter the edgelist into the desired outcome.
    '''

    edgelist_filtered = edgelist[(edgelist[column1] == filter1) & (edgelist[column2] == filter2)]

    return(edgelist_filtered)


def age_group_proportion(pivot:pd.DataFrame.pivot_table):
    '''
    This function takes a pivot table as input and calculates the proportion 
    of friends from the following age groups for each year that users have.
    * [user's age -5 years, user's age +5 years]
    * [user's age +20 years, user's age +30 years]
    * [user's age -30 years, user's age -20 years].
    '''
    # initializing empty dictionaries
    count_5 = {}
    count_upper_20 = {}
    count_bottom_20 = {}

    # iterating through all ages in the pivot table and summing up the number of
    # friends in the years that fall in the desired range then dividing it by the
    # total number of friends for the given age.
    for age in pivot.index:
        count_5[age] = sum(pivot.loc[age, age-5:age+5]) / pivot.loc[age].sum()
        count_upper_20[age] = sum(pivot.loc[age, age+20:age+30]) / pivot.loc[age].sum()
        count_bottom_20[age] = sum(pivot.loc[age, age-30:age-20]) / pivot.loc[age].sum()
    
    # creating dataframe from the calculated dictionaries
    ret = pd.DataFrame([count_5, count_upper_20, count_bottom_20]).T
    ret = ret.rename(columns={0:'x-5;x+5', 1: 'x+20;x+30', 2: 'x-30;x-20'})

    return(ret)


def plot_age_group_proportion(nodes:pd.DataFrame, edgelist:pd.DataFrame):
    '''
    This function plots the different age groups of friends for male and female
    users on two sublots side by side. The age groups examined are the following:
    * [user's age -5 years, user's age +5 years]
    * [user's age +20 years, user's age +30 years]
    * [user's age -30 years, user's age -20 years].
    '''

    # getting extended edgelist
    edgelist_extended = extending_edgelist(nodes, edgelist)

    # dictionary of names and gender combination for dataframes
    df_info ={'mm': [1, 1], 'mf': [1, 0], 'fm': [0, 1], 'ff': [0, 0]}

    # creating dictionary of grouped dataframes for male-male, female-female, 
    # male-female female-male relations.
    filtered_dfs = {name: filtering_edgelist(edgelist_extended, 'gender_1', 'gender_2', gender[0], gender[1]) \
    for name, gender in df_info.items()}

    # creating dictionary from grouping dataframes and returning the count of each group
    grouped_dfs = {name:filtered_dfs[name].groupby(['age_1', 'age_2']) \
    .count()[['source']] for name in df_info.keys()}

    # creating dictionary of pivot-tables from the grouped dataframes
    pivoted_dfs = {name: grouped_dfs[name].pivot_table(index="age_1", columns="age_2", \
    values="source", aggfunc=np.sum).fillna(0) for name in df_info.keys()}

    # calculating proportion of different age groups of friends for users
    # and combining the dataframes to dictionaries
    result_dfs = {name: age_group_proportion(pivoted_dfs[name]) for name in df_info.keys()}

    # combining the necessary dataframes together and plotting them
    plot_df_male = result_dfs['mm'] \
    .rename(columns={'x-5;x+5':'M x-5;x+5', 'x+20;x+30': 'M x+20;x+30', 'x-30;x-20': 'M x-30;x-20'}) \
    .join(result_dfs['mf'] \
    .rename(columns={'x-5;x+5':'F(x-5;x+5)', 'x+20;x+30': 'F(x+20;x+30)', 'x-30;x-20': 'F(x-30;x-20)'}))

    plot_df_female = result_dfs['fm'] \
    .rename(columns={'x-5;x+5':'M x-5;x+5', 'x+20;x+30': 'M x+20;x+30', 'x-30;x-20': 'M x-30;x-20'}) \
    .join(result_dfs['ff'] \
    .rename(columns={'x-5;x+5':'F(x-5;x+5)', 'x+20;x+30': 'F(x+20;x+30)', 'x-30;x-20': 'F(x-30;x-20)'}))
    
    plot_dfs = {'male': plot_df_male, 'female': plot_df_female}

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    for i, gender in enumerate(GENDER_MAP.values()):
        axes[i].plot(plot_dfs[gender])
        # setting design
        axes[i].legend(plot_dfs[gender].columns, fontsize=12)
        axes[i].set_ylabel('Proportion', size=14)
        axes[i].set_xlabel(f'Age of {gender} user', size=14)
        fig.suptitle('Proportion of friends age', size = 24)


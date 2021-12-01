import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import logging
logging.basicConfig(level=logging.INFO)

GENDER_MAP = {0:'female', 1:'male'}
# TODO return the figure in every plotting function
# TODO make a params dictionary as a global variable and use
# its values in every plotting

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
    to create a new dimension on the charts. The heatmap contains the ratio 
    of number of friends in the given age versus number of total friends.
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
        # grouping dataframe by age_1 and age_2, using count function to get 
        # the number of friendships in each age pair.
        grouped_df = df.groupby(['age_1', 'age_2']).count()
        # creating pivot table from grouped df to show the ages on the two axes,
        # the count is the vaule.
        pivot_1 = grouped_df.pivot_table(index="age_2", columns="age_1", values="source", aggfunc=np.sum) \
        .fillna(0)
        # normalizing pivot table by sums in columns
        pivot_2 = pivot_1.apply(lambda x: x/x.sum())
        # plotting
        sns.heatmap(pivot_2, ax=axes[i], cmap='Spectral_r')
        # setting design
        axes[i].invert_yaxis()
        axes[i].set_ylabel('Demographic distribution of friends', size=14)
        axes[i].set_xlabel(f'age ({GENDER_MAP[gender]})', size=14)
    
    fig.suptitle('Age distribution of friends among men and women', size = 18)

        
def plot_heatmap_pair_dist(nodes:pd.DataFrame, edgelist:pd.DataFrame):
    '''
    This function takes the nodes and edgelist as input and creates four 
    subplots about number of friendships between the ages. One for total
    one for M-M, one for F-F and one for M-F pairs The heatmap contains the
    actual number of pairs.
    '''

    # using edgelist and joining relevant data on it
    edgelist_extended = extending_edgelist(nodes, edgelist)

    # list of filters to be applied on the extended edgelist. The dictionary
    # is build up from lists of lists where the first list corresponds to the
    # first filter and the second to the second filter. 
    filters = {'all': [[0, 1], [0, 1]], 'M-M': [[1], [1]], 'F-F': [[0], [0]], 'M-F': [[1], [0]]}
    
    # creating formatter for heatmap colorbar
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,4))

    # creating plot
    fig, axes = plt.subplots(2, 2, figsize=(18,12))
    
    for i, filter in enumerate(filters.items()):
        # filtering the extended edgelist for the desired filter defined in the 
        # filters dictionary. filter[1][0] = first element of the value.
        df = filtering_edgelist(edgelist_extended, 'gender_1', 'gender_2', filter[1][0], filter[1][1], isin_method=True)
        # grouping dataframe by age_1 and age_2, using count function to get 
        # the number of friendships in each age pair. 
        grouped_df = df.groupby(['age_1', 'age_2']).count()
        # creating pivot table from grouped df to show the ages on the two axes,
        # the count is the vaule.
        pivot = grouped_df.pivot_table(index="age_2", columns="age_1", values="source", aggfunc=np.sum) \
        .fillna(0)
        # plotting
        sns.heatmap(pivot, ax=axes.reshape(-1)[i], cmap='Spectral_r', cbar_kws={'format':formatter})
        # setting design
        axes.reshape(-1)[i].invert_yaxis()
        axes.reshape(-1)[i].set_ylabel('Age', size=14)
        axes.reshape(-1)[i].set_xlabel('Age', size=14)
        axes.reshape(-1)[i].tick_params(labelsize=12)
        axes.reshape(-1)[i].set_title(f'# connections per {filter[0]} pairs', size=18)
    
    fig.suptitle('# connections per given pairs\nx-axis corresponds to the first element of pair', size=22)
    fig.tight_layout()


def filtering_edgelist(edgelist:pd.DataFrame, column1:str, column2:str, \
    filter1:int, filter2:int, isin_method=False) -> pd.DataFrame:
    '''
    This function is used to filter the edgelist into the desired outcome.
    If the isin_method is False, it the filters should be given as integers 
    and the function checks the equivalence. If the isin_method is set True,
    the filters should be given as lists and the function checks whether the
    given colums' values satisfy the condition that they are in the given list.  
    '''

    if isin_method:
        edgelist_filtered = edgelist[(edgelist[column1].isin(filter1)) & (edgelist[column2].isin(filter2))]
    else:
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
    combination_dict ={'mm': [1, 1], 'mf': [1, 0], 'fm': [0, 1], 'ff': [0, 0]}

    # creating dictionary of grouped dataframes for male-male, female-female, 
    # male-female female-male relations.
    filtered_dfs = {name: filtering_edgelist(edgelist_extended, 'gender_1', 'gender_2', gender[0], gender[1]) \
    for name, gender in combination_dict.items()}

    # creating dictionary from grouping dataframes and returning the count of each group
    grouped_dfs = {name:filtered_dfs[name].groupby(['age_1', 'age_2']) \
    .count()[['source']] for name in combination_dict.keys()}

    # creating dictionary of pivot-tables from the grouped dataframes
    pivoted_dfs = {name: grouped_dfs[name].pivot_table(index="age_1", columns="age_2", \
    values="source", aggfunc=np.sum).fillna(0) for name in combination_dict.keys()}

    # calculating proportion of different age groups of friends for users
    # and combining the dataframes to dictionaries
    result_dfs = {name: age_group_proportion(pivoted_dfs[name]) for name in combination_dict.keys()}

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


def create_triad_df(graph:nx.Graph) -> pd.DataFrame:
    '''
    This function takes the graph as input and creates a dataframe from triads
    with age and gender for each member. Then it creates a new dataframe from 
    minimum age, maximum age and sum of gender for each triad that is the
    returned dataframe.
    '''

    # getting node_df
    node_df = creating_node_df(graph)

    # findinig all cliques in the graph, listing the ones that are triads
    cliques = nx.find_cliques(graph)
    triad_list = [clique for clique in cliques if len(clique)==3]
    logging.info('Triad list obtained.')

    # creating dictionaries of lists from triad members' age and gender
    age_dict = {0: [], 1: [], 2: []}
    gender_dict = {0: [], 1: [], 2: []}

    for triad in triad_list:
        for i, element in enumerate(triad):
            age_dict[i].append(node_df.loc[element]['age'])
            gender_dict[i].append(node_df.loc[element]['gender'])
    logging.info("Dataframe from triad members' age and gender created")

    # creating dataframe from the lists combined, renaming columns
    triad_df = pd.DataFrame(age_dict) \
    .rename(columns={0: 'age_1', 1: 'age_2', 2: 'age_3'}) \
    .join(pd.DataFrame(gender_dict) \
    .rename(columns={0: 'gender_1', 1: 'gender_2', 2: 'gender_3'}))

    # creating plot_df from triad df by selecting the minimum of the ages, max
    # of ages, sum of genders (this'll be used to select triads based on genders)
    #sum_gender values meaning: 3: MMM, 2: MMF, 1: MFF, 0: FFF
    plot_df = pd.DataFrame(dict( 
    sum_gender = triad_df['gender_1'] + triad_df['gender_2']+ triad_df['gender_3'],
    min_age = triad_df[['age_1', 'age_2', 'age_3']].min(axis=1),
    max_age = triad_df[['age_1', 'age_2', 'age_3']].max(axis=1)
    ))
    logging.info('dataframe containing min age, max age, and sum of genders created.')

    return plot_df


def plot_triad_heatmap(graph:nx.Graph):
    '''
    This function's objective to create a (2,2) heatmap from triads in the graph
    with minumun age on the x axis, maximum age on the y axis. Four subplots are
    created based on the composition of the triad (FFF, MFF, MMF, MMM).
    '''
    # TODO rewrite triad listing to make it faster.
    # TODO outsource formatter function.


    # getting plot_df
    plot_df = create_triad_df(graph)
    # creating triad_info. triad_info contains info on gender 
    # composition in the triad. See called function for more details
    triad_info = {0: 'FFF', 1: 'MFF', 2: 'MMF', 3: 'MMM'}

    # creating plots
    fig, axes = plt.subplots(2, 2, figsize=(18,12))

    # creating formatter for heatmap colorbar
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,3))

    for i, triad in enumerate(triad_info.keys()):
        # filtering plot_df for the desired male-female triad
        plot_df_filt = plot_df[plot_df.sum_gender == triad]
        # grouping dataframe by min age, max age, calculating the counts
        grouped_df = plot_df_filt.groupby(['min_age', 'max_age']).count()
        # creating pivot table from the grouped df
        pivot_df = grouped_df.pivot_table(index='max_age', columns='min_age', \
        values='sum_gender', aggfunc=np.sum).fillna(0)

        # visualizing the pivoted dataframe on a heatmap
        sns.heatmap(pivot_df, ax=axes.reshape(-1)[i], cmap='Spectral_r', cbar_kws={'format':formatter})
    
        # setting design
        axes.reshape(-1)[i].set_ylabel(f'Max age of {triad_info[triad]}', size=16)
        axes.reshape(-1)[i].set_xlabel(f'Min age {triad_info[triad]}', size=16)
        axes.reshape(-1)[i].set_title(f'{triad_info[triad]}', size=20)
        axes.reshape(-1)[i].tick_params(labelsize=12)
        axes.reshape(-1)[i].invert_yaxis()
    
    fig.suptitle("Minimum and maximum age in different male-female triads", size=24)
    fig.tight_layout()

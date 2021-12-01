import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix

# setting seed to always get back the same train-test split
random.seed(10)

def train_test_split(nodes:pd.DataFrame):
    '''
    This function takes the nodes dataframe, creates a test set of the 10% of 
    the nodes saves the test sample's values in a new dataframe and sets the
    test values to np.nan. The function takes the id as given as user_id.
    '''

    # selecting random 10% of the records in the original dataframe
    test_len = int(len(nodes) / 10)

    test_ids = np.random.choice(nodes.user_id.unique(), test_len, replace=False)
    
    test_df = nodes[nodes.user_id.isin(test_ids)]
    in_sample = nodes[~nodes.user_id.isin(test_ids)]

    return in_sample, test_df


def test_duplicator(test_df:pd.DataFrame, target:list):
    '''
    This function copies the test dataframe and returns the original one and 
    a new version where the target variables are replaced by np.nan.
    '''

    empty_df = test_df.copy(deep=True)
    empty_df.loc[:,target] = np.nan

    return empty_df


def predict_from_neighbors(graph:nx.Graph, graph_nodes:pd.DataFrame, test_df:pd.DataFrame):
    '''
    This function predicts the gender og the given node by calculating the number
    of males and females among the node's neighbors. If they are equal, male is
    predicted as there are more males in the original dataset. The function uses
    the id as user_id. The return value is the prediction dataframe with the 
    user_id and predicted gender in it.
    '''

    pred = []
    # iterating through the ids in the test dataset and counting the number of
    # male and female neighbors.
    for id in tqdm(test_df.user_id):
        # listing neighbors
        neighbors = set(graph.neighbors(id))
        # filtering the train_df dataframe for the neighbors and grouping the
        # filtered one by gender and counting them. If the neighbor has no 
        # gender info, they are not used in the calculation.
        genders = graph_nodes[graph_nodes.user_id.isin(neighbors)].dropna() \
        .groupby('gender').count()
        try:
            males = genders.loc[1, 'user_id']
        except:
            males = 0
        try:
            females = genders.loc[0, 'user_id']
        except:
            females = 0

        if males >= females:
            pred.append(1)
        else:
            pred.append(0)
    

    pred_df = pd.DataFrame(dict(
        user_id = test_df.user_id,
        gender = pred
    ))

    return pred_df


def ceating_dataframe_for_prediction(graph:nx.Graph, graph_nodes:pd.DataFrame, user_ids_df=None):
    '''
    This function takes the graph and the graph_nodes dataframe as input and returns 
    a dataframe with four columns: the gender, the proportion of MM, MF, FF
    pairs in the triangles that contain the given user. The train df can contain
    the test nodes as well but without the age, in this case the returned record
    contains np.nan for the gender.
    TODO: remove [0:10000] limit, run again on the whole dataset
    '''

    if isinstance(user_ids_df, type(None)):
        user_ids_df = graph_nodes

    # finding triangles in the graph
    cliques = nx.find_cliques(graph)
    triad_list = [clique for clique in cliques if len(clique)==3]

    # initializing list for MM, MF, FF pairs
    mm = []
    mf = []
    ff = []
    
    # iterating through the user_ids given to the function
    # The value for all mm, mf, ff is 0 if there are no triads containing the 
    # given node. Triads with missing gender value are eliminated.
    for id in tqdm(user_ids_df.user_id[0:10000]):

        # Listing triads that contain the given node
        triads_id = [triad for triad in triad_list if id in triad]

        res = []
        # looping through the triads containing the given node
        for triad in triads_id:
            # initializing list to contain neighbors' gender
            neighbors_gender = []
            # looping through the elements of the triad that are not the given node
            for neighbor_id in set(triad)-set([id]): 
                # finding the gender of the given neighbor
                gender = graph_nodes[graph_nodes.user_id == neighbor_id].iloc[0, 1]
                # if the gender is missing, it is not added to the gender list
                # only relevant in case of the test df
                if not np.isnan(gender):
                    neighbors_gender.append(gender)
            
            # if the neighbors in the triad don't have gender info for both of them,
            # the triad is not added to the result and won't be counted.
            # only relevant in case of the test df
            if len(neighbors_gender) == 2:
                res.append(sum(neighbors_gender))

        # if the node has no valid triad (no triad at all or mising gender), all values
        # are going to be set 0. Otherwise mm, mf and ff represent the proportion of the
        # given triads from all triads. 
        length = len(res) if len(res) > 0 else 1
        mm.append(res.count(2)/length)
        mf.append(res.count(1)/length)
        ff.append(res.count(0)/length)

    regression_df = pd.DataFrame(dict(
        gender = user_ids_df.gender[0:10000],
        mm = mm,
        mf = mf,
        ff = ff))

    return regression_df


def check_accuracy(y_test, y_pred):
    '''
    This function returns the confusion matrix and the accuracy score of
    the two given arrays.
    '''
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)

    print(f"Confusion matrix:\n{conf_matrix}\n\nThe accuracy score is {acc_score}")
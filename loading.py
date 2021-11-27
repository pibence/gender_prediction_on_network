import pandas as pd
import numpy as np

PROFILE_COLUMS = ['user_id',
'public',
'completion_percentage',
'gender',
'region',
'last_login',
'registration',
'age',
'body',
'I_am_working_in_field',
'spoken_languages',
'hobbies',
'I_most_enjoy_good_food',
'pets',
'body_type',
'my_eyesight',
'eye_color',
'hair_color',
'hair_type',
'completed_level_of_education',
'favourite_color',
'relation_to_smoking',
'relation_to_alcohol',
'sign_in_zodiac',
'on_pokec_i_am_looking_for',
'love_is_for_me',
'relation_to_casual_sex',
'my_partner_should_be',
'marital_status',
'children',
'relation_to_children',
'I_like_movies',
'I_like_watching_movie',
'I_like_music',
'I_mostly_like_listening_to_music',
'the_idea_of_good_evening',
'I_like_specialties_from_kitchen',
'fun',
'I_am_going_to_concerts',
'my_active_sports',
'my_passive_sports',
'profession',
'I_like_books',
'life_style',
'music',
'cars',
'politics',
'relationships',
'art_culture',
'hobbies_interests',
'science_technologies',
'computers_internet',
'education',
'sport',
'movies',
'travelling',
'health',
'companies_brands',
'more',
'extra']

def preparing_edges(path):
    '''
    This function reads the file containing the edges and makes them
    unweighted. It returns an edgelist for nodes that both had directed
    edges towards each other.
    '''

    # Reading txt file with relations data
    relations = pd.read_csv(path, sep='\t', header=None, names=['source', 'target'])
    
    # changing column order, merging together with original edgelist, filtering
    # the ones that are existing in the new table twice, meaning there are
    # directed edges from both nodes to the other.
    relations_reverse = relations[['target', 'source']]
    relations_reverse.columns = ['source', 'target']
    merged = pd.concat([relations, relations_reverse])

    edges = merged.groupby(['source', 'target']).size().reset_index()
    edges.rename(columns={0: 'count'}, inplace=True)
    res = edges[edges['count'] >1]
    
    # dropping count column
    res = res.drop('count', axis=1)
    return res

def preparing_relevant_nodes(path):
    '''
    This function is preparing the nodes from the file on the given path.
    It returns the a dataframe with info (gender, age). 
    Selection criteria:
    * completion status not null
    * is public
    * age is above 14
    * gender is specified.
    '''

    # reading data from the given path
    profiles = pd.read_csv(path, sep='\t', header=None, names=PROFILE_COLUMS, usecols=['user_id', 'public', 'completion_percentage', 'gender', 'age'], index_col=None)

    # creating selection criteria
    c_public = profiles.public == 1
    c_completion = profiles.completion_percentage != 0
    c_age = profiles.age > 14
    c_gender = profiles.gender.isin([0, 1])

    # applying the criteria
    profiles = profiles[c_public & c_completion & c_age & c_gender]
    profiles.drop(['public', 'completion_percentage'], axis=1, inplace=True)

    return(profiles)


def load_data(edgelist_path, node_path):
    '''
    This function calls the above defined loading functions and returns
    the edgelist and public nodes as a tuple of dataframes. It also prints
    relevant info on the reduced graph. In order the function to work, the 
    data should be unzipped before giving it to the function as the path.
    '''

    # TODO separate a part of the dataframe for out of sample, some part for testing
    # reading file with edgelist data
    edgelist = preparing_edges(edgelist_path)

    # reading file with public node data, filtering for relevant nodes
    nodes_total = preparing_relevant_nodes(node_path)

    # filtering edgelist for public nodes only
    edgelist_public = edgelist[edgelist.source.isin(nodes_total.user_id) & edgelist.target.isin(nodes_total.user_id)]

    # getting nodes that are in the reduced edgelist
    nodes_with_edges = set(edgelist_public.source).union(set(edgelist_public.target))

    # filtering nodes dataframe for selected nodes
    nodes = nodes_total[nodes_total.user_id.isin(nodes_with_edges)]

    # printing desricptive information on the graph
    print(f"Number of nodes in the graph: {len(nodes)}\n\
Number of edges in the graph: {len(edgelist_public)}")

    return nodes, edgelist_public


def load_data_from_csv(edgelist_path, node_path):
    '''This function loads the data from the csv files.'''

    nodes = pd.read_csv(node_path)
    edgelist = pd.read_csv(edgelist_path)

    print(f"Number of nodes in the graph: {len(nodes)}\n\
Number of edges in the graph: {len(edgelist)}")

    return nodes, edgelist
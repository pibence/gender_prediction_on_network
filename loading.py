import pandas as pd


def preparing_edges(path):
    '''
    This function reads the file containing the edges and makes them
    unweighted.
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

def preparing_nodes(path):
    '''
    This function is preparing the nodes from the file on the given path.
    '''

    profile_columns = ['user_id',
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

    profiles = pd.read_csv(path, sep='\t', header=None, names=profile_columns, usecols=['user_id', 'public', 'completion_percentage', 'gender', 'age'], index_col=None)

    # removing empty and not public profiles, dropping the columns
    profiles = profiles[(profiles.completion_percentage != 0) & (profiles.public == 1)]
    profiles.drop(['public', 'completion_percentage'], axis=1, inplace=True)

    return(profiles)


def load_data(edgelist_path, node_path):
    '''
    This function calls the above defined loading functions and returns
    the edgelist and public nodes as a tuple of dataframes. It also prints
    relevant info on the reduced graph.
    '''

    # reading file with edgelist data
    edgelist = preparing_edges(edgelist_path)
    
    # getting nodes that are in the reduced edgelist
    nodes_with_edges = set(edgelist.source).intersection(set(edgelist.target))
    
    # reading file with node data, filtering for relevant nodes
    nodes_total = preparing_nodes(node_path)

    nodes = nodes_total[nodes_total.user_id.isin(nodes_with_edges)]

    # printing desricptive information on the graph
    print(f"Number of nodes in the graph: {len(nodes)}\n\
Number of edges in the graph: {len(edgelist)}")

    return nodes, edgelist

    
    # filtering for nodes that are in the edgelist
    
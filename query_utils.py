# Import asyncio, this will be needed to perform asynchronous operations
import os
import asyncio
# HTTP Requests library
import requests
from bs4 import BeautifulSoup
# Importing multiprocessing to assign operations to threadpools
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
# Importing this to create necessary directories
import pathlib
from pathlib import Path
from datetime import datetime
import re
import csv
from tqdm import tqdm
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
import json 
import itertools
import math
from collections import Counter
import numpy as np
from tabulate import tabulate
import heapq
import pandas as pd
pd.options.mode.chained_assignment = None
import csv
import heapq

import indexing_utils

def print_query_results(results_ids, with_similarity = False, similarity_dict = {}) :
    """
    Pretty prints the documents associated to [results_ids] found by query.
    if [with_similarity] == True:
        prints similarity field basing on [similarity_dict]
    """

    file_path = './tsv_dataset/'
    results_files = []
    for _id in results_ids :
        with open(f"{file_path}{_id}.tsv", 'r', encoding='utf-8') as file:
            tsv = csv.reader(file, delimiter='\t')
            rows = list(tsv)
            values = rows[2]
            result = [values[0],
                (values[10][:8] + '..') if len(values[10]) > 8 else values[8],
                values[-1]]
            if(with_similarity):
                result.append(similarity_dict[_id])
            results_files.append(result)
    
    _headers = ['Title', 'Description', 'Url', 'Similarity']
    if(with_similarity) :
        _headers.append('Similarity')
    print(tabulate( results_files, headers=_headers, tablefmt='orgtbl'))
        

def perform_query(query_string):
    """
    Takes a query string and looks for documents containing each of given words
    """
    starting_time = datetime.now()
    # load vocabulary
    with open('./vocabulary.json', 'r', encoding='utf-8') as voc_file: 
        vocabulary = json.load(voc_file)
    # Load inverted index
    with open('./indexes/synopsis_index.json', 'r', encoding='utf-8') as index_file:
        index = json.load(index_file)
    # extract query words
    query_words = indexing_utils.preprocess_string(query_string)
    query_ids = indexing_utils.associate_words_to_doc_ids(query_words, vocabulary)
    
    # returns a dictionary containing hits for every word as a key
    hits = indexing_utils.extract_documents_from_ids(query_ids, index)
    list_of_hits = hits.values()
    
    #   [ [anime_1, anime_2], [anime_2], [anime_1, anime_2] ]
    results = set.intersection(*map(set,list_of_hits))
    ending_time = datetime.now() 
    print(f"Results for \"{query_string}\" : {len(results)} | elapsed time: {(ending_time - starting_time).total_seconds()} seconds")
    print_query_results(results)
    
def generate_query_vector(query_array, vocabulary_dict) :
    """
    Generates vector based on query string
    having as 1 word in the query and 0 elsewhere
    """
    ## generate empty vector
    query_v = np.zeros(len(vocabulary_dict))
    for word in query_array : 
        query_v[list(vocabulary_dict.keys()).index(word)] = 1

    return query_v

def find_element_in_list(l, condition):
    """
    Find first element contained in a list that satisfies a condition
    """
    for el in l :
        if(condition(el)):
            return el

def generate_vector_for_given_document(document_id, vocabulary, index):
    """
    Generates vector based on document content
    having as values the tfidf factor of all words contained
    inside the document synopsis (animeDescription) and 0 elsewhere

    this looks for word ids in [vocabulary] and reads tfidf from [index]
    """
    ## generate empty vector
    doc_v = np.zeros(len(vocabulary))

    with open('./preprocessed_dataset/{}.json'.format(document_id)) as document_file:
        document = json.load(document_file)
    synopsis = document['animeDescription']
    
    for word in synopsis.split(' ') :
        try: 
            # find word id
            _id = vocabulary[word]
            tuples = index[_id]
            tfidf = 0
            # from every tuples obtained from this word inside the inverted index
            # find the one having given document_id
            for tup in tuples :
                if(tup[0] == document_id):
                    tfidf = tup[1]
                    break;
            # save tfidf score
            doc_v[list(vocabulary.keys()).index(word)] = tfidf
        except :
            raise Exception(f"Couldn't find the word {word} in our dictionary! Are you sure it's not a typo?")
    return doc_v



def query_K_top_documents(query_string, k):
    """
    Computes given query string by associating an _id to every word 
    contained in the query and finding all documents containing every word
    via the inverted index. 
    
    Then compute every document's tfidf score and find the best K results
    via cosine similarity.
    """
    starting_time = datetime.now()
    # load vocabulary
    with open('./vocabulary.json', 'r', encoding='utf-8') as voc_file: 
        vocabulary = json.load(voc_file)
    # Load inverted index
    with open('./indexes/tf_idf_synopsis_index.json', 'r', encoding='utf-8') as index_file:
        index = json.load(index_file)
    # extract query words
    query_words = indexing_utils.preprocess_string(query_string)
    query_ids = indexing_utils.associate_words_to_doc_ids(query_words, vocabulary)
    
    # query vector having as 1 words that are in query and 0 elsewhere
    query_v = generate_query_vector(query_words, vocabulary)

    # returns a dictionary containing hits for every word as a key
    hits = indexing_utils.extract_documents_from_ids(query_ids, index)
    list_of_hits = [None] * len(hits)
    hits_keys = list(hits.keys())
    for value in hits :
        for tupl in hits[value]: 
            if list_of_hits[hits_keys.index(value)] == None :
                list_of_hits[hits_keys.index(value)] = [tupl[0]]
            else :
                list_of_hits[hits_keys.index(value)].append(tupl[0])

    results = set.intersection(*map(set,list_of_hits))
    
    
    document_v_dictionary = {}
    for doc_id in list(results) :
        # query_words, document_id, vocabulary_dict, idf_index
        doc_v = generate_vector_for_given_document(doc_id, vocabulary, index )
        document_v_dictionary[doc_id] = doc_v
    
    top_k_results = compute_K_best_results_from_heap(query_v, document_v_dictionary, k)
    
    ending_time = datetime.now() 
    # top_k_results = ['anime_1', 'anime_2']
    print(f"Results for \"{query_string}\" : {len(results)} | elapsed time: {(ending_time - starting_time).total_seconds()} seconds")
    print_query_results(top_k_results.keys(), True, top_k_results)
    
    

def compute_K_best_results_from_heap(query_vector, document_vectors_dict, k) :
    """
    Build a maxheap with the scores of the cosine similarity between
    [query_vector] and document vectors contained in [document_vectors_dict], 
    then compute the best K results and return them.
    """
    output = {}
    # create a score heap structure and a dictionary of scores
    heap = list()
    heapq.heapify(heap)
    scores_dictionary = dict()
    
    for key,value in document_vectors_dict.items() :
        # Compute the cosine of the angle between the query vector and the document vector
        cos = np.dot(query_vector, value)/(np.linalg.norm(query_vector)*np.linalg.norm(value))
        scores_dictionary[cos] = key
        # Update the heap
        heapq.heappush(heap, cos)
    
    # find the k best scores
    top_k = heapq.nlargest(k, heap)
    for score in top_k :
        # find the doc associated to the score
        output[scores_dictionary[score]] = score
    
    return output


def print_resulting_dataframe(dataframe) :
    """
    Pretty prints dataframe
    """
    dataframe['animeDescription'] = dataframe['animeDescription'].apply(lambda string : string[:8] + '...')
    print(tabulate(dataframe, headers='keys', tablefmt='psql'))
        

def compute_query_on_jaccard_similarity(query_string, k, num_epidode_weight, date_weight, members_weight, score_weight, popularity_weight) :
    """
    Computes given query string by associating an _id to every word 
    contained in the query and finding all documents containing every word
    via the inverted index. 
    
    compute every document's jaccard similarity score and find the best K results
    after adapting similarity scores to given [weights] provided by the user.
    """
    
    starting_time = datetime.now()
    output_dataframe = pd.DataFrame(columns=['animeTitle', 'animeDescription', 'url', 'similarity'])
    # load vocabulary
    with open('./vocabulary.json', 'r', encoding='utf-8') as voc_file: 
        vocabulary = json.load(voc_file)
    # Load inverted index
    with open('./indexes/tf_idf_synopsis_index.json', 'r', encoding='utf-8') as index_file:
        index = json.load(index_file)
    # extract query words
    query_words = indexing_utils.preprocess_string(query_string)
    query_ids = indexing_utils.associate_words_to_doc_ids(query_words, vocabulary)
    
    # query vector having as 1 words that are in query and 0 elsewhere
    query_v = generate_query_vector(query_words, vocabulary)


    
    # returns a dictionary containing hits for every word as a key
    hits = indexing_utils.extract_documents_from_ids(query_ids, index)
    list_of_hits = [None] * len(hits)
    hits_keys = list(hits.keys())
    # reduce hits to a document _ids array
    for value in hits :
        for tupl in hits[value]: 
            if list_of_hits[hits_keys.index(value)] == None :
                list_of_hits[hits_keys.index(value)] = [tupl[0]]
            else :
                list_of_hits[hits_keys.index(value)].append(tupl[0])

    # find interception
    results = set.intersection(*map(set,list_of_hits))
    
    
    # compute dataframe 
    dataframe = compute_pandas_dataframe_for_results(results)
    
    # calculate similarity on Title, Description, Charachters
    jaccard_scores = compute_jaccard_scores(dataframe, query_words)
    
    # num_epidode_weight, date_weight, members_weight, score_weight, popularity_weight 
    
    dataframe['releaseDate'] =  dataframe['releaseDate'].apply(lambda d : datetime.strptime(d, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    #dataframe['releaseDate'] = pd.to_datetime(dataframe['releaseDate'], unit='ms')
    
    # compute sequent fields scores adapting them to user inputs
    date_scores = compute_column_scores(dataframe, 'releaseDate', 'dateScores', date_weight)
    num_ep_scores = compute_column_scores(dataframe, 'animeNumEpisode', 'numEpScores', num_epidode_weight)
    anime_members_scores = compute_column_scores(dataframe, 'animeNumMembers', 'memberScores', members_weight)
    anime_rate_scores = compute_column_scores(dataframe, 'animeScore', 'ratingScore', score_weight)
    anime_popularity_score = compute_column_scores(dataframe, 'animePopularity', 'popularityScore', popularity_weight)
    
    # compute the total similarity score of every document by 
    # mixing up both the jaccard score and the weights adapted
    # by the user preferences
    for index, row in jaccard_scores.iterrows() :
        overall_similarity = row['similarity'] + date_scores['dateScores'][index] + num_ep_scores['numEpScores'][index] + anime_members_scores['memberScores'][index] + anime_rate_scores['ratingScore'][index]+ anime_popularity_score['popularityScore'][index]
        temp_row = pd.DataFrame(data = {'animeTitle' : [dataframe['animeTitle'][index]],'animeDescription' : [dataframe['animeDescription'][index]], 'url' : [dataframe['url'][index]], 'similarity':[overall_similarity]})   
        output_dataframe = output_dataframe.append(temp_row)

    # normalize similarities finding a top one
    norm_factor = output_dataframe['similarity'].max()
    output_dataframe['similarity'] = output_dataframe['similarity'].apply(lambda x : x / norm_factor)
    
    # compute the top k result
    top_k_results = compute_top_k_dataset(output_dataframe, k)
    
    print_resulting_dataframe(top_k_results)


def compute_pandas_dataframe_for_results(results_ids):
    """
    Builds up a pandas dataframe containing every document associated with [results_ids]
    """
    documents_dataframe = pd.DataFrame(data = {
        "animeTitle": [],
        "animeType": [],
        "animeNumEpisode": [],
        "releaseDate": [],
        "endDate": [],
        "animeNumMembers": [],
        "animeScore": [],
        "animeUsers": [],
        "animeRank": [],
        "animePopularity": [],
        "animeDescription": [],
        "animeRelated": [],
        "animeCharacters": [],
        "animeVoices": [],
        "animeStaff": [],
        "url": [],
        "similarity": []
    })
    anime_files = []
    
    # 1 : read docs
    for _id in results_ids :
        with open('./tsv_dataset/{}.tsv'.format(_id), 'r', encoding='utf-8') as document_file:
            tsv = csv.reader(document_file, delimiter = '\t')
            # 2 : add docs to dataset
            r = next(tsv)
            r = next(tsv)
            tsv_data = next(tsv)
            
            temp = pd.DataFrame(data = {
                "animeTitle" : [tsv_data[0]],
                "animeType" : [tsv_data[1]],
                "animeNumEpisode" : [tsv_data[2]],
                "releaseDate" :[ tsv_data[3]],
                "endDate" : [tsv_data[4]],
                "animeNumMembers" : [tsv_data[5]],
                "animeScore" : [tsv_data[6]],
                "animeUsers" : [tsv_data[7]],
                "animeRank" : [tsv_data[8]],
                "animePopularity" : [tsv_data[9]],
                "animeDescription" : [tsv_data[10]],
                "animeRelated" :[ tsv_data[11]],
                "animeCharacters" : [tsv_data[12]],
                "animeVoices" : [tsv_data[13]],
                "animeStaff" : [tsv_data[14]],
                "url" :[tsv_data[15]],
                "similarity" : [0]
            })
            # concats found document to output
            documents_dataframe = pd.concat([documents_dataframe, temp], sort = True)

    return documents_dataframe.reset_index(drop=True)

def jaccard_similarity(query_array, text_array):
    """
    Compute jaccard similarity on given query and text
    """
    query = set(query_array)
    text = set(text_array)
    return len(query.intersection(text)) / len(query.union(text))


def normalize_jaccard(values, min_jac, max_jac):
    """
    Normalize jaccard based on min and max
    """
    return [(x - min_jac)  / (max_jac - min_jac) for x in values]

def compute_jaccard_scores(documents_dataframe, query_text):
    """
    Computes the jaccard scores similarity between
    every relevant field in a given document and
    the given query_text
    """
    jac_dataframe = documents_dataframe[["animeTitle","animeDescription","animeRelated","animeCharacters","similarity"]]
    
    temp_scores = []
    jac_scores = []
    
    cols = jac_dataframe.columns
    
    # calculate jaccart scores
    for index, row in jac_dataframe.iterrows():
        score = []
        for col in cols:
            text = str(row[col])
            # append the score of every variable of a given document
            # animeTitle : score, animeDesc : score... exc...
            temp_scores.append(jaccard_similarity(query_text, indexing_utils.preprocess_string(text)))
        
        # sum up all the scores of a given document into a single score
        jac_scores.append(sum(temp_scores))
    
    if(len(jac_scores) > 1) :
        jac_scores = normalize_jaccard(jac_scores, min(jac_scores), max(jac_scores))
    
    # normalize jaccard 
    jac_dataframe['similarity'] = list(map(lambda x: round(x, 2), jac_scores))
    return jac_dataframe[["animeTitle", "animeDescription", "similarity"]]

def compute_top_k_dataset(input_dataset, k) :
    '''
    This function takes a pandas dataframe having:
    animeTitle | animeDescription | url | similarity
    as fields.
    It heap sorts the dataframe's rows based on similarity and returns
    k best elements
    '''
    score_to_row_dictionary = {}
    output = pd.DataFrame(columns = input_dataset.columns)
    heap = list()
    heapq.heapify(heap)
    
    for index, row in input_dataset.iterrows() :
        score_to_row_dictionary[row['similarity']] = row
        heapq.heappush(heap, row['similarity'])
    
    top_k_scores = heapq.nlargest(k, heap)
    
    for score in top_k_scores :
        output = output.append([score_to_row_dictionary[score]])
    
    return output

def convert_str_to_float(string):
    """
    floats given string, if this is not possible return 0
    """
    try:
        return float(string)
    except:
        return 0

def compute_column_scores(dataframe, source_column, output_column, output_weight) :
    '''
    This takes a [dataframe] as an input and returns a series of scores
    of a given [source_column] based on [output_weight], named as [output_column]
    '''
    result = pd.DataFrame(columns=[output_column])
    source = dataframe[source_column].apply(lambda x : convert_str_to_float(x))
    
    norm_factor = source.max()
    result[output_column] = source.apply(lambda x: (x / norm_factor) * output_weight)
    return result


def take_simple_query_from_user():
    """
    Ask the user to perform a simple query.
    """
    try:
        print("Hi! Type in your query : ")
        query = str(input())
        perform_query(query)
    except Exception as e :
        print(e)

def take_top_k_of_query_from_user() :
    """
    Ask the user to perform a simple query and the amount of results to return.
    this computes the best k results based on tfidf scores
    """
    try:
        print("Hi! Type in your query : ")
        query = str(input())
        print("How many results would you like to retrieve?")
        k = int(input())
        query_K_top_documents(query, k)
    except Exception as e :
        print(e)
        
def take_biased_query_from_user():
    """
    Ask the user to perform a complex query by asking him a query string
    and a series to preferences that will adapt the weights of every asked column
    on his preferences. 
    this computes the best k results adapting jaccard similarity scores to the user preferences
    """
    try:
        print("Hi! Type in your query : ")
        query = str(input())
        print("How many results would you like to retrieve?")
        k = int(input())
        print("You will now be asked to type in some preferences. \nConsider the range from -5 to 5, where -5 is the least, 5 is the max and 0 is indifferent.\n")
        print("How many episodes do you prefer?")
        ep_weight = int(input())/10
        print("Are you looking for a newer or an older anime?")
        date_weight = int(input())/10
        print("Are you looking for an anime with a large fanbase?")
        members_weight = int(input())/10
        print("Are you looking for an anime with a good score?")
        score_weight = int(input())/10
        print("Are you looking for a very popular anime?")
        popularity_weight = int(input())/10

        compute_query_on_jaccard_similarity(query, k, ep_weight, date_weight, members_weight, score_weight, popularity_weight)
    except Exception as e :
        print(e)
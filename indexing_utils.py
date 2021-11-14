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



            
            
def preprocess_string(input_string, stemmer = EnglishStemmer(), tokenizer = word_tokenize ) :
    """
    Takes in an input string and preprocess it by sequent steps:   
        - Removing Punctuation
        - Removing Stopwords
        - Stemming
    """

    # Base case
    if not input_string :
        return ''
    # define stopwords
    stop_words = set(stopwords.words('english'))
    # define punctuation
    punctuation = string.punctuation
    translation_table = str.maketrans('', '', punctuation)
    output = []

    for token in [t.lower() for t in tokenizer(input_string)]:
        # remove punctuation
        token = token.translate(translation_table)
        try :
            if token == '' or token in stop_words:
              # If token is a stopword, go on to the next word
              continue
        except Exception as e:
            print(f"{e} thrown while using stop_words")
        
        # If token was not a stopword, stem it after having removed punctuation
        if stemmer:
            token = stemmer.stem(token)
        
        # add the processed word to the output
        output.append(token)
        
    return output
      
def preprocess_tsv(file_path):
    """
    Takes in the file_path of an input tsv and preprocess
    all of its values except dates and url by sequent steps:   
        - Removing Punctuation
        - Removing Stopwords
        - Stemming
    
    This makes use of [preprocess_string] function.
    """
    try:
        # read file
        file_name = str(file_path).split('\\')[-1]
        with open(file_path, 'r', newline='', encoding="utf-8") as f:
            # make output dir if nonexistant
            Path("./preprocessed_dataset").mkdir(parents=True, exist_ok=True)
            # preprare output dictionary
            output = {}
            # read tsv
            tsv = csv.reader(f, delimiter='\t')
            # skip labels row
            columns = next(tsv)
            # skip blank row
            next(tsv)
            # read data
            data = next(tsv)
            # preprocess data
            for i in range(len(columns)) :
                if(columns[i] not in ['releaseDate', 'endDate', 'url']) :
                    output[columns[i]] = ' '.join(preprocess_string(data[i]))
                else :
                    output[columns[i]] = data[i]
            # dump all preprocessed informations inside a json 
            with open('./preprocessed_dataset/{}.json'.format(file_name.split('\\')[-1].split('.')[0]), 'w', encoding="utf-8") as out_file:
                json.dump(output, out_file)

    except Exception as e:
        print("Error on file {} : {}".format(file_path, e))
            
def map_input_file_into_dictionary(dictionary, file_path):
    """
    Reads a file from a [file_path] and maps each of its word inside a dictionary.
    """
    f = open(file_path, 'r', encoding='utf-8')
    data = json.load(f)
    values = ' '.join(data.values()).split(' ')
    # sort words for better performance
    values.sort()
    for word in values :
        # if word is not in the dictionary, map it.
        if(word not in dictionary) :
            # make up an id from the word hashcode
            _id = str(hash(word))
            dictionary[word] = _id
                
def hydrate_vocabulary_from_files(file_paths, cores_amount):
    """
    Multiprocess, based on [cores_amount] reading all files
    relative to [file_paths] and mapping their content inside a dictionary.
    """
    pool = ThreadPool(cores_amount)
    with open('./vocabulary.json', 'w+', encoding='utf-8') as dict_file :
        # our JSON dictionaries
        dictionary = json.load(dict_file) if os.stat("./vocabulary.json").st_size > 0 else {}
        pool.map(lambda path : map_input_file_into_dictionary(dictionary, path), file_paths )
        # hydrate dictionary
        json.dump(dictionary, dict_file)

        
        
def map_input_to_tf_dictionary_given_vocabulary(file_id, input_words, output_dict,
                                                vocabulary_dictionary,
                                                tot_documents, idf_source_dictionary) : 
    """
    Takes informations of a file as input and then maps its content inside a tfidf dictionary
    that will be used as an inverted index, saving file_id and tfidf score inside each record.

    For every word, read its id from the voucabulary and map it to output_dict.
    """
    
    # sorting content to speed up the process
    input_words.sort()
    '''
    for each processed word, find its id via vocabulary and
    update its reference into the output_dict
    '''

    #------------processing input words-------------
    # read input as a counter, speeding up the tf process
    words_counter = Counter(input_words)
    
    for word in words_counter.keys() : 
        _id = vocabulary_dictionary[word]
        tf = words_counter[word] / len(input_words)
        # # of docs that contain this word
        occurencies = len(idf_source_dictionary[_id])
        # idf is... log (len(index) / len(# of documents that contain the word)
        idf = math.log( tot_documents / occurencies)
        # save record in the output dictionary
        if(_id not in output_dict) :
            output_dict[_id] = [(file_id, tf * idf)]
        else :
            if file_id not in output_dict[_id] : 
                output_dict[_id].append( (file_id, tf * idf) )


def map_input_to_output_dictionary_given_vocabulary(file_id, input_words, output_dict, vocabulary_dictionary) : 
    """
    Read input_words and assigns an id to each of them to then
    map the file they belong to via file_id inside output_dict.
    """
    
    # sorting content to speed up the process
    input_words.sort()
    '''
    for each processed word, find its id via vocabulary and
    update its reference into the output_dict
    '''
    #------------processing input words-------------
    for word in input_words : 
        _id = vocabulary_dictionary[word]
        if(_id not in output_dict) :
            output_dict[_id] = [file_id]
        else :
            if file_id not in output_dict[_id] : 
                output_dict[_id].append(file_id)
    #------------finished processing input-------------
        
def process_file_synopsis_to_output_given_vocabulary(file_path, output_dictionary, vocabulary_dictionary, want_idf, tot_documents, idf_source_dictionary):
    """
    Reads file from [file_path] and processes every word inside his synopsis (animeDescription)
    field; then maps it to the [output_dictionary] via given [vocabulary_dictionary]

    IF [want_idf] is True:
        this function will map the word to the tfidf dictionary,
        basing its computation on [tot_documents] and [idf_source_dictionary]

        SO: [tot_documents] and [idf_source_dictionary] MUST be provided if [want_idf] == True
    """
    #------------started processing synopsis-------------
    input_file = open(file_path, 'r', encoding='utf-8')
    input_file_id = str(file_path).split('\\')[-1].split('.')[0]
    #------------defined file_id {input_file_id}-------------
    input_dict = json.load(input_file)
     
    '''
    Since we'll be working only on Synopsis for this index,
    we'll extract its value from the starting file
    '''
    content_to_map = input_dict['animeDescription'].split(' ')
    #print('2------------mapping content ------------------')
    if(want_idf == False) :
        map_input_to_output_dictionary_given_vocabulary(input_file_id, content_to_map, output_dictionary, vocabulary_dictionary)
    else :
        map_input_to_tf_dictionary_given_vocabulary(input_file_id, content_to_map, output_dictionary, vocabulary_dictionary, tot_documents, idf_source_dictionary)
    
def hydrate_synopsys_inverted_index_with_given_files_and_vocabulary(
    inverted_index_path, file_paths, vocabulary_path, cores_amount, want_idf = False, tot_documents = 0, idf_source_dictionary_path='') :
    
    """
    Multiprocess, based on cores_amount, the creation of a new Inverted Index
    in [inverted_index_path] and its hydratation it via every file read from [file_paths]. 
    This assigns an _id to every word via a vocabulary read from [vocabulary_path]. 

    IF [want_idf] is True:
        this function will create a tfidf inverted index,
        basing its computation on [tot_documents] and [idf_source_dictionary]

        SO: [tot_documents] and [idf_source_dictionary] MUST be provided if [want_idf] == True
    """
    
    pool = ThreadPool(cores_amount)
    print('1------------starting hydrating inverted index-------------')
    try:
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            print('1------------opened inverted index file--------------------')
            inverted_index = json.load(f)
    except FileNotFoundError:
        print('1------------inverted index not yet existant---------------')
        inverted_index = {}
    with open(vocabulary_path, 'r', encoding='utf-8') as voc:
        print('1------------reading vocabulary----------------------------')
        vocabulary = json.load(voc)
    idf_source_dictionary = {}
    if(want_idf) :
        with open(idf_source_dictionary_path, 'r', encoding='utf-8') as idf_source_dict :
            idf_source_dictionary = json.load(idf_source_dict)
    print('1------------proceeding to multiprocess inputs-------------')
    result = pool.map(lambda path : process_file_synopsis_to_output_given_vocabulary(
        path, inverted_index, vocabulary, want_idf, tot_documents, idf_source_dictionary
    ), file_paths)
    print('1------------finished to multiprocess inputs---------------')
    
    print('1------------finished hydrating inverted index-------------')
    print('1------------dumping inverted index -----------------------')
    with open(inverted_index_path, 'w', encoding='utf-8') as index_file : 
        json.dump(inverted_index, index_file)
    
    print('1------------dumped inverted index to path-----------------')
    print(inverted_index_path)
        

def all_equal(iterable):
    """
    checks if all elements in an iterable are equal
    """
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

def associate_words_to_doc_ids(words, vocabulary):
    """
    Reads word id from vocabulary
    """
    output = []
    for word in words :
        try:
            output.append(vocabulary[word])
        except :
            raise Exception(f"Couldn't find the word {word} in our dictionary! Are you sure it's not a typo?")
    return output

def extract_documents_from_ids(ids, index) :
    """
    Extracts all the documents that contain a word with id from an index
    """
    results = {}
    for _id in ids :
        docs = index[_id]
        results[_id] = docs
    return results


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
    query_words = preprocess_string(query_string)
    query_ids = associate_words_to_doc_ids(query_words, vocabulary)
    
    # returns a dictionary containing hits for every word as a key
    hits = extract_documents_from_ids(query_ids, index)
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
    query_words = preprocess_string(query_string)
    query_ids = associate_words_to_doc_ids(query_words, vocabulary)
    
    # query vector having as 1 words that are in query and 0 elsewhere
    query_v = generate_query_vector(query_words, vocabulary)

    # returns a dictionary containing hits for every word as a key
    hits = extract_documents_from_ids(query_ids, index)
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
    query_words = preprocess_string(query_string)
    query_ids = associate_words_to_doc_ids(query_words, vocabulary)
    
    # query vector having as 1 words that are in query and 0 elsewhere
    query_v = generate_query_vector(query_words, vocabulary)


    
    # returns a dictionary containing hits for every word as a key
    hits = extract_documents_from_ids(query_ids, index)
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
            temp_scores.append(jaccard_similarity(query_text, preprocess_string(text)))
        
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
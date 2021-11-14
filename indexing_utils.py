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
    Multiprocess, based on cores_amount, the creation of a new Inverted Index\n
    in [inverted_index_path] and its hydratation it via every file read from [file_paths].\n
    This assigns an _id to every word via a vocabulary read from [vocabulary_path].\n

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

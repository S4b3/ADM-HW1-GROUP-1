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



'''
This function performs an HTTP Get Request to MyAnimeList and places its results in a given array.
Params: 
    [index] : Simply the page index. Sets up the url for pagination and defines where the page will be placed inside [destination_array]
    [destination_array] : where the retrieved page will be stored. The result will be placed in index [index]
'''
def fetch_page(index, destination_array):
    destination_array[index] = requests.get(f"https://myanimelist.net/topanime.php{'?limit={}'.format(50*index) if(index > 0) else ''}")
    
'''
Finds all URL contained in a MyAnimeList top animes page, then substitutes them to the starting page inside [pages] array.
Params: 
    [page]  : MyAnimeList's Top Animes HTML Page
    [pages] : Array containing all the pages. 
'''
def fetch_urls_in_page(page, pages):
    # Defining an html parser
    soup = BeautifulSoup(page.content, "html.parser")
    # Find all URLs
    animeUrls = soup.find_all("a", class_="hoverinfo_trigger fl-l ml12 mr8", id=lambda x: x and x.startswith('#area'), href=True)
    animeUrls = [a['href'] for a in animeUrls]
    # Substitues starting page with its URLs
    pages[pages.index(page)] = animeUrls

    
'''
Performs a GET Request on a given [url] and saves its results as an HTML inside a folder called "page_[folder]".
The HTML file will be named "article_[index].html"
'''
def fetch_anime_and_parse_html(url, folder, index):
    # Get current page
    req = requests.get(url)
    # MyAnimeList might refuse to respond to large amount of requests, if this happens, we need to stop the process
    if(req.status_code != 200) : 
        raise Exception(f"My anime list has closed the connection.\nComplete the captcha and restart the process.\nCurrent Page was : {index}")
    # Define page's absolute destination path
    _directory_path = f"{pathlib.Path().resolve()}/dataset/page_{folder}"
    # Check if current page's destination folder exists... if not, create it!
    Path(_directory_path).mkdir(parents=True, exist_ok=True)
    # Write the html file in the destination directory.
    with open(f"{_directory_path}/article_{index}.html", 'w') as file:
        file.write(req.text)
    

'''
Assigns fetching to all available threads and calls (fetch_anime_and_parse_html) with given [folderNumber]
'''
def fetch_animes_and_save_file(urls, folderNumber, cores_amount):
    pool = ThreadPool(cores_amount)
    pool.map(lambda url : fetch_anime_and_parse_html(url, folderNumber, (50*(folderNumber-1)) + urls.index(url) +1), urls)

    
## Defining classes for each argument:
def extract_element_from_html(html, html_tag, class_name="", attrs= {}) :
  # title class_name
  soup = BeautifulSoup(html, "html.parser")
  # Find given content
  content = soup.find(html_tag, class_=class_name, attrs= attrs)
  # print(f"Found {html_tag}: {content}")
  return content

def extract_element_from_information_content_by_span_text(html, span_text) :
  # title class_name
  soup = BeautifulSoup(html, "html.parser")
  # Find given gontent
  pads = soup.find_all("div", class_="spaceit_pad", )
  for el in pads :
    span = el.find('span')
    if(span != None and span.text == span_text):
      a = el.find('a')
      if(a != None): 
        return a.text
      contents = el.contents
      if(len(contents) >= 2): 
        return contents[2].strip("\n ")
  return ""

def extract_related_animes(html):
  soup = BeautifulSoup(html, "html.parser")
  subtag = soup.find("table", "anime_detail_related_anime")
  #print(f"Found subtag {subtag}")
  related_animes = []
  if(subtag != None): 
    for el in subtag.find_all("a", href=True):
      #print(el)
      text = el.text
      if(text not in related_animes):
        related_animes.append(text)
  return related_animes


def extract_text_list_from_soup_and_class_names(soup, html_tag, class_name):
  tag_list = soup.find_all(html_tag, class_name)
  output = []
  for el in tag_list:
    text = el.text
    if(text not in output):
      output.append(text)
  return output 

def extract_soups_tag_list(html, html_tag, class_name):
  soup = BeautifulSoup(html, "html.parser")
  output = soup.find_all(html_tag, class_name)
  #print(len(output))
  return output

def parseDate(date, formats, file_path):
  for fmt in formats:
    try:
        return datetime.strptime(date, fmt)
    except ValueError:
        pass
  print(f"No valid date format found for : {date} on {file_path}")
  return ""

def extract_url_from_html(html) :
    soup = BeautifulSoup(html, "html.parser")
    url = soup.find("meta", property="og:url")
    if(url):
        return url["content"]
    else : 
        return ''

def extract_informations_from_anime_html(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        file_path = str(file_path)
        try:
            html = f.read()
        except:
            print("Exception reading html")
        animeTitle = extract_element_from_html(html, "h1", "title-name h1_bold_none")
        animeTitle = "" if animeTitle == None else animeTitle.text

        animeType = extract_element_from_information_content_by_span_text(html, "Type:")
        animeNumEpisode = extract_element_from_information_content_by_span_text(html, "Episodes:")
        rel_and_end_dates = extract_element_from_information_content_by_span_text(html, "Aired:")

        dates = rel_and_end_dates.split(" to ")
        date_formats = ["%b %d, %Y", "%Y", "%b %Y"]

        releaseDate = ""
        if (dates[0] != None) :
          releaseDate = parseDate(dates[0], date_formats, file_path)


        endDate = ""
        if (len(dates) >= 2 and dates[1] != None) :
          endDate = parseDate(dates[1], date_formats, file_path)

        animeNumMembers = ""
        try : 
          animeNumMembers = int(extract_element_from_html(html, "span", "numbers members").text.split()[1].replace(',', ''))
        except Exception as e :
          pass
          #print(f"animeNumMembers - {e} on {file_path}");

        animeScore = ""
        try:
          animeScore = float(extract_element_from_html(html, "div", "score-label").text)
        except Exception as e :
          pass
          #print(f"animeScore - {e} on {file_path}");
        animeUsers = ""
        try: 
          animeUsers = int(extract_element_from_html(html, "div", "fl-l score").get('data-user').split()[0].replace(',',''))
        except Exception as e :
          pass
          #print(f"animeUsers - {e} on {file_path}");
        animeRank = ""
        try: 
          animeRank = int(extract_element_from_html(html, "span", "numbers ranked").text.split()[1].replace('#', '').replace(',',''))
        except Exception as e :
          pass
          #print(f"animeRank - {e} on {file_path}");

        animePopularity = ""
        try:
          animePopularity = int(extract_element_from_html(html, "span", "numbers popularity").text.split()[1].replace('#', '').replace(',',''))
        except Exception as e :
          pass
          #print(f"animePopularity - {e} on {file_path}");
        animeDescription = ""
        try:
          animeDescription = extract_element_from_html(html, "p", "", {"itemprop": "description"}).text
        except Exception as e :
          pass
          #print(f"animeDescription - {e} on {file_path}");
        animeRelated = extract_related_animes(html)
        char_voices_staff_table = extract_soups_tag_list(html, "div", "detail-characters-list clearfix")

        animeCharacters = []
        try: 
          animeCharacters = extract_text_list_from_soup_and_class_names(char_voices_staff_table[0], "h3", "h3_characters_voice_actors")
        except Exception as e :
          pass
          #print(f"animeCharacters {e} on {file_path}")

        animeVoices = []
        try: 
          animeVoices = extract_text_list_from_soup_and_class_names(char_voices_staff_table[0], "td", "va-t ar pl4 pr4")
          animeVoices = [voice.strip('\n').split('\n')[0] for voice in animeVoices]
        except Exception as e :
          pass
          #print(f"animeVoices {e} on {file_path}") 

        animeStaff = []
        try: 
          animeStaff = extract_text_list_from_soup_and_class_names(char_voices_staff_table[1], "td", "borderClass")
          animeStaff = [re.split('\n+', staff) for staff in list(filter(None, [staff.strip('\n') for staff in animeStaff]))]
        except Exception as e :
          pass
          #print(f"animeStaff {e} on {file_path}") 
        url = ''
        try:
            url = extract_url_from_html(html)
        except:
            pass

        article_i = re.findall(re.compile('[0-9]+'), file_path.split('/n')[-1])[-1]
        inherited_name = f"anime_{article_i}.tsv"
        #print(inherited_name)
        Path("./tsv_dataset").mkdir(parents=True, exist_ok=True)

        with open('./tsv_dataset/{}'.format(inherited_name), 'wt', encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['animeTitle','animeType','animeNumEpisode','releaseDate','endDate','animeNumMembers','animeScore',
                                 'animeUsers','animeRank','animePopularity','animeDescription','animeRelated','animeCharacters','animeVoices','animeStaff','url'])
            tsv_writer.writerow([animeTitle, animeType, animeNumEpisode, releaseDate, endDate, animeNumMembers, animeScore,
                                 animeUsers, animeRank, animePopularity, animeDescription, animeRelated, animeCharacters, animeVoices, animeStaff, url])


            
            
def preprocess_string(input_string, stemmer = EnglishStemmer(), tokenizer = word_tokenize ) :
    if not input_string :
        return ''
    # define stopwords
    stop_words = set(stopwords.words('english'))
    # define punctuation
    punctuation = string.punctuation
    translation_table = str.maketrans('', '', punctuation)
    output = []
    for token in [t.lower() for t in tokenizer(input_string)]:
        #print(f"Processing token : {token}")
        #print("removing punctuation")
        token = token.translate(translation_table)
        try :
            if token == '' or token in stop_words:
              #print("token was a stopword, continuing.")
              continue
        except Exception as e:
            print(f"{e} thrown while using stop_words")
        #print("token was NOT a stopword")
        #print(f"token after punctuation removal: {token}")
        if stemmer:
            #print("Stemming token")
            token = stemmer.stem(token)
            #print(f"token after stemming was {token}")
        output.append(token)
        #print(output)
    return output
      
def preprocess_tsv(file_path):
    try:
        file_name = str(file_path).split('\\')[-1]
        with open(file_path, 'r', newline='', encoding="utf-8") as f:
            Path("./preprocessed_dataset").mkdir(parents=True, exist_ok=True)
            output = {}
            tsv = csv.reader(f, delimiter='\t')

            columns = next(tsv)
            next(tsv)
            data = next(tsv)
            for i in range(len(columns)) :
                if(columns[i] not in ['releaseDate', 'endDate', 'url']) :
                    output[columns[i]] = ' '.join(preprocess_string(data[i]))
                else :
                    output[columns[i]] = data[i]
            with open('./preprocessed_dataset/{}.json'.format(file_name.split('\\')[-1].split('.')[0]), 'w', encoding="utf-8") as out_file:
                json.dump(output, out_file)
    except Exception as e:
        print("Error on file {} : {}".format(file_path, e))
            
def map_input_file_into_dictionary(dictionary, file_path):
    f = open(file_path, 'r', encoding='utf-8')
    data = json.load(f)
    values = ' '.join(data.values()).split(' ')
    values.sort()
    for word in values :
        _id = str(hash(word))
        if(word not in dictionary) :
            dictionary[word] = _id
                
def hydrate_vocabulary_from_files(file_paths, cores_amount):
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
    # sorting content to speed up the process
    input_words.sort()
    '''
    for each processed word, find its id via vocabulary and
    update its reference into the output_dict
    '''
    #print('3------------processing input words-------------')
    words_counter = Counter(input_words)
    
    for word in words_counter.keys() : 
        _id = vocabulary_dictionary[word]
        tf = words_counter[word] / len(input_words)
        # # of docs that contain this word
        occurencies = len(idf_source_dictionary[_id])
        # log (len(index) / len(# of documents that contain the word)
        idf = math.log( tot_documents / occurencies)
        
        if(_id not in output_dict) :
            output_dict[_id] = [(file_id, tf * idf)]
        else :
            if file_id not in output_dict[_id] : 
                output_dict[_id].append( (file_id, tf * idf) )


def map_input_to_output_dictionary_given_vocabulary(file_id, input_words, output_dict, vocabulary_dictionary) : 
    # sorting content to speed up the process
    input_words.sort()
    '''
    for each processed word, find its id via vocabulary and
    update its reference into the output_dict
    '''
    #print('3------------processing input words-------------')
    for word in input_words : 
        _id = vocabulary_dictionary[word]
        if(_id not in output_dict) :
            output_dict[_id] = [file_id]
        else :
            if file_id not in output_dict[_id] : 
                output_dict[_id].append(file_id)
    #print('3------------finished processing input-------------')
        
def process_file_synopsis_to_output_given_vocabulary(file_path, output_dictionary, vocabulary_dictionary, want_idf, tot_documents, idf_source_dictionary):
    
    #print('2------------started processing synopsis-------------')
    input_file = open(file_path, 'r', encoding='utf-8')
    input_file_id = str(file_path).split('\\')[-1].split('.')[0]
    #print(f'2------------defined file_id {input_file_id}-------------')
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
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

def associate_words_to_doc_ids(words, vocabulary):
    output = []
    for word in words :
        try:
            output.append(vocabulary[word])
        except :
            raise Exception(f"Couldn't find the word {word} in our dictionary! Are you sure it's not a typo?")
    return output

def extract_documents_from_ids(ids, index) :
    results = {}
    for _id in ids :
        docs = index[_id]
        results[_id] = docs
    return results


def print_query_results(results_ids, with_similarity = False, similarity_dict = {}) :
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
  ## generate empty vector
  query_v = np.zeros(len(vocabulary_dict))
  for word in query_array : 
    query_v[list(vocabulary_dict.keys()).index(word)] = 1

  return query_v

def find_element_in_list(l, condition):
  for el in l :
    if(condition(el)):
      return el

def generate_vector_for_given_document(document_id, vocabulary, index):
    doc_v = np.zeros(len(vocabulary))

    with open('./preprocessed_dataset/{}.json'.format(document_id)) as document_file:
        document = json.load(document_file)
    synopsis = document['animeDescription']
    
    for word in synopsis.split(' ') :
        try: 
            _id = vocabulary[word]
            tuples = index[_id]
            idf = 0
            for tup in tuples :
                if(tup[0] == document_id):
                    idf = tup[1]
                    break;

            doc_v[list(vocabulary.keys()).index(word)] = idf
        except :
            raise Exception(f"Couldn't find the word {word} in our dictionary! Are you sure it's not a typo?")
    return doc_v



def query_K_top_documents(query_string, k):
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
    
    top_k = heapq.nlargest(k, heap)
    for score in top_k :
        output[scores_dictionary[score]] = score
    
    return output


def print_resulting_dataframe(dataframe) :
    dataframe['animeDescription'] = dataframe['animeDescription'].apply(lambda string : string[:8] + '...')
    print(tabulate(dataframe, headers='keys', tablefmt='psql'))
        

def compute_query_on_jaccard_similarity(query_string, k, num_epidode_weight, date_weight, members_weight, score_weight, popularity_weight) :
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
    for value in hits :
        for tupl in hits[value]: 
            if list_of_hits[hits_keys.index(value)] == None :
                list_of_hits[hits_keys.index(value)] = [tupl[0]]
            else :
                list_of_hits[hits_keys.index(value)].append(tupl[0])

    results = set.intersection(*map(set,list_of_hits))
    
    
    # compute dataframe 
    dataframe = compute_pandas_dataframe_for_results(results)
    
    # calculate similarity on Title, Description, Charachters
    jaccard_scores = compute_jaccard_scores(dataframe, query_words, k)
    
    # num_epidode_weight, date_weight, members_weight, score_weight, popularity_weight 
    
    dataframe['releaseDate'] =  dataframe['releaseDate'].apply(lambda d : datetime.strptime(d, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    #dataframe['releaseDate'] = pd.to_datetime(dataframe['releaseDate'], unit='ms')
    
    date_scores = compute_column_scores(dataframe, 'releaseDate', 'dateScores', date_weight)
    num_ep_scores = compute_column_scores(dataframe, 'animeNumEpisode', 'numEpScores', num_epidode_weight)
    anime_members_scores = compute_column_scores(dataframe, 'animeNumMembers', 'memberScores', members_weight)
    anime_rate_scores = compute_column_scores(dataframe, 'animeScore', 'ratingScore', score_weight)
    anime_popularity_score = compute_column_scores(dataframe, 'animePopularity', 'popularityScore', popularity_weight)
    

    
    for index, row in jaccard_scores.iterrows() :
        overall_similarity = row['similarity'] + date_scores['dateScores'][index] + num_ep_scores['numEpScores'][index] + anime_members_scores['memberScores'][index] + anime_rate_scores['ratingScore'][index]+ anime_popularity_score['popularityScore'][index]
        temp_row = pd.DataFrame(data = {'animeTitle' : [dataframe['animeTitle'][index]],'animeDescription' : [dataframe['animeDescription'][index]], 'url' : [dataframe['url'][index]], 'similarity':[overall_similarity]})   
        output_dataframe = output_dataframe.append(temp_row)
    norm_factor = output_dataframe['similarity'].max()
    output_dataframe['similarity'] = output_dataframe['similarity'].apply(lambda x : x / norm_factor)
    top_k_results = compute_top_k_dataset(output_dataframe, k)
    
    print_resulting_dataframe(top_k_results)


def compute_pandas_dataframe_for_results(results_ids):
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
            # fix this
            documents_dataframe = pd.concat([documents_dataframe, temp], sort = True)
            
    # print(f"documents dataframe is {documents_dataframe}")
    return documents_dataframe.reset_index(drop=True)

def jaccard_similarity(_query_array, _text_array):
    _query = set(_query_array)
    _text = set(_text_array)
    return len(_query.intersection(_text)) / len(_query.union(_text))


def normalize_jaccard(values, min_jac, max_jac):
    return [(x - min_jac)  / (max_jac - min_jac) for x in values]

def compute_jaccard_scores(documents_dataframe, query_text, k):
    jac_dataframe = documents_dataframe[["animeTitle","animeDescription","animeRelated","animeCharacters","similarity"]]
    
    jac_scores = []
    jac_score_final = []
    
    cols = jac_dataframe.columns
    
    # calculate jaccart scores
    for index, row in jac_dataframe.iterrows():
        score = []
        for col in cols:
            text = str(row[col])
            # append the score of every variable of a given document
            # animeTitle : score, animeDesc : score... exc...
            jac_scores.append(jaccard_similarity(query_text, preprocess_string(text)))
        
        # sum up all the scores of a given document into a single score
        jac_score_final.append(sum(jac_scores))
    
    if(len(jac_score_final) > 1) :
        jac_score_final = normalize_jaccard(jac_score_final, min(jac_score_final), max(jac_score_final))
    
    # normalize jaccard 
    jac_dataframe['similarity'] = list(map(lambda x: round(x, 2), jac_score_final))
    return jac_dataframe[["animeTitle", "animeDescription", "similarity"]]
    
    

    
'''
This function takes a pandas dataframe having:
animeTitle | animeDescription | url | similarity
as fields.
It heap sorts the dataframe's rows based on similarity and returns
k best elements
'''
def compute_top_k_dataset(input_dataset, k) :
    
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
    try:
        return float(string)
    except:
        return 0

'''
This takes a result dataframe as an input and returns a series of scores based on weight
'''
def compute_column_scores(dataframe, source_column, output_column, output_weight) :
    result = pd.DataFrame(columns=[output_column])
    source = dataframe[source_column].apply(lambda x : convert_str_to_float(x))
   # source = pd.to_numeric(dataframe[source_column])
    norm_factor = source.max()
    result[output_column] = source.apply(lambda x: (x / norm_factor) * output_weight)
    return result


def take_simple_query_from_user():
    try:
        print("Hi! Type in your query : ")
        query = str(input())
        perform_query(query)
    except Exception as e :
        print(e)

def take_top_k_of_query_from_user() :
    try:
        print("Hi! Type in your query : ")
        query = str(input())
        print("How many results would you like to retrieve?")
        k = int(input())
        query_K_top_documents(query, k)
    except Exception as e :
        print(e)
        
def take_biased_query_from_user():
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
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

def save_urls_to_txt(pages_of_urls):
    """
    Takes an array of urls and sequentially parses them inside a txt.
    """
    count = 0
    with open('./dataset/anime_urls.txt', 'w', encoding='utf-8') as txt_file :
        for page in pages_of_urls :
            for url in page :
                txt_file.write(url + os.linesep)
                count+=1
    print(f'written {count} lines')
    return True

def fetch_page(index, destination_array):
    """
    This function performs an HTTP Get Request to MyAnimeList and places its results in a given array.\n
    Params: 
        [index] : Simply the page index. Sets up the url for pagination and defines where the page will be placed inside [destination_array]
        [destination_array] : where the retrieved page will be stored. The result will be placed in index [index]
    """
    destination_array[index] = requests.get(f"https://myanimelist.net/topanime.php{'?limit={}'.format(50*index) if(index > 0) else ''}")
    

def fetch_urls_in_page(page, pages):
    """
    Scrape all anime URLs contained in a MyAnimeList top animes page HTML page.
    Substitutes every url to its corresponding page inside the [pages] array.\n
    Params: 
        [page]  : MyAnimeList's Top Animes HTML Page
        [pages] : Array containing all the pages. 
    """
    # Defining an html parser
    soup = BeautifulSoup(page.content, "html.parser")
    # Find all URLs
    animeUrls = soup.find_all("a", class_="hoverinfo_trigger fl-l ml12 mr8", id=lambda x: x and x.startswith('#area'), href=True)
    animeUrls = [a['href'] for a in animeUrls]
    # Substitues starting page with its URLs
    pages[pages.index(page)] = animeUrls

    
def fetch_anime_and_parse_html(url, folder_id, index):
    '''
    Performs a GET Request on a given [url] to retrieve a particular anime page from MyAnimeList.
    The Result is saved as an HTML inside a folder called "page_[folder]".
    The HTML file will be named "article_[index].html"\n
    Params:
        [url] : Request url
        [folder_id] : id that completes the destination path:
            [directory_path] : "./dataset/page_{folder_id}"
        [index] : id that completes the file name :
            "./{directory_path}/article_{index}.html"
    '''
    # Get current page
    req = requests.get(url)
    # MyAnimeList might refuse to respond to large amount of requests, if this happens, we need to stop the process
    if(req.status_code != 200) : 
        raise Exception(f"My anime list has closed the connection.\nComplete the captcha and restart the process.\nCurrent Page was : {index}")
    # Define page's absolute destination path
    _directory_path = f"{pathlib.Path().resolve()}/dataset/page_{folder_id}"
    # Check if current page's destination folder exists... if not, create it!
    Path(_directory_path).mkdir(parents=True, exist_ok=True)
    # Write the html file in the destination directory.
    with open(f"{_directory_path}/article_{index}.html", 'w') as file:
        file.write(req.text)
    

def fetch_animes_and_save_file(urls, folderNumber, cores_amount):
    '''
    Use Multiprocessing to assign to given amount of threads
    the fetching and saving progress of all given urls into HTMLs.
    
    This calls [fetch_anime_and_parse_html] to save MyAnimeList pages into files.
    '''
    pool = ThreadPool(cores_amount)
    pool.map(lambda url : fetch_anime_and_parse_html(url, folderNumber, (50*(folderNumber-1)) + urls.index(url) +1), urls)

    
def extract_element_from_html(html, html_tag, class_name="", attrs= {}) :
    '''
    Scrapes given content from a given HTML.\n
    Params:
        [html] : html content to scrape
        [html_tag] : the tag to find, ex: div
        [class_name] : the class_name to find, ex: "h1 -b title"
        [attrs] : attributes dictionary, can be left blank if you are not looking for attributes
    
    Returns:
        [content] : Relative element of the given html that satisfies all given constraints.
    '''
    soup = BeautifulSoup(html, "html.parser")
    # Find given content
    content = soup.find(html_tag, class_=class_name, attrs= attrs)
    # print(f"Found {html_tag}: {content}")
    return content

def extract_element_from_information_content_by_span_text(html, span_text) :
    """
    Extracts from given html the text wrapped inside a span containing given span_text\n
    Params:
        [html] : source content
        [span_text] : text that characterizes the span containing what you're looking for
    """
    
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
    """
    Extracts all "Related Animes" content from a given MyAnimeList anime page.\n
    Params:
        [html] : Anime Html Page
    """
    soup = BeautifulSoup(html, "html.parser")
    subtag = soup.find("table", "anime_detail_related_anime")
    #print(f"Found subtag {subtag}")
    related_animes = []
    if(subtag != None): 
        for el in subtag.find_all("a", href=True):
            text = el.text
            if(text not in related_animes):
                related_animes.append(text)
    return related_animes


def extract_text_list_from_soup_and_class_names(soup, html_tag, class_name):
    """
    Extracts a list of unique texts contained in given [soup] with given [html_tag] and [class_name]\n
    Params:
        [soup] : BeautifulSoup to use
        [html_tag] : Tag to seek, ex: div
        [class_name] : class_name to seek, ex: "h1 b title"
    """
    tag_list = soup.find_all(html_tag, class_name)
    output = []
    for el in tag_list:
        text = el.text
        if(text not in output):
            output.append(text)
    return output 

def extract_soups_tag_list(html, html_tag, class_name):
    """
    Extract soup list of all html_tags with class_name in given html\n
    Params: 
        [html] : source html content
        [html_tag] : Tag to seek, ex: div
        [class_name] : class_name to seek, ex: "h1 b title"
    """
    soup = BeautifulSoup(html, "html.parser")
    output = soup.find_all(html_tag, class_name)
    return output

def parseDate(date, formats, file_path):
    """
    Given a list of possible date formats, this tries to parse
    a given date in one of those formats. If no format was possible, 
    logs the impossibility and returns "".  

    This is particularly useful to scrape dates out of htmls,
    since they appear in many different formats, including "?"
    """
    for fmt in formats:
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            pass
    print(f"No valid date format found for : {date} on {file_path}")
    return ""

def extract_url_from_html(html) :
    """
    Extracts the html source URL from html's Metadata.
    """
    soup = BeautifulSoup(html, "html.parser")
    url = soup.find("meta", property="og:url")
    if(url):
        return url["content"]
    else : 
        return ''

def extract_informations_from_anime_html(file_path):
    """
    This function takes as input an html and extracts all useful informations into an output tsv file.
    This makes use of all the functions contained in this module to scrape informations out of the html.

    Given the input file, the informations that will be extracted are:

    [ animeTitle, animeType, animeNumEpisode, releaseDate, endDate, animeNumMembers, animeScore, 
    animeUsers, animeRank, animePopularity, animeDescription, animeRelated, animeCharacters,
    animeVoices, animeStaff, url ]

    Params:
        [file_path]: Path of the html file to read
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        file_path = str(file_path)
        try:
            html = f.read()
        except:
            print("Exception reading html")

        # Extract title given its relative tag and class_name from html
        animeTitle = extract_element_from_html(html, "h1", "title-name h1_bold_none")
        # If there is an error we leave an empty string
        animeTitle = "" if animeTitle == None else animeTitle.text

        # Extract type value by seeking it inside the "Type: " span
        animeType = extract_element_from_information_content_by_span_text(html, "Type:")
        # Extract type value by seeking it inside the "Episodes: " span
        animeNumEpisode = extract_element_from_information_content_by_span_text(html, "Episodes:")
        # Extract date value by seeking them inside the "Aired: " span
        rel_and_end_dates = extract_element_from_information_content_by_span_text(html, "Aired:")
        # split dates into two values
        dates = rel_and_end_dates.split(" to ")
        # define possible date formats
        date_formats = ["%b %d, %Y", "%Y", "%b %Y"]

        # Parse release and end dates in any of the possible formats, leaving an empty string if it's impossible
        releaseDate = ""
        if (dates[0] != None) :
            releaseDate = parseDate(dates[0], date_formats, file_path)

        endDate = ""
        if (len(dates) >= 2 and dates[1] != None) :
            endDate = parseDate(dates[1], date_formats, file_path)

        animeNumMembers = ""
        # Extract members given their relative tag and class_name from html
        try : 
            animeNumMembers = int(extract_element_from_html(html, "span", "numbers members").text.split()[1].replace(',', ''))
        except Exception as e :
            pass

        animeScore = ""
        try:
            # Extract score given its relative tag and class_name from html
            animeScore = float(extract_element_from_html(html, "div", "score-label").text)
        except Exception as e :
            pass
            #print(f"animeScore - {e} on {file_path}");
        animeUsers = ""
        try: 
            # Extract users div given its relative tag and class_name from html,
            # then scrapes data user attribute and parses it as an int.
            animeUsers = int(extract_element_from_html(html, "div", "fl-l score").get('data-user').split()[0].replace(',',''))
        except Exception as e :
            pass
        animeRank = ""
        try: 
            # Extract rank span given its relative tag and class_name from html,
            # then parses its text as an int.
            animeRank = int(extract_element_from_html(html, "span", "numbers ranked").text.split()[1].replace('#', '').replace(',',''))
        except Exception as e :
            pass
            #print(f"animeRank - {e} on {file_path}");

        animePopularity = ""
        try:
            # Extract popularity span given its relative tag and class_name from html,
            # then parses its text as an int.
            animePopularity = int(extract_element_from_html(html, "span", "numbers popularity").text.split()[1].replace('#', '').replace(',',''))
        except Exception as e :
            pass
        
        animeDescription = ""
        try:
            # Extract description text given its relative tag and class_name from html,
            animeDescription = extract_element_from_html(html, "p", "", {"itemprop": "description"}).text
        except Exception as e :
            pass
        
        # Extract related animes
        animeRelated = extract_related_animes(html)
        # Extract characters voices and staff tags subtree from source html
        char_voices_staff_table = extract_soups_tag_list(html, "div", "detail-characters-list clearfix")

        animeCharacters = []
        try: 
            # Extracts characters from subtree
            animeCharacters = extract_text_list_from_soup_and_class_names(char_voices_staff_table[0], "h3", "h3_characters_voice_actors")
        except Exception as e :
            pass

        animeVoices = []
        try: 
            # Extracts voices from subtree
            animeVoices = extract_text_list_from_soup_and_class_names(char_voices_staff_table[0], "td", "va-t ar pl4 pr4")
            animeVoices = [voice.strip('\n').split('\n')[0] for voice in animeVoices]
        except Exception as e :
            pass

        animeStaff = []
        try: 
            # Extracts staff from subtree
            animeStaff = extract_text_list_from_soup_and_class_names(char_voices_staff_table[1], "td", "borderClass")
            animeStaff = [re.split('\n+', staff) for staff in list(filter(None, [staff.strip('\n') for staff in animeStaff]))]
        except Exception as e :
            pass
        
        url = ''
        try:
            # Extracts url from html metadata
            url = extract_url_from_html(html)
        except:
            pass
        
        # Compiles article _id from source path
        article_i = re.findall(re.compile('[0-9]+'), file_path.split('/n')[-1])[-1]
        # Build up output file's name
        inherited_name = f"anime_{article_i}.tsv"

        # Check if output directory exists, if not, make it.
        Path("./tsv_dataset").mkdir(parents=True, exist_ok=True)

        # Write output's tsv
        with open('./tsv_dataset/{}'.format(inherited_name), 'wt', encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['animeTitle','animeType','animeNumEpisode','releaseDate','endDate','animeNumMembers','animeScore',
                                 'animeUsers','animeRank','animePopularity','animeDescription','animeRelated','animeCharacters','animeVoices','animeStaff','url'])
            tsv_writer.writerow([animeTitle, animeType, animeNumEpisode, releaseDate, endDate, animeNumMembers, animeScore,
                                 animeUsers, animeRank, animePopularity, animeDescription, animeRelated, animeCharacters, animeVoices, animeStaff, url])


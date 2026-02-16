#coding: utf-8
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
from urllib.parse import urlparse
import datetime
import random
import re
from gtts import gTTS
import time
import sys

keyword=sys.argv[1]

class Content:
    """
    Common base class for all articles/pages
    """
    def __init__(self, url, title, body):
        self.url = url
        self.title = title
        self.body = body

    def print(self):
        """
        Flexible printing function controls output
        """
        print('URL: {}'.format(self.url))
        print('TITLE: {}'.format(self.title))
        print('BODY:\n{}'.format(self.body))


def getPage(url):
    """
    Utilty function used to get a Beautiful Soup object from a given URL
    """

    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    try:
        req = session.get(url, headers=headers)
    except requests.exceptions.RequestException:
        return None
    bs = BeautifulSoup(req.text, 'html.parser')
    return bs

def scrapeNYTimes(url):
    bs = getPage(url)
    title = bs.find('h1').text
    lines = bs.select('div.StoryBodyCompanionColumn div p')
    body = '\n'.join([line.text for line in lines])
    return Content(url, title, body)

def scrapeBrookings(url):
    bs = getPage(url)
    title = bs.find('h1').text
    body = bs.find('div', {'class', 'post-body'}).text
    return Content(url, title, body)

def scrapeNews(url):
    bs = getPage(url)
    title = bs.find('div').title
    body = bs.find('div', {'class', 'title'})
    return Content(url, title, body)

def scrapBingNews(url):
    base = url
    response=requests.get(base)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body

class Website:
    """ 
    Contains information about website structure
    """

    def __init__(self, name, url, titleTag, bodyTag):
        self.name = name
        self.url = url
        self.titleTag = titleTag
        self.bodyTag = bodyTag

#data clean
def getNewsDataClean(full_text, content_dict, str_datetime):
    contents = re.sub(r'([a-zA-Z0-9]*新聞[a-zA-Z0-9]*)','AAAAA',full_text)
    #data split
    delimiters = "a", "...", "(C)", "(R)"
    regexPattern = '|'.join(map(re.escape, delimiters)) # 'a|\\.\\.\\.|\\(C\\)'
    content = re.split(regexPattern, contents) # 
    #data record into dictionary
    regexp = re.compile(r'(.+AAAAA.+)')
    regexc = re.compile(r'(.+AAAAA.+)')
    for line in content: 
        fields0 = re.sub(r'(([a-zA-Z]+\.)+|\.){2,4}','', line)
        fields1 = re.sub(r'.+(新闻|新聞|News|\n+).+',' ',fields0);
        if(len(fields1)>=20 and not regexp.search(fields1)):
            if fields1 in content_dict.keys():
              print("skip")
            else:
                content_dict[fields1]=str(datetime.datetime.now());
    str_full_text = str(content_dict)
    return str_full_text, content_dict

def get_datetime():
    str_now = str(datetime.datetime.now())
    return str_now

def save_files(content_dict, basepath, str_now, keyword):
    str_now2 = re.sub(r'[-|: ]','_',str_now)
    str_now3 = re.sub(r'_[0-9]+_[0-9]+.[0-9]+$','00',str_now2)
    str_now = str_now3
    #save csv file as $date.csv
    csv_file_name = basepath + str_now + '_' + keyword + '.csv'
    
    print(csv_file_name)
    print(content_dict)
    with open(csv_file_name, 'w') as f:
        for key in content_dict.keys():
            f.write("%s,%s\n"%(key,content_dict[key]))
    tts=gTTS(text=str_full_text, lang='zh', slow=False)
    radio_filename = basepath + str_now + '_' + keyword + '.mp3'
    print(radio_filename)
    tts.save(radio_filename)
    return 1

# parameter setting
pages=set()
random.seed(datetime.datetime.now())
basepath='/Users/rubylintu/Desktop/News_DB/Data/Data/'
content_dict={}
keyword = sys.argv[1]
print(keyword)
#initialize
url = r"https://www.bing.com/news/search?q=%22" + keyword + "%22&go=搜尋&qs=n&form=QBNT&sp=-1&pq=%22" + keyword


# Main

str_now= get_datetime()
url, title, fulltext = scrapBingNews(url)

str_full_text, content_dict = getNewsDataClean(fulltext, content_dict, str_now)
return_val = save_files(content_dict, basepath, str_now, keyword)
print(return_val)


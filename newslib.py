#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:32:54 2021

@author: rubylintu
"""
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
import json
import os
from openai import OpenAI
import docx
import pdfplumber

def getPage(url):
    """
    Utilty function used to get a Beautiful Soup object from a given URL
    """

    session = requests.Session()
    headers = {
    'host': 'www.google.co.kr',
    'method': 'GET',
    'referer': 'https://www.google.com/',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    proxies = {'http':'http://10.10.10.10:8765','https':'https://10.10.10.10:8765'}
    cookies = dict(uuid='b18f0e70-8705-470d-bc4b-09a8da617e15',UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')
    #res = requests.get(url, headers=headers, cookies=cookies)
    
    try:
        req = session.get(url, headers=headers, cookies=cookies)
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

def scrapBingNews(keyword):
    url = r"https://www.bing.com/news/search?q=%22" + keyword + "%22&go=搜尋&qs=n&form=QBNT&sp=-1&pq=%22" + keyword
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body

def scrapGoogleNews(keyword):
    url="https://www.google.com/search?q="+keyword+"&tbm=nws&start=%d"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body

def getGoogleBaInfo(num):
    url="https://www.google.com/search?q=可轉債 上櫃&start=%d"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs
def gettpexCBInfo(num):
    url="https://www.tpex.org.tw/web/bond/publish/convertible_bond_search/memo.php?l=zh-tw"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs



def getGoodStockInfo(num):
    url="https://goodinfo.tw/StockInfo/StockDetail.asp?STOCK_ID="+str(num)
    response=requests.get(url)
    response.encoding = "utf-8"
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs


def scrapeGoogleStockInfo(num):
    url="https://www.google.com/search?q="+str(num)+"股價&start=%d"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    response=requests.get(url, headers=headers)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    str_stockprice=re.search("股價[0-9]+",body)
    mat = re.split("股價",str_stockprice.group(0))
    stockprice = str(mat[1])
    url="https://www.google.com/search?q="+str(num)+"成交量&start=%d"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    str_deal=re.search("[0-9]成交量",body)
    mat = re.split("成交量",str_deal.group(0))
    deal = str(mat[0])
    return deal, stockprice




    
def scrape_stockclub(num):
    if(num == -999):
        num = 3896602#5769561 #5794329
    url="https://www.cmoney.tw/follow/channel/articles-" + str(num) + "#/"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = re.sub("\n","",bs.text)
    #data = json.loads(body)
    return url, title, bs


def get_stock_info(num):
    url="https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_"+str(num)+".tw"
    headers = {
    'host': 'www.google.co.kr',
    'method': 'GET',
    'referer': 'https://www.google.com/',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36'}
    proxies = {'http':'http://10.10.10.10:8765','https':'https://10.10.10.10:8765'}
    cookies = dict(uuid='b18f0e70-8705-470d-bc4b-09a8da617e15',UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')
    res = requests.get(url, headers=headers, cookies=cookies)
    data = res.json()
    data2 = data['msgArray']
    dict2=data2[0]
    num=dict2['c']
    name = dict2['n']
    deal = dict2['z']
    dealamount = dict2['tv']
    wholedealamount = dict2['v']
    startprice = dict2['o']
    highestprice = dict2['h']
    lowestprice = dict2['l']
    yestprice = dict2['y']
    data3 = [num, name, deal, dealamount, wholedealamount, startprice, highestprice, lowestprice, yestprice]
    #columns = ['c','n','z','tv','v','o','h','l','y']
    #df = pd.DataFrame(data['msgArray'], columns=columns)
    #df.columns = ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']
    #df.insert(9, "漲跌百分比", 0.0) 
    return url, data3


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

def save_files(str_full_text, content_dict, basepath, str_now, keyword):
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
def save_mp3(str_full_text, content_dict, keyword):
    tts=gTTS(text=str_full_text, lang='zh', slow=False)
    radio_filename = 'html/' + keyword + '.mp3'
    print(radio_filename)
    tts.save(radio_filename)
    return 1


def save_csv_file(str_full_text, content_dict, basepath, str_now, keyword):
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
    f.close()
    return 1
def create_filename(keyword):
    str_now = get_datetime()
    str_now2 = re.sub(r'[-|: ]','_',str_now)
    str_now3 = re.sub(r'_[0-9]+_[0-9]+.[0-9]+$','00',str_now2)
    str_now = str_now3
    filename =str_now + '_' + keyword
    return filename

def highlight_word(fulltext, word):
    fulltext2 = re.sub(word, "<mark>"+word+"</mark>", fulltext)
    return fulltext2

def lowlight_word(fulltext, word):
    fulltext2 = re.sub(word, "<mark style=\"color: white; background-color:green\">"+word+"</mark>", fulltext)
    return fulltext2

def add_keyword(keyword):
    dict1 = read_keyword('bull')
    dict2 = read_keyword(keyword)
    dict3 = read_keyword('bear')
    for word in dict2.keys():
        if(type(dict2[word])==str):
            continue
        if (float(dict2[word]) > 1) and (word not in dict1):
            dict1[word] = dict2[word]
            print(dict2[word], word)
        elif (float(dict2[word]) < -1) and (word not in dict3):
            dict3[word] = dict2[word]
            print(dict2[word], word)
    #save file
    filename = 'Data/dict_bull.csv'
    f1 = open(filename,'w')
    for word in dict1.keys():
        print(word)
        f1.write(word+','+str(dict1[word])+',')
    f1.close()
    filename = 'Data/dict_bear.csv'
    f2 = open(filename,'w')
    for word in dict3.keys():
        f2.write(word+','+str(dict3[word])+',')
    f2.close()
    return 1
    
def read_keyword(keyword):
    dict_keyword = {}
    f=open('Data/dict_'+keyword+'.csv','r')
    line = f.readline()
    lines = f.readlines()
    for line in lines:
        mat=re.split(',|\n',line)
        if len(mat) >= 1:
            i=0
            while i <len(mat)-2:
                try: 
                    if int(mat[i+1]) and (len(mat[i+1]) > 0):
                        dict_keyword[mat[i]] = int(mat[i+1])
                        i=i+2
                        continue
                except:
                    i = i + 1
    f.close()
    return dict_keyword


# ── DOCX / PDF → MP3 (OpenAI TTS) ──────────────────────────────

def read_docx(filepath):
    """讀取 .docx 檔案，回傳全文字串"""
    doc = docx.Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return '\n'.join(paragraphs)


def read_pdf(filepath):
    """用 pdfplumber 逐頁提取文字，回傳全文字串"""
    texts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
    return '\n'.join(texts)


def chunk_text(text, max_chars=4000):
    """將文字切成 <= max_chars 的區塊，優先在句號/問號/驚嘆號處切割"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while len(text) > max_chars:
        slice_ = text[:max_chars]
        # 優先：句號、問號、驚嘆號
        cut = -1
        for sep in ['。', '.', '！', '!', '？', '?']:
            idx = slice_.rfind(sep)
            if idx > cut:
                cut = idx
        # 備用：逗號、換行
        if cut < max_chars // 2:
            for sep in ['，', ',', '\n']:
                idx = slice_.rfind(sep)
                if idx > cut:
                    cut = idx
        # 最後手段：硬切
        if cut < max_chars // 2:
            cut = max_chars - 1
        cut += 1  # 包含分隔符
        chunks.append(text[:cut])
        text = text[cut:]
    if text.strip():
        chunks.append(text)
    return chunks


def text_to_mp3_openai(text, output_path, voice='alloy', model='tts-1'):
    """呼叫 OpenAI TTS API 將文字轉為 MP3"""
    client = OpenAI()
    chunks = chunk_text(text)

    if len(chunks) == 1:
        response = client.audio.speech.create(
            model=model, voice=voice, input=chunks[0]
        )
        response.stream_to_file(output_path)
    else:
        tmp_files = []
        try:
            for i, chunk in enumerate(chunks):
                tmp_path = f"{output_path}.part{i}.mp3"
                tmp_files.append(tmp_path)
                response = client.audio.speech.create(
                    model=model, voice=voice, input=chunk
                )
                response.stream_to_file(tmp_path)
            # MP3 frames 可直接二進位串接
            with open(output_path, 'wb') as out:
                for tmp in tmp_files:
                    with open(tmp, 'rb') as f:
                        out.write(f.read())
        finally:
            for tmp in tmp_files:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print(f"已儲存: {output_path}")
    return 1


def doc_to_mp3(filepath, output_path=None, voice='alloy', model='tts-1'):
    """主入口：自動偵測 .docx/.pdf，提取文字後轉 MP3"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.docx':
        text = read_docx(filepath)
    elif ext == '.pdf':
        text = read_pdf(filepath)
    else:
        raise ValueError(f"不支援的檔案格式: {ext}（僅支援 .docx / .pdf）")

    if not text.strip():
        raise ValueError("檔案中未提取到任何文字")

    if output_path is None:
        output_path = os.path.splitext(filepath)[0] + '.mp3'

    return text_to_mp3_openai(text, output_path, voice=voice, model=model)

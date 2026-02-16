# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:21:58 2021

@author: rubby
"""
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
from urllib.parse import urlparse
import datetime
import random
import re
import time
import sys
import json

from newslib import *


url = 'https://www.tpex.org.tw/web/bond/publish/convertible_bond_search/memo.php?l=zh-tw'
bs = getPage(url)

mat = bs.table;
nodes = mat.find_all('td')
tdi = 0
for line in nodes:
    if tdi%8 == 0:
        num=int(line.string)
        name = nodes[tdi+1].string
        fullname = nodes[tdi+2].string
        bgntime = nodes[tdi+3].string
        endtime = nodes[tdi+4].string
        issueamount = nodes[tdi+6].string
        tmp_dict=nodes[tdi+7].a
        website = tmp_dict['href']
        
        mattime0 = re.split('/',bgntime)
        bgnyy = int(mattime0[0])
        bgnmm = int(mattime0[1])
        bgndd = int(mattime0[2])
        
        mattime1 = re.split('/',endtime)
        endyy = int(mattime1[0])
        endmm = int(mattime1[1])
        enddd = int(mattime1[2])
        
        print(num, name, bgnyy, endyy)
    tdi = tdi + 1

#,{'class':"text-center"}
#for link in bs.find_all('div', {'id':"rpt_result"}).tr.odd:
#    print(link)
    #    if 'href' in link.attrs:
#        print(link.attrs['href'])
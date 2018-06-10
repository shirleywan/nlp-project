#coding=utf-8

import os
import sys
import pickle
import urllib2
from bs4 import BeautifulSoup as bs
from news import News

#爬虫

def getNewsClass(url, typeName):#按照类别返回新闻
    page = urllib2.urlopen(url)#使用 urllib2 这个模块来抓取网页, 模块提供了读取 web 页面数据的接口，我们可以像读取本地文件一样读取 www 和 ftp 上的数据.
    html = page.read()
    #通过调用 urlopen 并传入 Request 对象，响应后将返回 response 文件对象，再调用 read() 函数读取抓取的网页内容。
    soup = bs(html)
    articleBody = soup.find('div', class_ = 'article') #取所有含有article字样的，标题是articleTitle，新闻本体是articletext
    title = articleBody.find('h1').getText()#找当前tag的所有tag子节点，并判断是否符合过滤器条件
    title = title.strip()
    timeStamp = articleBody.find('span', class_ = 'arti_update').getText()#更新时间，取里面的文本
    timeStamp = timeStamp.strip()
    fromWhere = articleBody.find('span', class_ = 'arti_from').getText()
    fromWhere = fromWhere.strip()
    content = ''#内容
    if articleBody.findAll('p', class_ = 'p_text_indent_2'):
        for part in articleBody.findAll('p', class_ = 'p_text_indent_2'):
            content += part.getText().strip()
    if not content and articleBody.find('div', class_ = 'content_old'):
        content = articleBody.find('div', class_ = 'content_old').getText()
    if not content and articleBody.find('div', class_ = 'wp_articlecontent'):
        content = articleBody.find('div', class_ = 'wp_articlecontent').getText()
    content = content.strip()
    if not content:
        print title, url, 'have no content'
    return News(url, typeName, timeStamp, fromWhere, content, title)

def getNewsofType(typeName):#获取该列别下的urls
    urls = None
    with open('Url/%s' % typeName, 'r') as inFile:
        urls = pickle.load(inFile)#使用pickle模块，反序列化对象。将文件中的数据解析为一个 Python 对象。
    newsList = []
    for url in urls:
        if url == '/4d/17/c416a19735/page.htm' or url.startswith('http'):
            continue
        news = getNewsClass('http://news.hit.edu.cn' + url, typeName)
        newsList.append(news)
    os.mkdir('Data')
    with open('Data/%s.obj' % typeName, 'w') as outFile:
        pickle.dump(newsList, outFile) #序列化对象，并将结果数据流写入到文件对象中。

if __name__ == '__main__':
    newsTypes = ['人才培养','学校要闻','校友之苑','理论学习','媒体看工大','他山之石','时势关注','校园文化','科研在线','国际合作','服务管理','深度策划','综合新闻']
    for newsType in newsTypes:
        getNewsofType(newsType)
import random
import re
import traceback

import jieba
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB

jieba.load_userdict("train\\word.txt")
stop = [line.strip() for line in open('ad\\stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
#str.strip():返回移除字符串头尾指定的字符生成的新字符串。
#print (stop)


def build_key_word(path):  # 通过词频产生特征
    d = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            for word in jieba.cut(line.strip()):
                p = re.compile(r'\w', re.L) #re：正则表达式；\w：匹配任意数字和字母:[a-zA-Z0-9]
                    # re.L：表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境；
                result = p.sub("", word) #re.sub 是个正则表达式方面的函数，用来实现通过正则表达式，实现比普通字符串的 replace 更加强大的替换功能；

                if not result or result == ' ':  # 空字符
                    continue
                if len(word) > 1:  # 避免大量无意义的词语进入统计范围
                    d[word] = d.get(word, 0) + 1#字典操作，如字典d中有word，则取出后+1；如果没有，设置值为0，+1放入d中
    kw_list = sorted(d, key=lambda x: d[x], reverse=True)#根据x的值对d进行排序，出现次数；
    size = int(len(kw_list) * 0.3)  # 取最前的30%
    mood = set(kw_list[:size])#取出，构成集合
    return list(mood - set(stop))# 删除停用词后返回list


def loadDataSet(path):  # 返回每条微博的分词与标签
    line_cut = []
    label = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            temp = line.strip()
            try:
                sentence = temp[2:].lstrip()  # 每条微博内容 ，lstrip()：去除字符串从首位开始与之相匹配的字符
                label.append(int(temp[:2]))  # 获取标注
                word_list = []
                sentence = str(sentence).replace('\u200b', '') #\u200b：zero with space
                for word in jieba.cut(sentence.strip()):#分词
                    p = re.compile(r'\w', re.L) #re.L：使预定字符类 \w \W \b \B \s \S 取决于当前区域设定
                    result = p.sub("", word) #re.sub:函数进行以正则表达式为基础的替换工作
                    if not result or result == ' ':  # 空字符
                        continue
                    word_list.append(word) #加到word_list中
                word_list = list(set(word_list) - set(stop) - set('\u200b')
                                 - set(' ') - set('\u3000') - set('️'))
                line_cut.append(word_list) #修建后的句子形成文章的list
            except Exception:
                continue
    return line_cut, label  # 返回每条微博的分词和标注


def setOfWordsToVecTor(vocabularyList, moodWords):  # 每条微博向量化
    vocabMarked = [0] * len(vocabularyList) #每条微博是一个向量
    for smsWord in moodWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return np.array(vocabMarked)


def setOfWordsListToVecTor(vocabularyList, train_mood_array):  # 将所有微博准备向量化,单词表+文章list
    vocabMarkedList = []
    for i in range(len(train_mood_array)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, train_mood_array[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


def trainingNaiveBayes(train_mood_array, label):  # 计算先验概率 -- 根据已有的目标文件统计后的结果
    numTrainDoc = len(train_mood_array)
    numWords = len(train_mood_array[0]) #每条微博向量化的长度，单词表的长度
    prior_Pos, prior_Neg, prior_Neutral = 0.0, 0.0, 0.0
    for i in label:
        if i == 1: #positive的数目
            prior_Pos = prior_Pos + 1
        elif i == 2: #消极的数目
            prior_Neg = prior_Neg + 1
        else: #其他
            prior_Neutral = prior_Neutral + 1
    prior_Pos = prior_Pos / float(numTrainDoc) #positive微博的概率
    prior_Neg = prior_Neg / float(numTrainDoc)
    prior_Neutral = prior_Neutral / float(numTrainDoc)
    wordsInPosNum = np.ones(numWords) #生成全1的向量；
    wordsInNegNum = np.ones(numWords)
    wordsInNeutralNum = np.ones(numWords)
    PosWordsNum = 2.0  # 如果一个概率为0，乘积为0，故初始化1，分母2
    NegWordsNum = 2.0
    NeutralWordsNum = 2.0
    for i in range(0, numTrainDoc):#比对每一个
        try:
            if label[i] == 1: #统计积极的微博类型中，单词的个数
                wordsInPosNum += train_mood_array[i]
                PosWordsNum += sum(train_mood_array[i])  # 统计Positive中语料库中词汇出现的总次数
            elif label[i] == 2:
                wordsInNegNum += train_mood_array[i] #wordsInNegNum是向量，每个单词出现次数；
                NegWordsNum += sum(train_mood_array[i])  # 统计negitive中语料库中词汇出现的总次数
            else:
                wordsInNeutralNum += train_mood_array[i]
                NeutralWordsNum += sum(train_mood_array[i])
        except Exception as e:
            traceback.print_exc(e)
    pWordsPosicity = np.log(wordsInPosNum / PosWordsNum) #positive在单词表向量中个数/总单词个数 -- 每个单词出现频率；
            #np.log：计算各元素的自然对数；np.log10(a) np.log2(a) : 计算个元素对10、2 为底的对数
    pWordsNegy = np.log(wordsInNegNum / NegWordsNum)
    pWordsNeutral = np.log(wordsInNeutralNum / NeutralWordsNum)
    return pWordsPosicity, pWordsNegy, pWordsNeutral, prior_Pos, prior_Neg, prior_Neutral


def classify(pWordsPosicity, pWordsNegy, pWordsNeutral, prior_Pos, prior_Neg, prior_Neutral,
             test_word_arrayMarkedArray):
            #这里是计算每条微博中，根据单词表向量化后，乘以每个单词是positive的概率向量，加上log(P)
    pP = sum(test_word_arrayMarkedArray * pWordsPosicity) + np.log(prior_Pos)#微博label是positive的概率
    pN = sum(test_word_arrayMarkedArray * pWordsNegy) + np.log(prior_Neg)
    pNeu = sum(test_word_arrayMarkedArray * pWordsNeutral) + np.log(prior_Neutral)

    if pP > pN > pNeu or pP > pNeu > pN:
        return pP, pN, pNeu, 1 #分类为positive
    elif pN > pP > pNeu or pN > pNeu > pP:
        return pP, pN, pNeu, 2
    else:
        return pP, pN, pNeu, 3


def predict(test_word_array, test_word_arrayLabel, testCount, PosWords, NegWords, NeutralWords, prior_Pos, prior_Neg,
            prior_Neutral):
    errorCount = 0
    for j in range(testCount):#testcount：微博数量
        try:
            pP, pN, pNeu, smsType = classify(PosWords, NegWords, NeutralWords, prior_Pos, prior_Neg, prior_Neutral,
                                             test_word_array[j]) #test_word_array：微博总list
            if smsType != test_word_arrayLabel[j]: #如果预测结果与真实结果不同，预测出错；
                errorCount += 1
        except Exception as e:
            traceback.print_exc(e)
    print(errorCount / testCount) #打印错误率


if __name__ == '__main__':
    for m in range(1,11):
        vocabList = build_key_word("train/train.txt")#构建的单词表
        line_cut, label = loadDataSet("train/train.txt")#加载数据和原始标记
        train_mood_array = setOfWordsListToVecTor(vocabList, line_cut)#以单词表为标准返回line_cut文档的向量
        test_word_array = []
        test_word_arrayLabel = []
        testCount = 100  # 从中随机选取100条用来测试，并删除原来的位置
        for i in range(testCount):
            try:
                randomIndex = int(random.uniform(0, len(train_mood_array))) #uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内
                test_word_arrayLabel.append(label[randomIndex])
                test_word_array.append(train_mood_array[randomIndex])
                del (train_mood_array[randomIndex])
                del (label[randomIndex])
            except Exception as e:
                print(e)

        multi=MultinomialNB() #朴素贝叶斯分类器
        multi=multi.fit(train_mood_array,label)
        joblib.dump(multi, 'model/gnb.model') #模型的持久化，用来保存模型，然后进行评估预测等等；
        muljob=joblib.load('model/gnb.model') #加载模型
        result=muljob.predict(test_word_array) #预测
        count=0
        for i in range(len(test_word_array)):
            type=result[i] #每条微博的预测结果
            if type!=test_word_arrayLabel[i]: #预测错误的微博数量
                count=count+1
            # print(test_word_array[i], "----", result[i])
        print("mul",count/float(testCount)) #错误率
        PosWords, NegWords, NeutralWords, prior_Pos, prior_Neg, prior_Neutral = \
            trainingNaiveBayes(train_mood_array, label) #计算先验概率
        predict(test_word_array, test_word_arrayLabel, testCount, PosWords, NegWords, NeutralWords, prior_Pos, prior_Neg,
                prior_Neutral)


import pynlpir #中文分词工具
import re #处理string
from nltk.classify.scikitlearn import SklearnClassifier #nltk，简单文本分析；
from sklearn.svm import SVC, LinearSVC, libsvm, liblinear #SVC是基于libsvm实现
from sklearn.naive_bayes import MultinomialNB, BernoulliNB #sklearn中有3种NB分类器：高斯NB，MultinomialNB，BernoulliNB
from sklearn.linear_model import LogisticRegression
from random import shuffle
from nltk.probability import FreqDist, ConditionalFreqDist # -- 计算频率
    # FreqDist 继承自 dict，所以我们可以像操作字典一样操作 FreqDist 对象；
    # 布朗语料库中使用条件概率分布函数 ConditionalFreqDist，可以查看每个单词在各新闻语料中出现的次数；
from nltk.metrics import BigramAssocMeasures #用来计算信息增益
    # 要计算信息增益，首先我们需要计算每个词的频率：其整体频率及其各类别内的频率；
    # 用 FreqDist 来表示单词的整体频率；
    # ConditionalFreqDist 的条件是类别标签；
    # 用 BigramAssocMeasures.chi_sq 函数为词汇计算评分，然后按分数排序，放入一个集合里

# 文本分类之情感分析– 去除低信息量的特征

pynlpir.open()
stop = [line.strip() for line in open('ad/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词


def read_file(filename):
    f = open(filename, 'r', encoding='utf-8')
    line = f.readline()
    str = [] #处理后的文本
    while line:
        s = line
        p = re.compile(r'http?://.+$')  # 正则表达式，提取URL
        result = p.findall(line)  # 找出所有url
        if len(result):
            for i in result:
                s = s.replace(i, '')  # 一个一个的删除
        temp = pynlpir.segment(s, pos_tagging=False)  # 分词
        for i in temp:
            if '@' in i:
                temp.remove(i)  # 删除分词中的名字
        str.append(list(set(temp) - set(stop) - set('\u200b') - set(' ') - set('\u3000')))
        line = f.readline()
    return str


def pynlpir_feature(number):  # 选取number个特征词
    normalWords = []
    advWords = []
    for items in read_file('ad/normal.txt'):  # 把集合的集合变成集合 -- 所有word的集合
        for item in items:
            normalWords.append(item)
    for items in read_file('ad/advertise.txt'): # advertise-word的合集
        for item in items:
            advWords.append(item)
    word_fd = FreqDist()  # 可统计所有词的词频，是个FreqDist对象
    cond_word_fd = ConditionalFreqDist()  # 可统计正常文本中的词频和广告文本中的词频，ConditionalFreqDist对象
    for word in normalWords:
        word_fd[word] += 1
        cond_word_fd['normal'][word] += 1 # ConditionalFreqDist 的条件是类别标签；
    for word in advWords:
        word_fd[word] += 1
        cond_word_fd['adv'][word] += 1
    normal_word_count = cond_word_fd['normal'].N()  # 正常词的数量
    adv_word_count = cond_word_fd['adv'].N()  # 广告词的数量
    total_word_count = normal_word_count + adv_word_count
    word_scores = {}  # 包括了每个词和这个词的信息量
    for word, freq in word_fd.items(): #word_fd是个dict
        # 计算正常词的卡方统计量，这里也可以计算互信息等其它统计量
        normal_score = BigramAssocMeasures.chi_sq(cond_word_fd['normal'][word],
                                                  (freq, normal_word_count),
                                                  total_word_count)
        adv_score = BigramAssocMeasures.chi_sq(cond_word_fd['adv'][word],
                                               (freq, adv_word_count),
                                               total_word_count)  # 同理
        # 一个词的信息量等于正常卡方统计量加上广告卡方统计量
        word_scores[word] = normal_score + adv_score
    best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    # χ²=∑(Oi-Ei)/Ei~χ²(k-1)
    # i=1~k
    # Oi是观测值
    # Ei是期望值
    # 统计量大于临界值时,拒绝原假设

    best_words = set([w for w, s in best_vals]) #只返回best里的w；
    return dict([(word, True) for word in best_words])


def build_features(): #提取特征
    feature = pynlpir_feature(600)  # pynlpir分词，方法在上面
    normalFeatures = []
    for items in read_file('ad/normal.txt'):
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        normalWords = [a, 'normal']  # 为正常文本赋予"normal"
        normalFeatures.append(normalWords)
    advFeatures = []
    for items in read_file('ad/advertise.txt'):
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        advWords = [a, 'adv']  # 为广告文本赋予"adv"
        advFeatures.append(advWords)
    return normalFeatures, advFeatures


def traintest(string):
    str = []
    str.append(list(set(pynlpir.segment(string, pos_tagging=False)) - set(stop)))
    feature = pynlpir_feature(300)  # nlpir分词
    trainword = []
    for items in str:
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        trainword.append(a)
    return tuple(trainword)


def score(classifier, train, test): #评分函数
    classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口
    classifier.train(train)  # 训练分类器
    pred = classifier.classify_many(test)  # 对测试集的数据进行分类，给出预测的标签
    n = 0
    s = len(pred)
    for i in range(0, s):
        if pred[i] == tag[i]:
            n = n + 1   # 预测正确，则计数值加一
    print('准确度为: %f' % (n / s))
    result = n / s
    return classifier, result


if __name__ == '__main__':
    avg = 0.0
    for count in range(1, 21): #做20次计算，变化之处在于shuffle
        normalFeatures, advFeatures = build_features()  # 获得训练数据
        shuffle(normalFeatures)  # 把文本的排列随机化
        shuffle(advFeatures)  # 把文本的排列随机化
        train = normalFeatures[80:] + advFeatures[80:]  # 训练集(后80条)
        for_test = normalFeatures[:80] + advFeatures[:80]  # 预测集(验证集)(前面80条)
        test, tag = zip(*for_test)  # 分离测试集合的数据和标签，便于验证和测试

        # zip()：用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # unzip():相当于解压
        # a = [1,2,3]
        # b = [4,5,6]
        # zipped = zip(a,b)     # 打包为元组的列表 -- [(1, 4), (2, 5), (3, 6)]
        # zip(*zipped)          # 与 zip 相反，可理解为解压，返回二维矩阵式 [(1, 2, 3), (4, 5, 6)]

        # print('BernoulliNB`s accuracy is %f' % score(BernoulliNB()))
        # print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB()))
        # print('LogisticRegression`s accuracy is  %f' % score(LogisticRegression()))
        # print('SVC`s accuracy is %f' % score(SVC()))
        classifier, temp = score(LinearSVC(), train, test)
        avg = temp + avg
        # word_test = traintest(
        #     '你以为只要走的很潇洒，就不会有[汽车]太多的痛苦，就不会有留恋，可是，为什么在喧闹的人群中会突然沉默下来，为什么听歌听到一半会突然哽咽不止。你不知道自己在期待什么，不知道自己在坚持什么，脑海里挥之不去的，都是过往的倒影。')  # 预测集(验证集)(20%)
        # result = classifier.classify_many(word_test)
        # print(result)
    # print('NuSVC`s accuracy is %f' % score(NuSVC()))
    print("平均准确度为：", avg / 20)

from sklearn import datasets
from sklearn import model_selection
    # cross_validation已经不能导入了，需要改成model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = MultinomialNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', weights=[1,2,1])
    # VotingClassifier，是一种集成学习的方法，这里用 hard voting 的方式集成了三个不同的分类器
    # 可以根据其他分类器的准确度调整集成分类器的weights；

# 分别计算3种单独的分类器和集成学习后的分类器
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    # print (clf)
    print (label)
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy') #5折交叉验证后计算准确率
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
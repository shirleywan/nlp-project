import re
word="jofwjoifA级哦啊接我金佛安fewfae慰剂serge"
p = re.compile(r'\w', re.L)#去掉英文
result = p.sub("", word)
print(result)
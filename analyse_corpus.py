import nltk
from utils import word_segmentation, question_list
import matplotlib.pyplot as plt

segmented_question_list = word_segmentation(question_list)
words_in_question_list = [word for segmented_sentence in segmented_question_list for word in
                          segmented_sentence]  # 将语料库的问题列表中所有单词提取出来
len_words = len(words_in_question_list)  # 计算单词的数量（含重复）
words_freq_dict = nltk.FreqDist(words_in_question_list)  # 利用nltk库进行词频统计
num_words = words_freq_dict.B()  # 返回词典的长度，即计算单词的数量（不含重复）

print('一共出现了' + str(len_words) + '个单词')
print('一共出现了' + str(num_words) + '个不同的单词')

top_five_freq = words_freq_dict.most_common(5)  # 统计数量最多的前5个单词
print('最常见的5个单词为：' + str(top_five_freq))

word_frequencies = []
for word in words_freq_dict:
    word_frequencies.append(words_freq_dict[word])  # 将所有单词词频的数值插入列表word_frequencies

plt.plot(sorted(word_frequencies, reverse=True))  # 画出所有单词的词频情况
plt.show()

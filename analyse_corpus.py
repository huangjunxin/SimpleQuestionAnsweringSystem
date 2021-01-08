import nltk
from utils import word_segmentation, question_list
import matplotlib.pyplot as plt

# 对问题列表中的所有问句进行分词
segmented_question_list = word_segmentation(question_list)

# 将语料库的问题列表中所有单词提取出来
words_in_question_list = [word for segmented_sentence in segmented_question_list for word in segmented_sentence]
# 计算单词的数量（含重复）
len_words = len(words_in_question_list)
# 利用nltk库进行词频统计
words_freq_dict = nltk.FreqDist(words_in_question_list)
# 返回词典的长度，即计算单词的数量（不含重复）
num_words = words_freq_dict.B()

print('一共出现了' + str(len_words) + '个单词')
print('一共出现了' + str(num_words) + '个不同的单词')

# 统计数量最多的前5个单词
top_five_freq = words_freq_dict.most_common(5)
print('最常见的5个单词为：')
for word in top_five_freq:
    print(word)

word_frequencies = []
for word in words_freq_dict:
    # 将所有单词词频的数值插入列表word_frequencies
    word_frequencies.append(words_freq_dict[word])

# 画出所有单词的词频情况
plt.plot(sorted(word_frequencies, reverse=True))
plt.show()

import pickle
from utils import low_freq_words_construction, preprocessing_corpus
from analyse_corpus import question_list, words_freq_dict, segmented_question_list

# 构建低频词库
low_freq_words = low_freq_words_construction(words_freq_dict)

# 构建倒排索引表
inverted_index = {}
for word, freq in words_freq_dict.items():
    if 100 < freq < 1000:  # 将出现次数大于100小于1000的词语作为索引
        inverted_index[word] = []

for index, segmented_sentence in enumerate(segmented_question_list):
    for word in segmented_sentence:
        if word in inverted_index.keys():  # 若单词已经在倒排表中
            inverted_index[word].append(index)  # 将句子的索引值写入倒排表中

with open('model/inverted_index.pkl', 'wb') as fw:
    pickle.dump(inverted_index, fw)

# 对问题列表进行预处理
preprocessed_question_list = preprocessing_corpus(question_list, low_freq_words)

# 保存预处理后的问题列表到文件
with open('model/preprocessed_question_list.pkl', 'wb') as fw:
    pickle.dump(preprocessed_question_list, fw)

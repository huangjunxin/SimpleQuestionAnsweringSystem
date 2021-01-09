import json
import string
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# 读入语料库方法
def read_corpus():
    # 预创建两个空列表
    question_list = []
    answer_list = []
    with open('data/train-v2.0.json') as f:
        # 读入jsonString，并赋值首个data项给data_array
        data_array = json.load(f)['data']
        # 遍历data_array
        for data in data_array:
            # 赋值paragraphs项给paragraphs
            paragraphs = data['paragraphs']
            # 遍历paragraphs
            for paragraph in paragraphs:
                # 赋值qas项给qas
                qas = paragraph['qas']
                # 遍历qas
                for qa in qas:
                    # 先提取问题到问题列表
                    question_list.append(qa['question'])
                    # 对于答案，有两种情况，一种为plausible_answers
                    if 'plausible_answers' in qa:
                        answer_list.append(qa['plausible_answers'][0]['text'])
                    else:  # 另一种为answers
                        answer_list.append(qa['answers'][0]['text'])
    # 确保问题列表与答案列表的长度一样
    assert len(question_list) == len(answer_list)
    return question_list, answer_list


# 全局语料库
question_list, answer_list = read_corpus()


# 按空格分词方法
def word_segmentation(sentences):
    segmented_sentences = []
    for sentence in sentences:
        segmented_sentences.append(sentence.replace('?', '').strip().split(' '))  # 将问号删除后按空格分词
    return segmented_sentences


# 低频词库构建方法
def low_freq_words_construction(words_freq_dict):
    low_freq_words = []
    for word, freq in words_freq_dict.items():
        # 将词频低于2的词归为低频词
        if freq < 2:
            low_freq_words.append(word.lower())
    return low_freq_words


# 语料库预处理方法
def preprocessing_corpus(input_list, low_freq_words=None):
    if low_freq_words is None:
        low_freq_words = []
    # 按空格分词
    segmented_list = word_segmentation(input_list)
    # 初始化处理结果
    preprocessed_corpus = []
    for sentence in segmented_list:
        preprocessed_sentence = ''
        for word in sentence:
            # 将字母转换为小写
            word.lower()

            # 对单词进行Stemming
            ps = PorterStemmer()
            word = ps.stem(word)

            # 去除标点符号
            word = ''.join(char for char in word if char not in string.punctuation)

            # 处理数字
            if word.isdigit():
                word = '#number'

            # 引用nltk库的英语停用词
            stop_words = stopwords.words('english')

            # 排除停用词和低频词后写入处理后的句子
            if word not in stop_words and word not in low_freq_words:
                preprocessed_sentence += word + ' '
        # 删除多余空格，并存入结果
        preprocessed_corpus.append(preprocessed_sentence.strip())
    return preprocessed_corpus


# 建立GloVe矩阵
def get_glove_matrix():
    with open('data/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        glove_words = []
        glove_embeddings = []
        for line in f.readlines():
            row = list(line.split())
            glove_words.append(row[0])
            glove_embeddings.append(row[1:])

    embeddings = np.asarray(glove_embeddings)
    # 返回GloVe中的词头、矩阵化后的GloVe
    return glove_words, embeddings


# 通过GloVe模型和均值池化方法获取句向量
def get_sentence_vector(sentence, glove_words, embeddings, dimension):
    # 初始化单词索引列表
    words_index = []
    # 对已预处理的句子按空格分隔词语
    segmented_sentence = sentence.split(' ')
    # 初始化单词计数器
    words_number = 0
    for word in segmented_sentence:
        # 若句子中的单词在GloVe词头中出现
        if word in glove_words:
            # 单词计数器加一
            words_number += 1
            # 取出对应单词的索引
            index = glove_words.index(word)
            # 将当前单词的索引加入单词索引列表
            words_index.append(index)
    # 初始化句向量结果
    sentence_vector = np.zeros((dimension, ), dtype='float')
    # 根据单词索引列表提取对应的词向量列表，并进行遍历
    for embedding in embeddings[words_index]:
        sentence_vector += embedding.astype(float)
    # 若单词数大于0（避免除以零现象）
    if words_number > 0:
        sentence_vector = sentence_vector / words_number
    return sentence_vector

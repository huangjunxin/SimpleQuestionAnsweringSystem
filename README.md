# Simple Question Answering System

Training and Implementation of a Simple Question Answering System Based on Bag-of-words and GloVe Model

## ABSTRACT

With the rapid development of modern science and technology, Internet technology is changing rapidly, and the information knowledge of various industries is growing. Among them, question answering system is a representative product of artificial intelligence and natural language processing. How to extract short and accurate keywords from the question base and match the keywords entered by users has become one of the great challenges faced by many researchers.

How to accurately represent the semantic information contained in question is an important step in question matching of question answering system. In this paper, bag-of-words model and TF*IDF algorithm are used for discrete text representation, and good results are obtained initially. Later, GloVe model and mean pooling method are used for distributed text representation.

The question-and-answer corpus used in this system is SQuAD2.0 from Stanford University. This is a reading comprehension data set composed of questions and answers extracted from a group of Wikipedia articles by crowdsourcing workers. The answer of each question is a paragraph in the corresponding reading paragraph.

This paper mainly discusses the training and implementation of simple question answering system from the following aspects: first, it describes the purpose of this paper and the research status of related fields, which lays the foundation for the follow-up content of this paper; second, it clarifies the theoretical basis of this system, introduces the principle of bag of words model and GloVe model; third, according to the theory of the model, it realizes the training of simple question answering system The main Python code of the training data set and simple question answering system is introduced; fourth, the two models are optimized and tested, and the main results of this paper are summarized.

**Key words:** Question answering system; Question matching; Word vector; Natural language processing
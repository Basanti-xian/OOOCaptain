import sys
import numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

def feature(filename):
    sentiment = []
    process_sentences = []
    f = open(filename, 'rU')
    sentences = f.readlines()
    for longStr in sentences:
        sentiment.append(int(longStr[-2]))
        process_longStr = longStr[:-2]
        process_sentences.append(process_longStr)

    bigram_vectorizer = CountVectorizer(ngram_range=(1,2),
                                        token_pattern = r'\b\w+\b',
                                        stop_words='english')
    vectors = bigram_vectorizer.fit_transform(process_sentences).toarray()

#    analyze = bigram_vectorizer.build_analyzer()
    print type(vectors),vectors.shape,type(sentences)
#    print bigram_vectorizer.vocabulary_
    return vectors,sentiment


def main():
    # if len(sys.argv) != 2:
    #     print 'usage: ./task4.py file'
    #     sys.exit(1)
    # filename = sys.argv[1]
    filename = 'imdb_labelled.txt'
    vectors, sentiment = feature(filename)
    vectors_normalized = preprocessing.normalize(vectors,norm = 'l2')

    train_feature_vectors = vectors_normalized[:600, :]
    test_feature_vectors = vectors_normalized[600:, :]
    train_sentiment = sentiment[:600]
    test_sentiment = sentiment[600:]

    print 'train and test set shapes', train_feature_vectors.shape, test_feature_vectors.shape

    classifier = LinearSVC(C=1)
    classifier.fit(train_feature_vectors, train_sentiment)
    print 'i have trained my classifier to perform sentiment analysis'

    predicted_sentiment = classifier.predict(test_feature_vectors)
    acc = accuracy_score(test_sentiment, predicted_sentiment)
    print 'i have a test set accuracy of: ', acc


if __name__ == '__main__':
    main()
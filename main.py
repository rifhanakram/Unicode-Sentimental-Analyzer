# -*- coding: utf8 -*-
import nltk

# setting  own custom sinhala words to not to check.
# the words which doesnt have any negative or positive meaning.
s = open('stopwords.txt', 'r')
customWords = s.read().splitlines()

# Load positive sentences into a list
p = open('positive.txt', 'r')
postxt = p.readlines()  # List of all the lines of positive.txt file

# Load negative sentences into a list
n = open('negative.txt', 'r')
negtxt = n.readlines()  # List of all the lines of negative.txt file

# Rather than tagging each word with sentiment ,
# use "zip" function.which combines two lists into a list of tuples
neglist = []
poslist = []

# create a list of 'negatives' with the exact length of negative sentence list
for i in range(0, len(negtxt)):
    neglist.append('negative')

# Again create a list of 'positives' with the exact length of  positive sentence list
for i in range(0, len(postxt)):
    poslist.append('positive')

# Creates a list of tuples, with sentiment tagged
postagged = zip(postxt, poslist)
negtagged = zip(negtxt, neglist)

# Combines all of the tagged sentences to one large list
taggedSentences = postagged + negtagged

# now work with the individual words in the sentences - namely, getting a list
# of all of the words in the sentences, and then ordering them by the frequency in which they appear.
sentences = []  # List

# Create a list of words in the sentence, within a tuple.
for (word, sentiment) in taggedSentences:
    word_filter = [i.lower() for i in word.split()]
    sentences.append((word_filter, sentiment))


# Pull out all of the words in a list of tagged sentences, formatted in tuples
def getwords(sentences):
    allWords = []
    for (words, sentiment) in sentences:
        allWords.extend(words)  # extend() method appends the contents of seq to list
    return allWords


# Order a list of sentences by their frequency.
def getwordfeatures(listOfSentences):
    wordfreq = nltk.FreqDist(listOfSentences)  # this function always return in decreasing order
    words = wordfreq.keys()
    return words


# Calls above functions - gives us list of the words in the sentences, ordered by freq.
wordList = getwordfeatures(getwords(sentences))

# removing stopwords.txt and custom words from word list
wordList = [i for i in wordList if not i in customWords]


def feature_extractor(doc):
    docWords = set(doc)
    features = {}
    for i in wordList:
        features['contains(%s)' % i] = (i in docWords)
    return features


# creates a training set-classifier learns distribution of true/falses in the input.
training_set = nltk.classify.apply_features(feature_extractor, sentences)

# Train classifier on the training set just created.
# Based on NaiveBayes
classifier = nltk.NaiveBayesClassifier.train(training_set)

while True:
    input = raw_input("   Enter any sentence or 'exit' to quit   : ")
    if input == 'exit':
        break
    else:
        input = input.split()
        print '*********** ' + classifier.classify(
            feature_extractor(input)) + ' sentiment ************\n'

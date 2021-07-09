import pandas as pd
import re
import nltk
import warnings
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns

# Pre Proceesing: Remove Punctuations
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    resume_df = pd.read_csv(r'ResumeDataSet.csv')

    resume_df['Cleaned_Resume'] = ''

    print(resume_df['Category'].value_counts())
    resume_df['Cleaned_Resume'] = resume_df.Resume.apply(lambda x: cleanResume(x))

# Data interpretation
    plt.figure(figsize=(15, 15))
    plt.xticks(rotation=90)
    sns.countplot(y="Category", data=resume_df)

    # lemmatizing
    oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
    totalWords = []
    Sentences = resume_df['Resume'].values
    cleanedSentences = ""
    lemmatizer = WordNetLemmatizer()
    for i in range(0, 160):
        cleanedText = cleanResume(Sentences[i])
        cleanedSentences += cleanedText
        requiredWords = nltk.word_tokenize(cleanedText)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                lemmatizer.lemmatize(word)
                totalWords.append(word)

# Most common words
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(100)
    print(mostcommon)

    wc = WordCloud().generate(cleanedSentences)
    plt.figure(figsize=(15, 15))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # CONVERTEING THE CATEGORICAL VARIABLES INTO NUMERICALS
    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resume_df[i] = le.fit_transform(resume_df[i])
    print("CONVERTED THE CATEGORICAL VARIABLES INTO NUMERICALS")

    # Vectorization
    requiredText = resume_df['Cleaned_Resume'].values
    requiredTarget = resume_df['Category'].values

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)

    #knn
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("\n Classification report for classifier %s:\n%s\n" % (clf, classification_report(y_test, prediction)))

    #NB
    clf = OneVsRestClassifier(MultinomialNB()).fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print('Accuracy of MultinomialNB Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of MultinomialNB Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("\n Classification report for classifier %s:\n%s\n" % (clf, classification_report(y_test, prediction)))
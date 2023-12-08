import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

nltk.download('stopwords')

books = []

# horus heresy book series on goodreads.com
hh_url = 'https://www.goodreads.com/series/40983-the-horus-heresy'
hh_html = requests.get(hh_url)
hh_soup = BeautifulSoup(hh_html.content, 'html.parser')
# get the url key (for later) of all the books in the series
keys = hh_soup.find_all('a', class_='gr-h3--noMargin')
# get the 54 books in the series (the rest are omnibuses or subseries)
books.extend([key.get("href").strip()[11:] for key in keys][:54])

# siege of terra book series on goodreads.com
sot_url = 'https://www.goodreads.com/series/257309-the-siege-of-terra'
sot_html = requests.get(sot_url)
sot_soup = BeautifulSoup(sot_html.content, 'html.parser')
# get the url key (for later) of all the books in the series
keys = sot_soup.find_all('a', class_='gr-h3--noMargin')
# get the 9 books in the series (the rest are omnibuses or subseries)
books.extend([key.get("href").strip()[11:] for key in keys][:9])

# Uriel Ventris book series on goodreads.com
uv_url = 'https://www.goodreads.com/series/41737-ultramarines'
uv_html = requests.get(uv_url)
uv_soup = BeautifulSoup(uv_html.content, 'html.parser')
# get the url key (for later) of all the books in the series
keys = uv_soup.find_all('a', class_='gr-h3--noMargin')
# get the 6 books in the series (the rest are omnibuses or subseries)
books.extend([[key.get("href").strip()[11:] for key in keys][i] for i in [1, 3, 6, 7, 8, 9]])

df = pd.DataFrame(columns=['title', 'comment', 'rating'])

# for every book, go to the review page of that book using url key
for book in books:
    url = 'https://www.goodreads.com/book/show/' + book + '/reviews?reviewFilters={"languageCode":"en"}'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')

    title = soup.find('h1', class_='Text H1Title').text.strip()
    # get every review on the book's review page (top 30 reviews)
    reviews = soup.find_all(class_='ReviewCard__content')
    # for every review, get the rating and comment
    for review in reviews:
        # if the review is missing a rating (or comment), skip it
        try:
            rating = review.find(class_='RatingStars__small').get("aria-label").strip()[7]
            comment = review.find(class_='Formatted').text.strip()

            length = len(df)
            # remove blog link reviews (contain 'http')
            if 'http' not in comment:
                # populate a dataframe with the reviews
                df.loc[length] = [title, comment, rating]

        except AttributeError:
            pass

df.to_csv('.\RawReviews.csv', index=False)
data = pd.read_csv(r'.\RawReviews.csv')
df = pd.DataFrame(data)

stop_words = set(stopwords.words('english'))
print(df.head())
for i in range(len(df)):
    comment = df.iloc[i, 1]

    # remove all punctuation
    comment = re.sub(r'[^\w\s]', ' ', comment).strip()

    # # add title as one word to comment
    # comment = re.sub(r' ', '_', df.iloc[i, 0]) + ' ' + comment

    # convert to lower case
    comment = comment.lower()

    # convert string to list of words
    lst = [comment][0].split()
    comment = ""
    for j in lst:
        comment += j + ' '
        # # remove stopwords
        # if j not in stop_words:
        #     # convert list of words back to string
        #     comment += j + ' '

    # remove last extra space
    comment = comment[:-1]

    df.iloc[i, 1] = comment
print(df.head())

df.to_csv('.\Reviews.csv', index=False)
data = pd.read_csv(r'.\Reviews.csv')
df = pd.DataFrame(data)

tfidf = TfidfVectorizer(strip_accents='ascii', ngram_range=(1, 2), max_features=200000)
X = tfidf.fit_transform(df.iloc[:, 1])
y = df.iloc[:, 2]
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(X_train.shape)

clf = LinearSVC(class_weight='balanced', dual='auto', random_state=1).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# x = 'this book is the best ever i love it'
# vec = tfidf.transform([x])
# print('Comment:\n' + x + '\nPredicted Rating:\n' + str(clf.predict(vec)))
#
# x = 'this book is the worst ever i do not like it'
# vec = tfidf.transform([x])
# print('Comment:\n' + x + '\nPredicted Rating:\n' + str(clf.predict(vec)))
#
# x = 'this book is really bad i do not like it'
# vec = tfidf.transform([x])
# print('Comment:\n' + x + '\nPredicted Rating:\n' + str(clf.predict(vec)))

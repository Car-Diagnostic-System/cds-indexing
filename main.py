import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from kafka import KafkaConsumer
import pickle
import os
import numpy as np
import json
import string

from pythainlp.corpus import thai_words
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pandas as pd
from pythainlp import word_tokenize, subword_tokenize
from pythainlp.util import dict_trie
from scipy.sparse import hstack
from sklearn import model_selection, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
import smote_variants as sv
import boto3

s3 = boto3.resource(
    service_name='s3',
    region_name='ap-southeast-1',
    aws_access_key_id='AKIAS6J3LCRT4MC74Y75',
    aws_secret_access_key='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'
)

CONSUMER_TOPIC_NAME = "INDEX"
KAFKA_SERVER = "localhost:9092"

consumer = KafkaConsumer(
    CONSUMER_TOPIC_NAME,
    bootstrap_servers=KAFKA_SERVER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
)

def getInterview():
    interview = pd.read_excel('assets/interview.xlsx', sheet_name=None, usecols=[0, 1, 2, 3])
    for sheet in list(interview.keys()):
        if sheet == 'Form Responses 1':
            continue
        interview[sheet].columns = ['category', 'parts', 'broken_nature', 'symptoms']
    # concat sheet page
    df = pd.concat([interview[i] for i in list(interview.keys())])
    df = df.drop(['Timestamp', 'Untitled Question'], 1)
    df = df[df['symptoms'].notnull()]
    df['category'] = df['category'].ffill(0)
    df = df.reset_index(drop=True)
    return df

def customTokenDict():
    # custom words
    words = ['คลัช', 'ครัทช์', 'บู๊ช', 'ยาง', 'บน', 'หูแหนบ', 'ไส้กรอง', 'โซล่า', 'สปอร์ตไลน์', 'ยอย', 'ไดร์ชาร์จ',
             'โบลเวอร์', 'จาน', 'คลัทช์', 'หนวดกุ้ง', 'ปีกนก', 'ขาไก่', 'เพลา', 'ไทม์มิ่ง', 'ฟลายวีล', 'ปะเก็น', 'ดรัม', 'ดิส',
             'น้ำมัน', 'ดีเซล', 'เบนซิน', 'เกียร์', 'เครื่อง', 'เกียร์', 'ประเก็น', 'โอริง', 'เขม่า', 'ตามด', 'ขี้เกลือ', 'เพาเวอร์', 'เครื่อง',
             'ชาร์ฟ', 'ขุรขระ', 'กลิ่น', 'อาการ', 'สึกหรอ']
    custom_word_list = set(thai_words())
    custom_word_list.update(words)
    trie = dict_trie(dict_source=custom_word_list)
    return trie

def word_tokenizer(word, whitespace=False):
    token_word = word_tokenize(text=word, keep_whitespace=whitespace, custom_dict=trie)
    return token_word

from itertools import chain
def syllable_tokenizer(word, whitespace=False):
    syllable_word = subword_tokenize(word, engine='ssg', keep_whitespace=whitespace)
    syllable_word = [' '.join(word_tokenizer(w)).split() for w in syllable_word]
    syllable_word = list(chain.from_iterable(syllable_word))
    return syllable_word

def text_processor(text, whitespace=True):
    text = [w.lower() for w in word_tokenizer(text, whitespace)]
    text = [word.translate(str.maketrans('', '', string.punctuation + u'\xa0')) for word in text]
    text = [word for word in text if not word.isnumeric()]
    text = [word for word in text if len(word) > 1]
    text = ''.join(text)
    return text

def partsMean(dataframe):
    dist = dataframe['parts'].value_counts()
    mean_dist = dist[dist.values > dist.mean()]
    return mean_dist

def topicDict(dataframe, stop_words):
    lda_docs = dataframe['symptoms'].apply(lambda s: word_tokenizer(s))
    lda_docs = [[word.translate(str.maketrans('', '', string.punctuation + u'\xa0')) for word in doc] for doc in lda_docs]
    # lda_docs = [[word for word in doc if len(word) > 2] for doc in lda_docs]
    # remove stop thai word manually
    lda_docs = [[word for word in doc if word not in stop_words] for doc in lda_docs]
    return lda_docs

def topicExtraction(lda_dicts, dataframe):
    dictionary = Dictionary(lda_dicts)
    corpus = [dictionary.doc2bow(doc) for doc in lda_dicts]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = 60
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    temp = dictionary[0]
    id2word = dictionary.id2token

    lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)
    top_topics = lda_model.top_topics(corpus)  # , num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = " ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['dominant_topic', 'perc_contribution', 'topic_keywords']
    dataframe = pd.concat([dataframe, sent_topics_df], axis=1)
    return top_topics, dataframe

def send_email(receiver_email, body, excel):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()

    sender_email = 'cds.developer.team@gmail.com'
    sender_password = 'cdspassword'

    server.login(sender_email, sender_password)

    msg = MIMEMultipart()
    msg['Subject'] = 'Car Diagnostic System indexing process is done'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    msgText = MIMEText('<b>%s</b>' % (body), 'html')
    msg.attach(msgText)

    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
    print('Email is sent')


def upload_s3_folder(path):
    bucket = s3.Bucket('cds-bucket')
    bucket.objects.all().delete()
    directory_name = "pickles"
    bucket.put_object(Key=(directory_name + '/'))

    for subdir, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                bucket.put_object(Key=(directory_name + '/' + file), Body=data)



stop_words = ['รถ', 'เป็น', 'ที่', 'ทำให้', 'แล้ว', 'จะ', 'โดย', 'แต่',
                  'ถ้า', 'เช่น', 'คือ', 'เขา', 'ของ', 'แค่', 'และ', 'อาจ', 'ทำ', 'ให้',
                  'ว่า', 'ก็', 'หรือ', 'เพราะ', 'ที่', 'เป็น', 'ๆ']
trie = customTokenDict()
df = getInterview()
print('Load the assets successfully')

for msg in consumer:
    msg_df = pd.DataFrame(msg.value)
    msg_df.columns = ['parts', 'symptoms']

    df = pd.concat([df, msg_df], join='inner', ignore_index=True)

    mean_dist = partsMean(df)
    df = df[df['parts'].isin(mean_dist.index)]
    df = df.reset_index(drop=True)

    lda_dicts = topicDict(df, stop_words)
    lda_topic, df = topicExtraction(lda_dicts, df)

    token_vec = TfidfVectorizer(tokenizer=word_tokenizer, preprocessor=text_processor, stop_words=stop_words,
                                lowercase=True)
    syllable_vec = TfidfVectorizer(tokenizer=syllable_tokenizer, preprocessor=text_processor, stop_words=stop_words,
                                   lowercase=True)
    topic_vec = TfidfVectorizer(tokenizer=word_tokenizer, preprocessor=text_processor, stop_words=stop_words,
                                lowercase=True)

    oversampler = sv.polynom_fit_SMOTE()
    models = []

    X_token = token_vec.fit_transform(df['symptoms'])
    X_syllable = syllable_vec.fit_transform(df['symptoms'])
    X_topic = topic_vec.fit_transform(df['topic_keywords'])

    X = hstack([X_token, X_syllable, X_topic])


    for part in mean_dist.index:
        print('Train {} model'.format(part))
        train_df = df.copy()
        train_df['parts'] = np.where(train_df['parts'] == part, 1, 0)
        y = train_df['parts']
        # Oversampling dataset
        X_samp, y_samp = oversampler.sample(X.todense(), y)
        X_fit, X_test, y_fit, y_test = model_selection.train_test_split(X_samp, y_samp, test_size=0.2, random_state=42)
        # GradientBoosting
        gb_model = ensemble.GradientBoostingClassifier()
        m = gb_model.fit(X_fit, y_fit)
        models.append({'part': part, 'model': m})



    if not os.path.exists('pickles'):
        os.makedirs('pickles')
    pickle.dump(trie, open('pickles/trie.pkl', 'wb'))
    pickle.dump(models, open('pickles/models.pkl', 'wb'))
    pickle.dump(token_vec, open('pickles/token_vec.pkl', 'wb'))
    pickle.dump(syllable_vec, open('pickles/syllable_vec.pkl', 'wb'))
    pickle.dump(topic_vec, open('pickles/topic_vec.pkl', 'wb'))
    pickle.dump(lda_topic, open('pickles/lda_topic.pkl', 'wb'))

    upload_s3_folder('pickles')
    send_email('tulyawatt@gmail.com', 'The indexing process is done.', msg_df)





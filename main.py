import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from itertools import chain

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

class Index:
    @staticmethod
    def getInterview(path):
        interview = pd.read_excel(path, sheet_name=None, usecols=[0, 1, 2, 3])
        for sheet in list(interview.keys()):
            if sheet == 'Form Responses 1':
                continue
            interview[sheet].columns = ['category', 'parts', 'broken_nature', 'symptoms']
        # concat sheet page
        df = pd.concat([interview[i] for i in list(interview.keys())])
        df = df.drop(['Timestamp', 'Untitled Question'], 1)
        df = df[df['symptoms'].notnull()]
        df['category'] = df['category'].ffill(0)
        dist = df['parts'].value_counts()
        mean_dist = dist[dist.values > dist.mean()]
        df = df[df['parts'].isin(mean_dist.index)].reset_index(drop=True)
        return df

    @staticmethod
    def getDictTrie():
        # custom words
        words = ['คลัช', 'ครัทช์', 'บู๊ช', 'ยาง', 'บน', 'หูแหนบ', 'ไส้กรอง', 'โซล่า', 'สปอร์ตไลน์', 'ยอย', 'ไดร์ชาร์จ',
                 'โบลเวอร์', 'จาน', 'คลัทช์', 'หนวดกุ้ง', 'ปีกนก', 'ขาไก่', 'เพลา', 'ไทม์มิ่ง', 'ฟลายวีล', 'ปะเก็น', 'ดรัม', 'ดิส',
                 'น้ำมัน', 'ดีเซล', 'เบนซิน', 'เกียร์', 'เครื่อง', 'เกียร์', 'ประเก็น', 'โอริง', 'เขม่า', 'ตามด', 'ขี้เกลือ', 'เพาเวอร์', 'เครื่อง',
                 'ชาร์ฟ', 'ขุรขระ', 'กลิ่น', 'อาการ', 'สึกหรอ', 'ผ้าเบรค', 'แป้นเบรค']
        custom_word_list = set(thai_words())
        custom_word_list.update(words)
        trie = dict_trie(dict_source=custom_word_list)
        return trie

    @staticmethod
    def topicExtraction(dataframe):
        lda_dicts = dataframe['symptoms'].apply(lambda s: word_tokenize(text=s, keep_whitespace=False, custom_dict=trie))
        lda_dicts = [[word.translate(str.maketrans('', '', string.punctuation + u'\xa0')) for word in doc] for doc in
                    lda_dicts]
        lda_dicts = [[word for word in doc if word not in stop_words] for doc in lda_dicts]

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
        top_topics = lda_model.top_topics(corpus)

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
        return dataframe

    @staticmethod
    def sendEmail(receiver_email, body):
        try:
            server = smtplib.SMTP('smtp.outlook.com:587')
            server.ehlo()
            server.starttls()

            sender_email = 'cds.developer.team@outlook.com'
            sender_password = 'Cdspassword'

            server.login(sender_email, sender_password)

            msg = MIMEMultipart()
            msg['Subject'] = 'Car Diagnostic System indexing process is done'
            msg['From'] = sender_email
            msg['To'] = receiver_email

            msgText = MIMEText('<b>%s</b>' % (body), 'html')
            msg.attach(msgText)

            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            return 'Email is sent successfully'
        except:
            return 'Email is sent failed'

    @staticmethod
    def createWordVec(symptoms):
        word_vec = TfidfVectorizer(tokenizer=word_tokenizer, preprocessor=text_processor, stop_words=stop_words, lowercase=True)
        X_word = word_vec.fit_transform(symptoms)
        return word_vec, X_word

    @staticmethod
    def createSyllableVec(symptoms):
        syllable_vec = TfidfVectorizer(tokenizer=syllable_tokenizer, preprocessor=text_processor, stop_words=stop_words, lowercase=True)
        X_syllable = syllable_vec.fit_transform(symptoms)
        return syllable_vec, X_syllable

    @staticmethod
    def createTopicVec(topics):
        topic_vec = TfidfVectorizer(tokenizer=word_tokenizer, preprocessor=text_processor, stop_words=stop_words, lowercase=True)
        X_topic = topic_vec.fit_transform(topics)
        return topic_vec, X_topic


    @staticmethod
    def upload_s3_folder(path, bucket_name):
        bucket = s3.Bucket(bucket_name)
        try:
            bucket.objects.all().delete()
            bucket.put_object(Key=(path + '/'))

            for subdir, dirs, files in os.walk(path):
                for file in files:
                    full_path = os.path.join(subdir, file)
                    with open(full_path, 'rb') as data:
                        bucket.put_object(Key=(path + '/' + file), Body=data)
            return 'Upload is completed'
        except:
            return "Upload is failed"

    @staticmethod
    def trainModel(part, X):
        train_df = df.copy()
        train_df['parts'] = np.where(train_df['parts'] == part, 1, 0)
        y = train_df['parts']
        # Oversampling dataset
        X_samp, y_samp = oversampler.sample(X.todense(), y)
        X_fit, X_test, y_fit, y_test = model_selection.train_test_split(X_samp, y_samp, test_size=0.2, random_state=42)
        # GradientBoosting
        gb_model = ensemble.GradientBoostingClassifier()
        m = gb_model.fit(X_fit, y_fit)
        return m

def word_tokenizer(text, whitespace=False):
    token_word = word_tokenize(text=text, keep_whitespace=whitespace, custom_dict=trie)
    return token_word

def syllable_tokenizer(text, whitespace=False):
    syllable_word = subword_tokenize(text, engine='ssg', keep_whitespace=whitespace)
    syllable_word = [word_tokenize(text=w, keep_whitespace=whitespace, custom_dict=trie) for w in syllable_word]
    syllable_word = list(chain.from_iterable(syllable_word))
    return syllable_word

def text_processor(text, whitespace=True):
    text = [w.lower() for w in word_tokenizer(text, whitespace)]
    text = [word.translate(str.maketrans('', '', string.punctuation + u'\xa0')) for word in text]
    # NOTE: Remove number from text ***may be used
    # text = [word for word in text if not word.isnumeric()]
    text = ''.join(text)
    return text

stop_words = ['รถ', 'เป็น', 'ที่', 'ทำให้', 'แล้ว', 'จะ', 'โดย', 'แต่',
              'ถ้า', 'เช่น', 'คือ', 'เขา', 'ของ', 'แค่', 'และ', 'อาจ', 'ทำ', 'ให้',
              'ว่า', 'ก็', 'หรือ', 'เพราะ', 'ที่', 'เป็น', 'ๆ']
trie = Index.getDictTrie()
df = Index.getInterview('assets/interview.xlsx')
oversampler = sv.polynom_fit_SMOTE()

if __name__ == '__main__':
    print('Load the assets successfully')

    for msg in consumer:
        print("Indexing process is start")
        msg_df = pd.DataFrame(msg.value)
        msg_df.columns = ['parts', 'symptoms']

        df = pd.concat([df, msg_df], join='inner', ignore_index=True)
        # NOTE: Find the parts occurrence that more than mean

        df = Index.topicExtraction(df)

        word_vec, X_word = Index.createWordVec(df['symptoms'])
        syllable_vec, X_syllable = Index.createSyllableVec(df['symptoms'])
        topic_vec, X_topic = Index.createTopicVec(df['topic_keywords'])

        X = hstack([X_word, X_syllable, X_topic])

        models = []
        for part in df['parts'].value_counts().index:
            print('Train {} model'.format(part))
            m = Index.trainModel(part)
            models.append({'part': part, 'model': m})

        if not os.path.exists('pickles'):
            os.makedirs('pickles')
        pickle.dump(trie, open('pickles/trie.pkl', 'wb'))
        pickle.dump(models, open('pickles/models.pkl', 'wb'))
        pickle.dump(word_vec, open('pickles/word_vec.pkl', 'wb'))
        pickle.dump(syllable_vec, open('pickles/syllable_vec.pkl', 'wb'))
        pickle.dump(topic_vec, open('pickles/topic_vec.pkl', 'wb'))

        Index.upload_s3_folder('pickles', 'cds-bucket')
        Index.sendEmail('tulyawatt@gmail.com', 'The indexing process is done.')





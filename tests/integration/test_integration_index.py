from unittest import TestCase
from main import *

class TestIntegrationIndex(TestCase):
    @classmethod
    def setUp(cls):
        cls.df = Index.getInterview('../../assets/interview.xlsx')

    def test_topic_extraction(self):
        df = Index.topicExtraction(self.df)
        self.assertEqual((1665, 7), df.shape)

    def test_upload_s3_folder(self):
        response = Index.upload_s3_folder('pickles', 'cds-bucket')
        self.assertEqual('Upload is completed', response)


    def test_create_word_vec(self):
        word_vec, X_word = Index.createWordVec(self.df['symptoms'])
        self.assertEqual('TfidfVectorizer', word_vec.__class__.__name__)
        self.assertEqual('csr_matrix', X_word.__class__.__name__)


    def test_create_syllable_vec(self):
        syllable_vec, X_syllable = Index.createSyllableVec(self.df['symptoms'])
        self.assertEqual('TfidfVectorizer', syllable_vec.__class__.__name__)
        self.assertEqual('csr_matrix', X_syllable.__class__.__name__)

    def test_create_topic_vec(self):
        df = Index.topicExtraction(self.df)
        topic_vec, X_topic = Index.createTopicVec(df['topic_keywords'])
        self.assertEqual('TfidfVectorizer', topic_vec.__class__.__name__)
        self.assertEqual('csr_matrix', X_topic.__class__.__name__)

    def test_train_model(self):
        df = Index.topicExtraction(self.df)

        word_vec, X_word = Index.createWordVec(df['symptoms'])
        syllable_vec, X_syllable = Index.createSyllableVec(df['symptoms'])
        topic_vec, X_topic = Index.createTopicVec(df['topic_keywords'])
        X = hstack([X_word, X_syllable, X_topic])

        model = Index.trainModel('ผ้าดิสเบรค', X)
        self.assertEqual('GradientBoostingClassifier', model.__class__.__name__)

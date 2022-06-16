from unittest import TestCase
from main import *
from main import word_tokenizer, syllable_tokenizer, text_processor

class TestIndex(TestCase):
    @classmethod
    def setUp(cls):
        cls.df = Index.getInterview()
        cls.trie = Index.getDictTrie()
        cls.stop_words = ['รถ', 'เป็น', 'ที่', 'ทำให้', 'แล้ว', 'จะ', 'โดย', 'แต่',
                      'ถ้า', 'เช่น', 'คือ', 'เขา', 'ของ', 'แค่', 'และ', 'อาจ', 'ทำ', 'ให้',
                      'ว่า', 'ก็', 'หรือ', 'เพราะ', 'ที่', 'เป็น', 'ๆ']


    def test_word_tokenizer(self):
        actual_one = word_tokenizer('กินผ้าเบรค เวลาเบรคจะสะท้านที่แป้นเบรค', False)
        actual_two = word_tokenizer('รถกินน้ำมัน มากกว่าปกติ', True)
        actual_three = word_tokenizer('Start รถไม่ติด', False)

        self.assertEqual(['กิน', 'ผ้าเบรค', 'เวลา', 'เบรค', 'จะ', 'สะท้าน', 'ที่', 'แป้นเบรค'], actual_one)
        self.assertEqual(['รถ', 'กินน้ำมัน', ' ', 'มากกว่า', 'ปกติ'], actual_two)
        self.assertEqual(['Start', 'รถ', 'ไม่', 'ติด'], actual_three)

    def test_syllable_tokenizer(self):
        actual_one = syllable_tokenizer('น้ำมันเกียร์ รั่วซึม', False)
        actual_two = syllable_tokenizer('น้ำมันเกียร์ รั่วซึม', True)
        actual_three = syllable_tokenizer('มีไฟ show ที่หน้าปัด', False)
        self.assertEqual(['น้ำ', 'มัน', 'เกียร์', 'รั่ว', 'ซึม'], actual_one)
        self.assertEqual(['น้ำ', 'มัน', 'เกียร์', ' ', 'รั่ว', 'ซึม'], actual_two)
        self.assertEqual(['มี', 'ไฟ', 'show', 'ที่', 'หน้า', 'ปัด'], actual_three)

    def test_text_processor(self):
        actual_one = text_processor('มีไฟ Show (ที่หน้าปัด)', False)
        actual_two = text_processor('มีไฟ Show (ที่หน้าปัด)', True)

        self.assertEqual('มีไฟshowที่หน้าปัด', actual_one)
        self.assertEqual('มีไฟ show ที่หน้าปัด', actual_two)


    def test_get_interview(self):
        df = Index.getInterview()
        self.assertEqual((1665, 4), df.shape)

    def test_get_dict_trie(self):
        trie = Index.getDictTrie()
        self.assertEqual("pythainlp.util.trie", trie.__module__)

    def test_topic_extraction(self):
        df = Index.topicExtraction(self.df)
        self.assertEqual((1665, 7), df.shape)

    def test_send_email(self):
        response = Index.sendEmail('tulywatt@gmail.com', 'test send email')
        self.assertEqual('Email is sent successfully',response)

    def test_upload_s3_folder(self):
        response = Index.upload_s3_folder('pickles', 'cds-bucket')
        self.assertEqual('Upload is completed', response)

    def test_create_word_vec(self):
        word_vec, X_word = Index.createWordVec(self.df['symptoms'])
        self.assertEqual('sklearn.feature_extraction.text', word_vec.__module__)
        self.assertEqual('scipy.sparse.csr', X_word.__module__)


    def test_create_syllable_vec(self):
        syllable_vec, X_syllable = Index.createSyllableVec(self.df['symptoms'])
        self.assertEqual('sklearn.feature_extraction.text', syllable_vec.__module__)
        self.assertEqual('scipy.sparse.csr', X_syllable.__module__)

    def test_create_topic_vec(self):
        df = Index.topicExtraction(self.df)
        topic_vec, X_topic = Index.createTopicVec(df['topic_keywords'])
        self.assertEqual('sklearn.feature_extraction.text', topic_vec.__module__)
        self.assertEqual('scipy.sparse.csr', X_topic.__module__)


    def test_train_model(self):
        df = Index.topicExtraction(self.df)

        word_vec, X_word = Index.createWordVec(df['symptoms'])
        syllable_vec, X_syllable = Index.createSyllableVec(df['symptoms'])
        topic_vec, X_topic = Index.createTopicVec(df['topic_keywords'])
        X = hstack([X_word, X_syllable, X_topic])
        model = Index.trainModel('ผ้าดิสเบรค', X)

        self.assertEqual('sklearn.ensemble._gb', model.__module__)

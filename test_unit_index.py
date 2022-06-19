from unittest import TestCase
from main import *
from main import word_tokenizer, syllable_tokenizer, text_processor

class TestUnitIndex(TestCase):
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
        df = Index.getInterview('assets/interview.xlsx')
        self.assertEqual((1665, 4), df.shape)

    def test_get_dict_trie(self):
        trie = Index.getDictTrie()
        self.assertEqual("Trie", trie.__class__.__name__)

    def test_send_email(self):
        response = Index.sendEmail('tulywatt@gmail.com', 'test send email')
        self.assertEqual('email sent successfully', response)

    def test_upload_s3_folder(self):
        actual_one = Index.upload_s3_folder('pickles', 'cds-bucket')
        actual_two = Index.upload_s3_folder('pickles', 'non-existed')
        self.assertEqual('upload successful', actual_one)
        self.assertEqual('upload failed', actual_two)

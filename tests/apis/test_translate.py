import unittest
from mleus.apis.translate import translate


class TestTranslate(unittest.TestCase):

    def test_translate(self):
        english = "My name is Ahmed"
        arabic = "اسمي أحمد"
        translated = translate(english)

        self.assertEqual(arabic, translated)


if __name__ == '__main__':
    unittest.main()

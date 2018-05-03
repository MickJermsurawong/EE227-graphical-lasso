import nltk

nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
from nltk.stem.snowball import SnowballStemmer


class Featurizer(object):
    def __init__(self):

        self.stemmer = SnowballStemmer('english', ignore_stopwords=True)
        self.tkn = tokenize.TreebankWordTokenizer()
        self.text_count_v = CountVectorizer(stop_words='english', min_df=50, tokenizer=self.my_tokenizer)

        self.white_space_tkn = tokenize.WhitespaceTokenizer().tokenize
        self.source_count_v = CountVectorizer(lowercase=False, tokenizer=self.white_space_tkn)
        self.person_count_v = CountVectorizer(lowercase=False, tokenizer=self.white_space_tkn)

    def fit_text_one_hot(self, total_introductions):
        texts = Featurizer.get_attr_val_from_introductions('text', total_introductions)
        transformed = self.text_count_v.fit_transform(texts)
        return transformed.toarray(), self.text_count_v.get_feature_names()

    def fit_source_one_hot(self, total_introductions):
        sources = Featurizer.get_attr_val_from_introductions('source', total_introductions)
        transformed = self.source_count_v.fit_transform(sources)
        return transformed.toarray(), self.source_count_v.get_feature_names()

    def fit_person_one_hot(self, total_introductions):
        persons = Featurizer.get_attr_val_from_introductions('person', total_introductions)
        transformed = self.person_count_v.fit_transform(persons)
        return transformed.toarray(), self.person_count_v.get_feature_names()

    @staticmethod
    def get_attr_val_from_introductions(attr, total_introductions):
        return [intro[attr] for intro in total_introductions]

    def my_tokenizer(self, sen):
        ts = self.tkn.tokenize(sen)
        res = []
        for t in ts:
            if t.isalpha():
                res.append(t)

        return [self.stemmer.stem(r) for r in res]

    def counter_global(self, attribute, introduction_array):
        count_dict = {}
        attr_val = Featurizer.get_attr_val_from_introductions(attribute, introduction_array)
        for v in attr_val:
            count_dict.setdefault(v, 0)
            count_dict[v] += 1
        return count_dict

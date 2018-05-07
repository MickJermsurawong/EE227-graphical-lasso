import nltk

nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import numpy as np


class Featurizer(object):

    def __init__(self):

        self.stemmer = SnowballStemmer('english', ignore_stopwords=True)
        self.tkn = tokenize.TreebankWordTokenizer()

        self.fit_text_title = False
        self.text_title_count_v = CountVectorizer(stop_words='english', min_df=50, tokenizer=self.my_tokenizer)

        self.white_space_tkn = tokenize.WhitespaceTokenizer().tokenize
        self.source_count_v = CountVectorizer(lowercase=False, tokenizer=self.white_space_tkn)
        self.person_count_v = CountVectorizer(lowercase=False, tokenizer=self.white_space_tkn)

    def fit_text_and_title(self, total_introductions):

        if self.fit_text_title:
            return
        else:
            texts = Featurizer.get_attr_val_from_introductions('text', total_introductions)
            titles = Featurizer.get_attr_val_from_introductions('title', total_introductions)
            self.text_title_count_v.fit(texts + titles)
            self.fit_text_title = True
            return

    def fit_text_one_hot(self, total_introductions):
        self.fit_text_and_title(total_introductions)
        texts = Featurizer.get_attr_val_from_introductions('text', total_introductions)
        transformed = self.text_title_count_v.transform(texts)
        return transformed.toarray(), self.text_title_count_v.get_feature_names()

    def fit_title_one_hot(self, total_introductions):
        self.fit_text_and_title(total_introductions)
        titles = Featurizer.get_attr_val_from_introductions('title', total_introductions)
        transformed = self.text_title_count_v.transform(titles)
        return transformed.toarray(), self.text_title_count_v.get_feature_names()

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

    @staticmethod
    def transfer_val(from_intros, to_intros, from_keys, to_keys, assertive_key, backup_key):
        assert len(from_intros) == len(to_intros), "From and to intros should be the same size"
        assert len(from_keys) == len(to_keys), "From and to keys should be the same size"

        for i in range(len(from_intros)):
            f_intro = from_intros[i]
            to_intro = to_intros[i]
            if assertive_key in f_intro:
                assert f_intro[assertive_key] == to_intro[assertive_key]
            else:
                assert f_intro[backup_key] == to_intro[backup_key]
            for j in range(len(from_keys)):
                f_key = from_keys[j]
                t_key = to_keys[j]
                to_intro[t_key] = f_intro[f_key]


def count_by_key(introductions, key):
    c = Counter()
    for i in introductions:
        c[i[key]] += 1
    return c

def adjust_imbalance_pca(data, introductions, source_weights):
    """

    :param data: matrix to scale row wise
    :param introductions: intro arrays
    :param source_weights: map source to real number weights
    :return:
    """
    assert data.shape[0] == len(introductions)
    scaled_data = np.zeros_like(data, dtype=np.float32)

    for i in range(len(introductions)):
        source = introductions[i]['source']
        w = source_weights[source]
        scaled_data[i,:] = data[i, :] * w
    return scaled_data


def random_indx_same_proportions(introductions, n):

    idx_and_sources = [(idx, intro['source']) for idx, intro in enumerate(introductions)]

    source_idx_map = {}
    for i_s in idx_and_sources:
        source_idx_map.setdefault(i_s[1], [])
        source_idx_map[i_s[1]].append(i_s[0])

    K = len(source_idx_map)
    n_k = int(n/K)

    ran_idx = np.zeros((K, n_k), dtype=np.int32)

    for i, s in enumerate(source_idx_map):
        ran_idx[i, :] = np.random.choice(source_idx_map[s], n_k)
    return ran_idx.reshape(-1)





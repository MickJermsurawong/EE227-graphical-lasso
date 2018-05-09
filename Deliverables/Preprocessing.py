
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import PolynomialFeatures


def filter_introductions(total_introductions,
                         text_one_hot_filter,
                         count_person_dict, min_person,
                         min_word):
    kept_introductions = []
    for i in tqdm(range(len(total_introductions))):
        intro = total_introductions[i]
        person = intro['person']
        if count_person_dict[person] < min_person:
            continue
        if np.sum(text_one_hot_filter[i]) < min_word:
            continue
        kept_introductions.append(intro)
    print("Total mentions", len(kept_introductions))



class OneHotFeature(object):

    def __init__(self, introductions, featurizer):

        self.text_one_hot, self.text_feats = featurizer.fit_text_one_hot(introductions)
        self.title_one_hot, self.title_feats = featurizer.fit_title_one_hot(introductions)
        self.source_one_hot, self.source_feats = featurizer.fit_source_one_hot(introductions)
        self.person_one_hot, self.person_feats = featurizer.fit_person_one_hot(introductions)

        self.text_feats = ["txt:" + t for t in self.text_feats]
        self.title_feats = ["title:" + t for t in self.title_feats]

        self.all_one_hot = [self.text_one_hot, self.title_one_hot, self.source_one_hot, self.person_one_hot]
        self.all_feats = [self.text_feats, self.title_feats, self.source_feats, self.person_feats]

        self.code = {"text": 0, "title": 1, "source": 2, "person": 3}

        assert self.text_one_hot.shape[0] == self.source_one_hot.shape[0]
        assert self.source_one_hot.shape[0] == self.person_one_hot.shape[0]
        assert self.person_one_hot.shape[0] == self.title_one_hot.shape[0]

        print(f"Text one hot: {self.text_one_hot.shape}")
        print(f"Title one hot: {self.title_one_hot.shape}")
        print(f"Source one hot: {self.source_one_hot.shape}")
        print(f"Person one hot: {self.person_one_hot.shape}")

    def get_text_one_hot(self):
        return self.get_one_hot('text')

    def get_title_one_hot(self):
        return self.get_one_hot('title')

    def get_source_one_hot(self):
        return self.get_one_hot('source')

    def get_person_one_hot(self):
        return self.get_one_hot('person')

    def get_combined(self, combinations):
        combi_one_hot = []
        combi_feats = []
        for c in combinations:
            one_hot, feats = self.get_one_hot(c)
            combi_one_hot.append(one_hot)
            combi_feats += feats
        combi_one_hot = np.hstack(combi_one_hot)
        assert combi_one_hot.shape[1] == len(combi_feats)
        return combi_one_hot, combi_feats


    def get_one_hot(self, c):
        if c in self.code:
            one_hot = np.copy(self.all_one_hot[self.code[c]])
            feats = [f for f in self.all_feats[self.code[c]]]
            assert one_hot.shape[1] == len(feats)
            return one_hot, feats
        else:
            raise Exception(f"Code {c} is not supported.. Only {self.code} are supported.")



class SentimentTarget(object):

    def __init__(self, introductions, sentiment_keys=['sentiment', 'liu_sentimtnet', 'title_sentiment']):


        def pack_sentiments(intro, sentiment_keys):
            return [intro[k] for k in sentiment_keys]

        self.sentiment_keys_dict = {v: i for i, v in enumerate(sentiment_keys)}
        self.all_sentiments = np.array([pack_sentiments(intro, sentiment_keys) for intro in introductions])

        print(f"Sentiments: {self.all_sentiments.shape}")

    def get_sentiment(self, sentiment_keys):

        selector = np.array([self.sentiment_keys_dict[k] for k in sentiment_keys])

        return np.copy(self.all_sentiments[:,selector]), sentiment_keys

    def get_poly(self, sentiment_keys, degree=2):
        sentiments, s_feats = self.get_sentiment(sentiment_keys)

        poly = PolynomialFeatures(degree)
        transformed = poly.fit_transform(sentiments)

        poly_feats = []
        for feat in poly.powers_:
            f = ""
            for i, d in enumerate(feat):
                if d == 0:
                    continue
                elif d == 1:
                    f += str(s_feats[i])
                else:
                    f += str(s_feats[i]) + "^" + str(d)
            poly_feats.append(f)

        return transformed, poly_feats









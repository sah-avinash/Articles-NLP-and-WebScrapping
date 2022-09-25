"""
------------------BlackCoffer Assessment------------------------
    Web-Scrapping, Text Analysis & Pre-processing, NLP.
----------------------------------------------------------------
p.s.
This does both web-scrapping and text processing, to use/run the
web-scrapping uncomment the call for "get_and_save_text_data()".
"""

# Required Libraries
import pandas as pd
import re
import os

# for Web-Scrapping
from selenium import webdriver

# for Text Analysis and Processing
from textstat.textstat import textstatistics
from textblob import TextBlob
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# get the required data,
# available in (Data\text_data.csv) or can be scrapped new.
class GetData:
    def __init__(self, urls):
        self.Urls = urls

    def get_data_from_file(self):
        data = pd.read_csv(r"Data\text_data.csv")["0"].to_list()
        return data

    def get_data_from_url(self):
        path = r"chromedriver.exe"

        text_data = []
        for url in self.Urls:
            driver = webdriver.Chrome(path)
            driver.get(url)
            url_text = driver.find_element_by_xpath("//div[@class='td-post-content']").text.replace("\n", " ")
            text_data.append(url_text)
            driver.quit()
        return text_data

    # The files have some unwanted text.
    # StopWords are collected and cleaned here.
    def get_stopwords_from_file(self):
        files = os.listdir(r'Data\StopWords')
        base = r'Data\StopWords'
        stop_words = []
        for file in files:
            file_path = os.path.join(base, file)
            with open(file_path, 'r') as fr:
                sr = fr.read()
                cleaned_file = [re.sub("[|].*", "", s).strip().lower() for s in sr.split("\n")]
                stop_words.extend(cleaned_file[:-1])
        return set(stop_words)


# Collection of processes to get all features required.
# All features are rounded-off to the nearest thousandth (1e-4).
class CalculateFeatures(GetData):
    def __init__(self, urls):
        super().__init__(urls)
        self.text_data = self.get_data_from_file()
        self.stop_words = self.get_stopwords_from_file()

    def get_pn_tokens(self):
        with open(r"Data/MasterDictionary/positive-words.txt", 'r') as f:
            pov_tokens = f.read().split()
        with open(r"Data/MasterDictionary/negative-words.txt", 'r') as f:
            neg_tokens = f.read().split()
        return pov_tokens, neg_tokens

    def get_personal_pronouns(self, text):
        pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b', re.I)
        personal_pronouns = len(pronounRegex.findall(text))
        return personal_pronouns

    def get_word_count_and_avg_length(self, text):
        word_count = len(text.split())
        word_lengths = [len(word) for word in text.split()]
        avg_word_length = round(sum(word_lengths) / len(text.split()), 4)
        return word_count, avg_word_length

    def get_syllable_per_word(self, text):
        syllable = [textstatistics().syllable_count(word) for word in text.split()]
        syllable_per_word = round(sum(syllable) / len(text.split()), 4)
        return syllable_per_word

    def get_sentence_length_and_avg_word(self, sentences):
        sent_lengths = [len(sent) for sent in sentences]
        avg_sent_length = round(sum(sent_lengths) / len(sent_lengths), 4)
        word_count = [len(sent.split()) for sent in sentences]
        avg_word_length = round(sum(word_count) / len(sent_lengths), 4)
        return avg_sent_length, avg_word_length

    def complex_words(self, text):
        complex_words = set()
        for word in text.split():
            syllable_count = textstatistics().syllable_count(word)
            if word not in self.stop_words and syllable_count >= 2:
                complex_words.add(word)
        return len(complex_words)

    def get_fog_index(self, text):
        per_complex_words = (self.complex_words(text) / self.get_word_count_and_avg_length(text)[0] * 100) + 5
        grade = 0.4 * (self.get_sentence_length_and_avg_word(text)[0] + per_complex_words)
        return round(grade, 4)

    def get_polarity_and_subjectivity(self, text):
        blobed_text = TextBlob(text)
        return blobed_text.sentiment

    def get_word_lemma(self, text):
        lemmatizer = WordNetLemmatizer()
        text = text.lower()
        text = text.split()
        word_lemma = [lemmatizer.lemmatize(word) for word in text if word not in self.stop_words]
        word_lemma = ' '.join(word_lemma)
        return word_lemma

    def get_pn_score(self, text, p_tokens, n_tokens):
        corpus = [word for word in text.split() if word not in self.stop_words]
        p_words = [word for word in corpus if word in p_tokens]
        n_words = [word for word in corpus if word in n_tokens]

        p_score = round(len(p_words) / len(corpus), 4)
        n_score = round(len(n_words) / len(corpus), 4)

        return p_score, n_score

    def get_features(self):
        features = {
            "URL": [],
            "POSITIVE SCORE": [],
            "NEGATIVE SCORE": [],
            "POLARITY SCORE": [],
            "SUBJECTIVITY SCORE": [],
            "AVG SENTENCE LENGTH": [],
            "PERCENTAGE OF COMPLEX WORDS": [],
            "FOG INDEX": [],
            "AVG NUMBER OF WORDS PER SENTENCE": [],
            "COMPLEX WORD COUNT": [],
            "WORD COUNT": [],
            "SYLLABLE PER WORD": [],
            "PERSONAL PRONOUNS": [],
            "AVG WORD LENGTH": []
        }

        p_tokens, n_tokens = self.get_pn_tokens()

        features["URL"].extend(Urls)
        for text in self.text_data:
            text_cleaned1 = re.sub("[^a-zA-Z' ]+", ' ', text)
            text_cleaned2 = re.sub("[^a-zA-Z'. ]+", ' ', text)

            pronoun_count = self.get_personal_pronouns(text_cleaned1)
            features["PERSONAL PRONOUNS"].append(pronoun_count)

            word_count, avg_word_length = self.get_word_count_and_avg_length(text_cleaned1)
            features["WORD COUNT"].append(word_count)
            features["AVG WORD LENGTH"].append(avg_word_length)

            syllable_per_word = self.get_syllable_per_word(text_cleaned1)
            features["SYLLABLE PER WORD"].append(syllable_per_word)

            sentences = nltk.sent_tokenize(text_cleaned2)
            avg_sent_length, avg_word_per_sent = self.get_sentence_length_and_avg_word(sentences)
            features["AVG SENTENCE LENGTH"].append(avg_sent_length)
            features["AVG NUMBER OF WORDS PER SENTENCE"].append(avg_word_per_sent)

            complex_word_count = []
            for sentence in sentences:
                sentence = sentence.lower()
                complex_word_count.append(self.complex_words(sentence))
            features["COMPLEX WORD COUNT"].append(sum(complex_word_count))
            features["PERCENTAGE OF COMPLEX WORDS"].append(round(sum(complex_word_count) / word_count * 100, 4))

            fog_idx = self.get_fog_index(text_cleaned1)
            features["FOG INDEX"].append(fog_idx)

            polarity, subjectivity = self.get_polarity_and_subjectivity(text_cleaned1)
            features["POLARITY SCORE"].append(round(polarity, 4))
            features["SUBJECTIVITY SCORE"].append(round(subjectivity, 4))

            word_lemma = self.get_word_lemma(text_cleaned1)
            p_score, n_score = self.get_pn_score(word_lemma, p_tokens, n_tokens)
            features["POSITIVE SCORE"].append(p_score)
            features["NEGATIVE SCORE"].append(n_score)

        return features


Urls = pd.read_excel(r"Data/Input.xlsx", engine='openpyxl')["URL"]


def get_and_save_text_data():
    get_data = GetData(Urls)
    text_data = get_data.get_data_from_url()
    text_data = pd.DataFrame(text_data)
    pd.DataFrame.to_csv(text_data, "text_data.csv", encoding='utf-8')


#Uncomment this to scrape data from Urls and not use the saved file.
#get_and_save_text_data()

output = CalculateFeatures(Urls)
output_features = output.get_features()

output_features = pd.DataFrame(output_features)
output_features.to_excel(r"Data\Output.xlsx", engine='openpyxl')

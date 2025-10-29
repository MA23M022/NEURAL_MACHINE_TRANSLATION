from __future__ import unicode_literals, print_function, division
import pandas as pd
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def normalizeString(s):
    s = s.lower().strip()
    s = s[:-1]
    #s = re.sub(r"([.!?])", r" \1", s)              # Add space before punctuation
    #s = re.sub(r"[^\w\s.!?]", r"", s)              # Remove non-word characters except punctuation
    return s.strip()


def readLangs(lang1 : str, lang2 : str, df : pd.DataFrame):
    print("Reading lines...")

    pairs = []
    length = len(df)
    for i in range(length):
        pair = []
        bng_text = df[lang1][i]
        processed_bng_text = normalizeString(bng_text)
        pair.append(processed_bng_text)
        eng_text = df[lang2][i]
        processed_eng_text = normalizeString(eng_text)
        pair.append(processed_eng_text)
        pairs.append(pair)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1 : str, lang2 : str, df : pd.DataFrame):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, df)
    print("Read %s sentence pairs" % len(pairs))

    # pairs = filterPairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print(f"Save input language dictionary")
    input_lang_dict = {}
    input_lang_dict["word2index"] = input_lang.word2index
    input_lang_dict["index2word"] = input_lang.index2word
    with open("backend/model/input_lang_obj.pkl", "wb") as f:
        pickle.dump(input_lang_dict, f)

    print(f"Save output language dictionary")
    output_lang_dict = {}
    output_lang_dict["word2index"] = output_lang.word2index
    output_lang_dict["index2word"] = output_lang.index2word
    with open("backend/model/output_lang_obj.pkl", "wb") as f:
        pickle.dump(output_lang_dict, f)


    print(f"Save language pairs")
    with open("backend/model/language_pairs.pkl", "wb") as f:
        pickle.dump(pairs, f)


    return input_lang, output_lang, pairs



if __name__ == "__main__":
    dataframe = pd.read_csv("backend/data/cleaned_data/cleaned_eng_bng.csv", encoding='utf-8')
    input_lang, output_lang, pairs = prepareData("bn", "en", dataframe)

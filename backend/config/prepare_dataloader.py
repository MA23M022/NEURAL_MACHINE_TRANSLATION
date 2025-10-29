import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from backend.constants.constant import SOS_token, EOS_token, MAX_LENGTH
from backend.utils.prepare_data import prepareData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(in_language : str, out_language : str, dataframe : pd.DataFrame, batch_size = 32):
    input_lang, output_lang, pairs = prepareData(in_language, out_language, dataframe)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    print(input_ids.shape)

    # For checking purpose what are the indices associated with sentences
    a = indexesFromSentence(input_lang, pairs[1][0])
    print(a)
    b = indexesFromSentence(output_lang, pairs[1][1])
    print(b)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids       # idx represents pair number.
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    print(input_ids[1])
    print(target_ids[1])

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))

    print(train_data[1])

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, pairs


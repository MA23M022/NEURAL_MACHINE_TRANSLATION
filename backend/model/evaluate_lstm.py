from __future__ import unicode_literals, print_function, division
import random
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from backend.constants.constant import SOS_token, EOS_token, MAX_LENGTH
from backend.config.prepare_dataloader import tensorFromSentence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def evaluate(model, sentence, input_lang, output_lang):
    model.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)       # creating tensor from sentence.
        decoder_outputs, decoder_hidden, decoder_attn = model(input_tensor)

        _, topi = decoder_outputs.topk(1)           # searching the index with high probabilities.
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(model, pairs, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(model, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
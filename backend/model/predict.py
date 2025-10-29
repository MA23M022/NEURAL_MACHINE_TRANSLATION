import torch
import pickle
from backend.constants.constant import SOS_token, EOS_token, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_lang = None
output_lang = None
pairs = None

with open("backend/model/input_lang_obj.pkl", "rb") as f:
    input_lang = pickle.load(f)
with open("backend/model/output_lang_obj.pkl", "rb") as f:
    output_lang = pickle.load(f)
with open("backend/model/language_pairs.pkl", "rb") as f:
    pairs = pickle.load(f)

print(list(input_lang.keys()))
print(list(output_lang.keys()))
print(f"The number of language pairs is {len(pairs)}")

model_v1 = torch.load(f"C:/Users/soura/Desktop/Resume_Projects/NEURAL_MT/mlartifacts/893904718680578900/models/m-3fe0de9f032d4d8bac133973866de9ee/artifacts/data/model.pth", weights_only=False)
m1 = list(model_v1.modules())[0]

MODEL_VERSION = "1.0.0"

print(m1.parameters)

def indexesFromSentence(lang, sentence):
    return [lang["word2index"][word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


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
            decoded_words.append(output_lang["index2word"][idx.item()])
    return decoded_words, decoder_attn



def predict_output(input_lang_id : int):
    ben_sentence = pairs[input_lang_id][0]
    actual_eng_sentence = pairs[input_lang_id][1]
    output_words, _ = evaluate(m1, ben_sentence, input_lang, output_lang)

    # Remove <EOS> and other special tokens from predicted output
    output_words = [word for word in output_words if word not in ['<EOS>', '<PAD>', '<SOS>']]
    pred_eng_sentence = ' '.join(output_words)

    return {"ben_sentence" : ben_sentence, "actual_eng_sentence" : actual_eng_sentence, "pred_eng_sentence" : pred_eng_sentence}




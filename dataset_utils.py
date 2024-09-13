import csv
import torch
import torchaudio 
import numpy as np
import pandas as pd

from tokenizer.soundstream.AudioTokenizer import AudioTokenizer

def create_csv(in_path, out_path):
    data = pd.read_csv(in_path, sep="\t")
    data = data[['item_name', 'accompaniment_path']]
    data.rename(columns = {'accompaniment_path':'path'}, inplace=True)
    data.to_csv(out_path, sep="\t", index=False)
    print(f"the csv for vocoder training is saved at {out_path}")

def create_acoustic_codes(csv_path, tok_path, out_path):
    file = open(csv_path, mode='r')
    data_reader = csv.DictReader(
            file,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
    data_items = [e for e in data_reader]

    audio_tokenizer = AudioTokenizer(ckpt_path=tok_path, device=torch.device('cuda'))    
    
    codes = {}

    for item in data_items: 
        wav, sr = torchaudio.load(item['path'])
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        if wav.shape[0]>1:
            mono_wav = wav.mean(dim=0, keepdim=True)
        else:
            mono_wav = wav
        mono_wav_enc = audio_tokenizer.encode(mono_wav)
        codes[item['item_name']] = mono_wav_enc

    # .npy file
    np.save(out_path, codes, allow_pickle=True)
    print(f"Acoustic Codes file saved at {out_path}")
    
def load_acoustic_codes(in_path):
    # load .npy file
    acoustic_file = np.load(in_path, allow_pickle=True).item()
    return acoustic_file
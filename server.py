from flask import Flask, render_template, request, make_response, jsonify
import os
import numpy as np
import torch
import  soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from werkzeug.utils import secure_filename
import librosa
import subprocess

from wav2vec2_kenlm.decoder import *
import wav2vec2_kenlm.utils
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-xlsr-min/processor")#Load Processor(Input)
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-min/model/checkpoint-37100")#Load Model(Input)
model.to(device)


lm_path = "./3-gram-tat.arpa"
vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
vocab = []
for _, token in sort_vocab:
    vocab.append(token)
vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '

alpha=0.9
beta=0.9
beam_width=1024
beam_decoder = BeamCTCDecoder(vocab[:-2], lm_path=lm_path,
                alpha=alpha, beta=beta,
                cutoff_top_n=40, cutoff_prob=1.0,
                beam_width=beam_width, num_processes=16,
                blank_index=vocab.index(processor.tokenizer.pad_token))

# bulid and load config webserver
UPLOAD_FOLDER = './static/tmpfile'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB


# Predicte
def pred(audio):
    # 採樣率問題所以我改用librosa, 這樣輸入就不用管採樣率
	speech, sr = librosa.load(audio, sr=16000, mono=True)
	
	input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to(device)

	with torch.no_grad():
	  logits = model(input_values).logits

	pred_ids = torch.argmax(logits, dim=-1)
	pred_str = processor.batch_decode(pred_ids)[0]

	return pred_str

def decode_kenlm(audio):
    speech, sr = librosa.load(audio, sr = 16000, mono = True)
    input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_str = processor.batch_decode(pred_ids)[0]
    beam, beam_decoded_offsets = beam_decoder.decode(logits)
    return pred_str, beam[0][0]

@app.route("/")
def index():
    return render_template("index.html", res_text="")

@app.route("/predict",  methods=["POST"])
def predict():
    file = request.files['file_loader']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    if filename[-4:] != ".wav":
        ffmpeg = " ".join(
            ["ffmpeg", "-i", os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4] + ".wav"), "-y"]
        )
        print(ffmpeg)
        subprocess.run(ffmpeg, shell=True)
        filename = filename[:-4] + ".wav"
    
    text, text2 = decode_kenlm(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("index.html", rt=1,res_text=text, res_text2=text2, audio_path=os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route("/test")
def test():
    return render_template("test.html", res_text="")

def main():
    app.run(debug=True, host="::", port=5010)
    
if __name__ == "__main__":
    main()

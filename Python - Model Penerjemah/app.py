from flask import Flask, request, jsonify
import torch
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path
from train import get_ds_csv, get_model, translate_input_string
import os

app = Flask(__name__)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
config_id = get_config()

config['model_basename'] = "tmodel_new_ma_"
config_id['model_basename'] = "tmodel_new_id_"
# Training bahasa target menjadi bahasa Maanyan
config_id['lang_src'] = "Indonesia"
config_id['lang_tgt'] = "Maanyan"

train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds_csv(config)
train_dataloader_id, val_dataloader_id, tokenizer_src_id, tokenizer_tgt_id = get_ds_csv(config_id)

model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
model_id = get_model(config_id, tokenizer_src_id.get_vocab_size(), tokenizer_tgt_id.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = get_weights_file_path(config, f"29")
model_filename_id = get_weights_file_path(config_id, f"29")

state = torch.load(model_filename)
state_id = torch.load(model_filename_id)

model.load_state_dict(state['model_state_dict'])
model_id.load_state_dict(state_id['model_state_dict'])


@app.route('/terjemahkan', methods=['POST'])
def terjemahkan():
    try:
        # Mendapatkan data JSON dari permintaan POST
        data = request.get_json()

        # Memeriksa apakah "teks" ada dalam data yang diterima
        if 'teks' not in data:
            return jsonify({"error": "Data 'teks' tidak ditemukan dalam permintaan."}), 400

        # Mengambil teks dari data
        input_string = data['teks']

        # Menerjemahkan teks menggunakan model
        # translation = translate_input_string(model, input_string, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        translation = translate_input_string(model, input_string, tokenizer_src, tokenizer_tgt, len(input_string.split(" "))+1, device)

        # Mengembalikan hasil terjemahan dalam format JSON
        return jsonify({"terjemahan": translation}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/terjemahkanid', methods=['POST'])
def terjemahkan_id():
    try:
        # Mendapatkan data JSON dari permintaan POST
        data = request.get_json()

        # Memeriksa apakah "teks" ada dalam data yang diterima
        if 'teks' not in data:
            return jsonify({"error": "Data 'teks' tidak ditemukan dalam permintaan."}), 400

        # Mengambil teks dari data
        input_string = data['teks']

        # Menerjemahkan teks menggunakan model
        # translation = translate_input_string(model_id, input_string, tokenizer_src_id, tokenizer_tgt_id, config_id['seq_len'], device)
        translation = translate_input_string(model_id, input_string, tokenizer_src_id, tokenizer_tgt_id, len(input_string.split(" "))+1, device)

        # Mengembalikan hasil terjemahan dalam format JSON
        return jsonify({"terjemahan": translation}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import pandas as pd

from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            # kode
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
            # kode 
            break

    return decoder_input.squeeze(0)

def greedy_decode_test(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    print(encoder_output)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # print(f'===============================')
    # print(decoder_input)
    # print(f'===============================')
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        print(f'decoder input size : {decoder_input.size(1)}')
        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        print(f'decoder mask : {decoder_mask}')
        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        print(f'out : {out}')

        # get next token
        prob = model.project(out[:, -1])
        print(f'prob : {prob}')

        # print(prob)
        _, next_word = torch.max(prob, dim=1)
        print(f'_ : {_}')
        print('next word: ')
        print(next_word)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        print(f'eos idx : {eos_idx}')
        if next_word == eos_idx:
            # kode
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
            # kode 
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        # metric = torchmetrics.CharErrorRate()
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        # metric = torchmetrics.WordErrorRate()
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        # metric = torchmetrics.BLEUScore()
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def eos_add(string):
    if (string.find(".") != -1) :
        if(string.find("[EOS]") != -1) :
            string = string.replace("[EOS]","")
        string = string.replace("."," [EOS]", 1)
        if(string.find(".") != -1) :
            string = string.replace(".","")
        return string

    string = string+' [EOS]'

    return string


def translate_input_string(model, input_string, tokenizer_src, tokenizer_tgt, max_len, device):
    
    # input_string = eos_add(input_string)
    model.eval()

    # Tokenize the input string
    src_tokens = tokenizer_src.encode(input_string).ids

    # Prepare the input tensors
    encoder_input = torch.tensor(src_tokens, dtype=torch.int64).unsqueeze(0).to(device)  # Add batch dimension
    encoder_mask = torch.ones((1, 1, 1, len(src_tokens)), dtype=torch.int64).to(device)

    # Perform greedy decoding to generate the target sentence
    translation = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    # Convert the translation from token IDs to text
    translation_text = tokenizer_tgt.decode(translation.cpu().numpy())

    return translation_text

def translate_input_string_test(model, input_string, tokenizer_src, tokenizer_tgt, max_len, device):
    
    # input_string = eos_add(input_string)
    model.eval()

    # Tokenize the input string
    src_tokens = tokenizer_src.encode(input_string).ids
    print(src_tokens)

    # Prepare the input tensors
    encoder_input = torch.tensor(src_tokens, dtype=torch.int64).unsqueeze(0).to(device)  # Add batch dimension
    encoder_mask = torch.ones((1, 1, 1, len(src_tokens)), dtype=torch.int64).to(device)

    print("encoder_mask")
    print(encoder_mask)

    # Perform greedy decoding to generate the target sentence
    translation = greedy_decode_test(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    # Convert the translation from token IDs to text
    translation_text = tokenizer_tgt.decode(translation.cpu().numpy())


#translate without trailing
def translate_input_string_alt(model, input_string, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()

    input_string ="[SOS] "+ input_string +" [EOS]"

    # Tokenize the input string
    src_tokens = tokenizer_src.encode(input_string).ids

    # Prepare the input tensors
    encoder_input = torch.tensor(src_tokens, dtype=torch.int64).unsqueeze(0).to(device)  # Add batch dimension
    encoder_mask = torch.ones((1, 1, 1, len(src_tokens)), dtype=torch.int64).to(device)

    # Perform greedy decoding to generate the target sentence
    translation = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    # Convert the translation from token IDs to text
    translation_text = tokenizer_tgt.decode(translation.cpu().numpy())

    return translation_text

def translate_input_string_bleu(model, input_string, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()

    # Tokenize the input string
    src_tokens = tokenizer_src.encode(input_string).ids

    # Prepare the input tensors
    encoder_input = torch.tensor(src_tokens, dtype=torch.int64).unsqueeze(0).to(device)  # Add batch dimension
    encoder_mask = torch.ones((1, 1, 1, len(src_tokens)), dtype=torch.int64).to(device)

    # Perform greedy decoding to generate the target sentence
    translation = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    # Convert the translation from token IDs to text
    translation_text = tokenizer_tgt.decode(translation.cpu().numpy())

    return translation_text

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# START NEW FOR CSV

def get_all_sentences_csv(df, lang):
    return df[lang]

def get_or_build_tokenizer_csv(config, df, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # min_frequency ditentukan agar suatu kata akan dimasukkan dalam tokenizer jika minimal muncul di dua kalimat 
        # Default min_frequency = 2, jika menggunakan model tmodel_eight_
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=4)
        tokenizer.train_from_iterator(get_all_sentences_csv(df, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds_csv(config):
    # Load data from CSV file
    csv_file = 'D:\Oscar Main Base\File Online\Aplikasi_Penerjemah\Develop\pytorch-transformer-main\datasets\datasetma4.csv'
    # Dataset ini untuk testing
    # csv_file = 'D:\Oscar Main Base\File Online\Aplikasi_Penerjemah\Python (2)\pytorch-transformer-main\datasets\datasetma4_semicolon.csv'
    # df = pd.read_csv(csv_file)
    df = pd.read_csv(csv_file, sep=';')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer_csv(config, df, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer_csv(config, df, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(df))
    val_ds_size = len(df) - train_ds_size
    # OSCAR : Perubahan train ds 
    # train_df, val_df = df[:train_ds_size], df[train_ds_size:]
    train_df, val_df = df[:train_ds_size], df[val_ds_size:]

    train_ds = BilingualDataset(train_df, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_df, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    # ORIGINAL CODE
    # for _, item in df.iterrows():
    #     src_ids = tokenizer_src.encode(item['source_text']).ids
    #     tgt_ids = tokenizer_tgt.encode(item['target_text']).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    for item in df[config['lang_src']]:
        src_ids = tokenizer_src.encode(item).ids
        max_len_src = max(max_len_src, len(src_ids))
    
    for item in df[config['lang_tgt']]:
        tgt_ids = tokenizer_tgt.encode(item).ids
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# END OF NEW FOR CSV

def get_model(config, vocab_src_len, vocab_tgt_len):
    # Oscar : Mengisi parameter N untuk mengubah jumlah layer dalam model
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'], N=8)
    return model

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds_csv(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']


    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['preload'] = "59"
    train_model(config)

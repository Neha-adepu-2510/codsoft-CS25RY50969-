import os
import zipfile
import json
import numpy as np
import string
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import to_categorical

# ---- STEP 1: Download Dataset using Kaggle API ----
def download_flickr8k():
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    # Create kaggle folder and move kaggle.json
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        os.rename("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    # Download dataset
    os.system("kaggle datasets download -d adityajn105/flickr8k -p dataset/ --unzip")

download_flickr8k()

# Paths
images_path = "dataset/Flicker8k_Dataset/"
captions_file = "dataset/Flickr8k.token.txt"

# ---- STEP 2: Load and clean captions ----
def load_captions(filepath):
    captions = {}
    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            img_id, caption = tokens[0].split('#')[0], tokens[1]
            caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
            captions.setdefault(img_id, []).append('startseq ' + caption + ' endseq')
    return captions

# ---- STEP 3: Extract features using ResNet50 ----
def extract_features(directory):
    model = ResNet50(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[name] = feature
    return features

# ---- STEP 4: Tokenize captions ----
def create_tokenizer(captions_dict):
    all_captions = []
    for caps in captions_dict.values():
        all_captions.extend(caps)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# ---- STEP 5: Create sequences ----
def create_sequences(tokenizer, max_length, captions_list, image_features, vocab_size):
    X1, X2, y = [], [], []
    for key, captions in captions_list.items():
        for cap in captions:
            seq = tokenizer.texts_to_sequences([cap])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(image_features[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# ---- STEP 6: Build model ----
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ---- STEP 7: Generate caption ----
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# ---- MAIN EXECUTION ----
print("📖 Loading and preparing data...")
captions = load_captions(captions_file)
features = extract_features(images_path)

print("🧠 Tokenizing...")
tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for caps in captions.values() for c in caps)

print("📊 Creating sequences...")
X1, X2, y = create_sequences(tokenizer, max_length, captions, features, vocab_size)

print("🧠 Training model...")
model = define_model(vocab_size, max_length)
model.fit([X1, X2], y, epochs=10, batch_size=64, verbose=1)

model.save("image_caption_model.h5")
print("✅ Model saved as image_caption_model.h5")

# Test caption on one image
print("🖼️ Testing caption on 1st image...")
sample_img = list(features.values())[0].reshape((1, 2048))
caption = generate_caption(model, tokenizer, sample_img, max_length)
print("📝 Generated Caption:", caption)

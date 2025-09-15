import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      
os.environ["CUDA_VISIBLE_DEVICES"] = ""       
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  

import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


print("Loading IMDB dataset...")
imdb = load_dataset("imdb")

X_train = [x["text"] for x in imdb["train"]]
y_train = np.array([x["label"] for x in imdb["train"]])
X_test = [x["text"] for x in imdb["test"]]
y_test = np.array([x["label"] for x in imdb["test"]])


max_words = 10000   
max_len = 200      

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")


model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])

print("\nTraining CNN model...")
history = model.fit(
    X_train_pad, y_train,
    epochs=3, batch_size=128,
    validation_split=0.2,
    verbose=1
)


y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nCNN Results:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")

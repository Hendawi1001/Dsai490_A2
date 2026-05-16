import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Reshape, RepeatVector, Layer

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class KLLossLayer(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return inputs

class ConditionalVAE:
    def __init__(self, seq_len, vocab_size, cond_vocab_sizes, latent_dim=64):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cond_vocab_sizes = cond_vocab_sizes
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.cvae = self.build_cvae()

    def build_condition_embeddings(self, inputs):
        embeds = []
        for i, size in enumerate(self.cond_vocab_sizes):
            emb = Embedding(input_dim=size, output_dim=16)(inputs[i])
            emb = Reshape((16,))(emb)
            embeds.append(emb)
        return Concatenate()(embeds)

    def build_encoder(self):
        seq_in = Input(shape=(self.seq_len, self.vocab_size), name="encoder_seq_in")
        cond_inputs = [Input(shape=(1,), name=f"enc_cond_{i}") for i in range(4)]
        
        x_seq = LSTM(128)(seq_in)
        
        cond_features = self.build_condition_embeddings(cond_inputs)
        x = Concatenate()([x_seq, cond_features])
        x = Dense(64, activation="relu")(x)
        
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        return tf.keras.Model([seq_in] + cond_inputs, [z_mean, z_log_var, z], name="Encoder")

    def build_decoder(self):
        z_in = Input(shape=(self.latent_dim,), name="decoder_z_in")
        cond_inputs = [Input(shape=(1,), name=f"dec_cond_{i}") for i in range(4)]
        
        cond_features = self.build_condition_embeddings(cond_inputs)
        x = Concatenate()([z_in, cond_features])
        
        x = RepeatVector(self.seq_len)(x)
        x = LSTM(128, return_sequences=True)(x)
        
        out = Dense(self.vocab_size, activation="softmax")(x)
        
        return tf.keras.Model([z_in] + cond_inputs, out, name="Decoder")

    def build_cvae(self):
        seq_in = Input(shape=(self.seq_len, self.vocab_size))
        cond_inputs = [Input(shape=(1,)) for _ in range(4)]
        z_mean, z_log_var, z = self.encoder([seq_in] + cond_inputs)
        reconstructed = self.decoder([z] + cond_inputs)
        
        KLLossLayer()([z_mean, z_log_var])
        
        model = tf.keras.Model([seq_in] + cond_inputs, reconstructed, name="cVAE")
        
        return model

import sys
import os
import numpy as np
import datetime
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Token import DateTokenizer

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def evaluate_predictions(predictions, conditions_list):
    valid_count = 0
    total = len(predictions)
    days_map = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
    months_map = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 
                  7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
    for date_str, cond in zip(predictions, conditions_list):
        req_day, req_month, req_leap, req_decade = cond
        try:
            d, m, y = map(int, date_str.split('-'))
            date_obj = datetime.date(y, m, d)
            act_day = days_map[date_obj.weekday()]
            act_month = months_map[date_obj.month]
            act_leap = str(is_leap_year(y))
            act_decade = str(y)[:3]
            if (act_day == req_day and act_month == req_month and 
                act_leap == req_leap and act_decade == req_decade):
                valid_count += 1
        except (ValueError, TypeError):
            continue
    return (valid_count / total) * 100 if total > 0 else 0


def cvae_inference(decoder, conditions, tokenizer, latent_dim=64):
    z = tf.random.normal([1, latent_dim])
    cond_tensors = [tf.constant([[c]], dtype=tf.float32) for c in conditions]
    predictions = decoder([z] + cond_tensors, training=False)
    
    token_ids = np.argmax(predictions[0].numpy(), axis=-1)
    return tokenizer.decode_date(token_ids)

def train_cvae(X_train, y_train, vocab_size, epochs=30, batch_size=64, latent_dim=64, epoch_callback=None):
    cond_vocab_sizes = [7, 12, 2, 41]
    seq_len = y_train.shape[1]
    
    cvae_obj = ConditionalVAE(seq_len, vocab_size, cond_vocab_sizes, latent_dim)
    model = cvae_obj.cvae
    decoder = cvae_obj.decoder
    
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    y_train_one_hot = tf.one_hot(y_train, depth=vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_one_hot)).shuffle(1024).batch(batch_size)
    
    @tf.function
    def train_step(batch_x, batch_y):
        cond_inputs = [batch_x[:, i:i+1] for i in range(4)]
        with tf.GradientTape() as tape:
            reconstructed = model([batch_y] + cond_inputs, training=True)
            recon_loss = loss_fn(batch_y, reconstructed)
            total_loss = recon_loss + sum(model.losses)
            
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    print("Starting cVAE training...")
    losses = []
    for epoch in range(epochs):
        epoch_loss, batches = 0.0, 0
        for batch_x, batch_y in dataset:
            loss = train_step(batch_x, batch_y)
            epoch_loss += loss.numpy()
            batches += 1
        losses.append(epoch_loss / batches)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {losses[-1]:.4f}")
        
        if epoch_callback is not None:
            epoch_callback(decoder, epoch)
            
    return model, decoder, losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Run inference')
    default_in = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../data/example_input.txt')
    default_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../data/output.txt')
    parser.add_argument('-i', '--input', type=str, default=default_in)
    parser.add_argument('-o', '--output', type=str, default=default_out)
    args = parser.parse_args()

    tokenizer = DateTokenizer()
    vocab_size = len(tokenizer.date_chars)

    if args.train:
        print("Loading data...")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../data/data.txt')
        X_train, y_train = tokenizer.process_dataset(data_path)
        sample_conditions = tokenizer.encode_conditions(['WED', 'JAN', 'False', '200'])
        
        def monitor_callback(decoder, epoch):
            print("  -> Generating sample date for [WED] [JAN] [False] [200] ...")
            generated_date = cvae_inference(decoder, sample_conditions, tokenizer)
            print(f"  -> Generated: {generated_date}")

        model, decoder, losses = train_cvae(X_train, y_train, vocab_size, epoch_callback=monitor_callback)
        
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        os.makedirs(weights_dir, exist_ok=True)
        decoder.save_weights(os.path.join(weights_dir, 'cvae_decoder.weights.h5'))
        
        plt.plot(losses)
        plt.title('cVAE Training Loss')
        plt.savefig(os.path.join(weights_dir, 'cvae_loss.png'))
        print("Training complete. Weights saved.")

    elif args.predict:
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        dec_weights_path = os.path.join(weights_dir, 'cvae_decoder.weights.h5')
        
        cond_vocab_sizes = [7, 12, 2, 41]
        seq_len = 12
        latent_dim = 64
        cvae_obj = ConditionalVAE(seq_len, vocab_size, cond_vocab_sizes, latent_dim)
        decoder = cvae_obj.decoder
        
        dummy_conds = [np.zeros((1, 1)) for _ in range(4)]
        dummy_z = np.zeros((1, latent_dim))
        decoder([dummy_z] + dummy_conds)
        decoder.load_weights(dec_weights_path)
        
        print(f"Reading inputs from {args.input}...")
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            count = 0
            for line in infile:
                line = line.strip()
                if not line: continue
                raw_conditions, _ = tokenizer.parse_line(line)
                encoded_conditions = tokenizer.encode_conditions(raw_conditions)
                generated_date = cvae_inference(decoder, encoded_conditions, tokenizer)
                output_line = f"[{raw_conditions[0]}] [{raw_conditions[1]}] [{raw_conditions[2]}] [{raw_conditions[3]}] {generated_date}\n"
                outfile.write(output_line)
                count += 1
                if count % 100 == 0: print(f"  ... Processed {count} lines ...")
        print(f"Predictions successfully saved to {args.output}")
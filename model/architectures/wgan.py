import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Reshape, Dropout, GaussianNoise, LeakyReLU
import sys
import os
import numpy as np
import datetime
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Token import DateTokenizer

class WGAN_GP:
    def __init__(self, seq_len, vocab_size, cond_vocab_sizes, latent_dim=100):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cond_vocab_sizes = cond_vocab_sizes 
        
        self.generator = self.build_generator()
        self.critic = self.build_critic()

    def build_condition_embeddings(self, inputs):
        embeds = []
        for i, size in enumerate(self.cond_vocab_sizes):
            emb = Embedding(input_dim=size, output_dim=16)(inputs[i])
            emb = Reshape((16,))(emb)
            embeds.append(emb)
        return Concatenate()(embeds)

    def build_generator(self):
        noise_in = Input(shape=(self.latent_dim,))
        cond_inputs = [Input(shape=(1,)) for _ in range(4)]
        cond_features = self.build_condition_embeddings(cond_inputs)
        x = Concatenate()([noise_in, cond_features])
        x = Dense(self.seq_len * 64, activation='relu')(x)
        x = Reshape((self.seq_len, 64))(x)
        x = LSTM(128, return_sequences=True)(x)
        out = Dense(self.vocab_size, activation='softmax')(x)
        return tf.keras.Model([noise_in] + cond_inputs, out, name="Generator")

    def build_critic(self):
        seq_in = Input(shape=(self.seq_len, self.vocab_size)) 
        noisy_seq_in = GaussianNoise(0.1)(seq_in)
        cond_inputs = [Input(shape=(1,)) for _ in range(4)]
        cond_features = self.build_condition_embeddings(cond_inputs)
        x_seq = LSTM(128)(noisy_seq_in)
        x = Concatenate()([x_seq, cond_features])
        x = Dense(64)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        out = Dense(1)(x)
        return tf.keras.Model([seq_in] + cond_inputs, out, name="Critic")

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

def plot_training_graphs(d_losses, g_losses, save_path="training_loss.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Critic Loss', color='blue')
    plt.plot(g_losses, label='Generator Loss', color='orange')
    plt.title('WGAN-GP Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def wgan_inference(generator, conditions, tokenizer, latent_dim=100):
    noise = tf.random.normal([1, latent_dim])
    cond_tensors = [tf.constant([[c]], dtype=tf.float32) for c in conditions]
    predictions = generator([noise] + cond_tensors, training=False)
    token_ids = np.argmax(predictions[0].numpy(), axis=-1)
    return tokenizer.decode_date(token_ids)

def gradient_penalty(critic, batch_y, generated_seqs, cond_inputs):
    alpha = tf.random.uniform([tf.shape(batch_y)[0], 1, 1], 0.0, 1.0)
    interpolated = alpha * batch_y + (1 - alpha) * generated_seqs
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated] + cond_inputs, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def train_wgan(X_train, y_train, vocab_size, epochs=20, batch_size=64, latent_dim=100, epoch_callback=None):
    cond_vocab_sizes = [7, 12, 2, 41] 
    seq_len = y_train.shape[1]
    
    wgan = WGAN_GP(seq_len, vocab_size, cond_vocab_sizes, latent_dim)
    gen, critic = wgan.generator, wgan.critic
    
    gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    critic_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    y_train_one_hot = tf.one_hot(y_train, depth=vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_one_hot)).shuffle(1024).batch(batch_size)
    
    critic_steps = 3
    gp_weight = 10.0

    @tf.function
    def train_critic_step(batch_x, batch_y):
        cond_inputs = [batch_x[:, i:i+1] for i in range(4)]
        current_batch_size = tf.shape(batch_x)[0]
        noise = tf.random.normal([current_batch_size, latent_dim])
        with tf.GradientTape() as critic_tape:
            generated_seqs = gen([noise] + cond_inputs, training=True)
            real_output = critic([batch_y] + cond_inputs, training=True)
            fake_output = critic([generated_seqs] + cond_inputs, training=True)
            c_cost = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = gradient_penalty(critic, batch_y, generated_seqs, cond_inputs)
            c_loss = c_cost + gp_weight * gp
        gradients = critic_tape.gradient(c_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        return c_loss

    @tf.function
    def train_gen_step(batch_x):
        cond_inputs = [batch_x[:, i:i+1] for i in range(4)]
        current_batch_size = tf.shape(batch_x)[0]
        noise = tf.random.normal([current_batch_size, latent_dim])
        with tf.GradientTape() as gen_tape:
            generated_seqs = gen([noise] + cond_inputs, training=True)
            fake_output = critic([generated_seqs] + cond_inputs, training=True)
            g_loss = -tf.reduce_mean(fake_output)
        gradients = gen_tape.gradient(g_loss, gen.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients, gen.trainable_variables))
        return g_loss

    print("Starting WGAN-GP training...")
    g_losses, c_losses = [], []
    for epoch in range(epochs):
        epoch_c_loss, epoch_g_loss, batches = 0.0, 0.0, 0
        for batch_x, batch_y in dataset:
            for _ in range(critic_steps):
                c_loss = train_critic_step(batch_x, batch_y)
            g_loss = train_gen_step(batch_x)
            epoch_c_loss += c_loss.numpy()
            epoch_g_loss += g_loss.numpy()
            batches += 1
            
        c_losses.append(epoch_c_loss / batches)
        g_losses.append(epoch_g_loss / batches)
        print(f"Epoch {epoch+1}/{epochs} | Critic Loss: {c_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f}")
        
        if epoch_callback is not None:
            epoch_callback(gen, epoch)
            
    return gen, c_losses, g_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
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
        
        def monitor_callback(gen, epoch):
            print("  -> Generating sample date for [WED] [JAN] [False] [200] ...")
            generated_date = wgan_inference(gen, sample_conditions, tokenizer)
            print(f"  -> Generated: {generated_date}")

        gen, c_losses, g_losses = train_wgan(X_train, y_train, vocab_size, epoch_callback=monitor_callback)
        
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        os.makedirs(weights_dir, exist_ok=True)
        gen.save_weights(os.path.join(weights_dir, 'wgan_generator.weights.h5'))
        
        plot_training_graphs(c_losses, g_losses, os.path.join(weights_dir, 'wgan_loss.png'))
        print("Training complete. Weights saved.")

    elif args.predict:
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        gen_weights_path = os.path.join(weights_dir, 'wgan_generator.weights.h5')
        
        cond_vocab_sizes = [7, 12, 2, 41]
        seq_len = 12
        latent_dim = 100
        wgan_obj = WGAN_GP(seq_len, vocab_size, cond_vocab_sizes, latent_dim)
        gen = wgan_obj.generator
        
        dummy_conds = [tf.zeros((1, 1)) for _ in range(4)]
        dummy_noise = tf.zeros((1, latent_dim))
        gen([dummy_noise] + dummy_conds)
        gen.load_weights(gen_weights_path)
        
        print(f"Reading inputs from {args.input}...")
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            count = 0
            for line in infile:
                line = line.strip()
                if not line: continue
                raw_conditions, _ = tokenizer.parse_line(line)
                encoded_conditions = tokenizer.encode_conditions(raw_conditions)
                generated_date = wgan_inference(gen, encoded_conditions, tokenizer)
                output_line = f"[{raw_conditions[0]}] [{raw_conditions[1]}] [{raw_conditions[2]}] [{raw_conditions[3]}] {generated_date}\n"
                outfile.write(output_line)
                count += 1
                if count % 100 == 0: print(f"  ... Processed {count} lines ...")
        print(f"Predictions successfully saved to {args.output}")

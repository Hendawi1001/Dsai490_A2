import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, LayerNormalization, Dropout, MultiHeadAttention

def causal_attention_mask(n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    return mask

class TransformerDecoder:
    def __init__(self, seq_len, vocab_size, cond_vocab_sizes, embed_dim=64, num_heads=4):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.cond_vocab_sizes = cond_vocab_sizes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.model = self.build_transformer()

    def build_transformer(self):
        cond_inputs = [Input(shape=(1,), name=f"cond_{i}") for i in range(4)]
        seq_in = Input(shape=(self.seq_len,), name="seq_in")
        
        embeds = []
        for i, size in enumerate(self.cond_vocab_sizes):
            emb = Embedding(input_dim=size, output_dim=self.embed_dim)(cond_inputs[i])
            embeds.append(emb)
        
        cond_prefix = Concatenate(axis=1)(embeds)
        
        seq_emb = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)(seq_in)
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_emb = Embedding(input_dim=self.seq_len, output_dim=self.embed_dim)(positions)
        seq_x = seq_emb + pos_emb
        
        x = Concatenate(axis=1)([cond_prefix, seq_x])
        total_len = 4 + self.seq_len
        
        mask = causal_attention_mask(total_len, total_len, tf.bool)
        
        for _ in range(3):
            attention_out = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)(
                x, x, attention_mask=mask
            )
            x = LayerNormalization(epsilon=1e-6)(x + attention_out)
            
            ffn_out = Dense(self.embed_dim * 4, activation="relu")(x)
            ffn_out = Dense(self.embed_dim)(ffn_out)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_out)
        
        seq_out = x[:, 4:, :] 
        out = Dense(self.vocab_size, activation="softmax")(seq_out)
        
        return tf.keras.Model(cond_inputs + [seq_in], out, name="TransformerDecoder")

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

@tf.function(reduce_retracing=True)
def predict_step(model, conds, seq):
    return model(conds + [seq], training=False)

def autoregressive_inference(model, conditions, tokenizer, max_len=11):
    current_seq = [tokenizer.char2idx['<SOS>']]
    cond_tensors = [tf.constant([[c]], dtype=tf.float32) for c in conditions]
    
    for _ in range(max_len):
        padded_seq = current_seq + [tokenizer.char2idx['<PAD>']] * (max_len - len(current_seq))
        seq_tensor = tf.constant([padded_seq], dtype=tf.float32)
        predictions = predict_step(model, cond_tensors, seq_tensor)
        predictions = predictions.numpy()
        next_token_idx = np.argmax(predictions[0, len(current_seq) - 1, :])
        current_seq.append(next_token_idx)
        if tokenizer.idx2char[next_token_idx] == '<EOS>':
            break
            
    return tokenizer.decode_date(current_seq)

def train_transformer(X_train, y_train, vocab_size, epochs=20,batch_size=128, embed_dim=128, epoch_callback=None):
    cond_vocab_sizes = [7, 12, 2, 41]
    seq_len_input = y_train.shape[1] - 1 
    
    transformer_obj = TransformerDecoder(seq_len_input, vocab_size, cond_vocab_sizes, embed_dim)
    model = transformer_obj.model
    
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size)
    
    @tf.function
    def train_step(batch_x, batch_y):
        cond_inputs = [batch_x[:, i:i+1] for i in range(4)]
        seq_in = batch_y[:, :-1]
        seq_target = batch_y[:, 1:]
        
        with tf.GradientTape() as tape:
            predictions = model(cond_inputs + [seq_in], training=True)
            loss = loss_fn(seq_target, predictions)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print("Starting Transformer training...")
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
            epoch_callback(model, epoch)
            
    return model, losses

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
        
        def monitor_callback(model, epoch):
            print("  -> Generating sample date for [WED] [JAN] [False] [200] ...")
            generated_date = autoregressive_inference(model, sample_conditions, tokenizer)
            print(f"  -> Generated: {generated_date}")

        model, losses = train_transformer(X_train, y_train, vocab_size, epoch_callback=monitor_callback)
        
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(os.path.join(weights_dir, 'transformer_model.weights.h5'))
        
        plt.plot(losses)
        plt.title('Transformer Training Loss')
        plt.savefig(os.path.join(weights_dir, 'transformer_loss.png'))
        print("Training complete. Weights saved.")

    elif args.predict:
        weights_path = os.path.join(os.path.dirname(__file__), '../weights/transformer_model.weights.h5')
        cond_vocab_sizes = [7, 12, 2, 41]
        seq_len_input = 11
        transformer_obj = TransformerDecoder(seq_len_input, vocab_size, cond_vocab_sizes, embed_dim=128)
        model = transformer_obj.model
        
        dummy_conds = [np.zeros((1, 1)) for _ in range(4)]
        dummy_seq = np.zeros((1, seq_len_input))
        model(dummy_conds + [dummy_seq])
        model.load_weights(weights_path)
        
        print(f"Reading inputs from {args.input}...")
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            count = 0
            for line in infile:
                line = line.strip()
                if not line: continue
                raw_conditions, _ = tokenizer.parse_line(line)
                encoded_conditions = tokenizer.encode_conditions(raw_conditions)
                generated_date = autoregressive_inference(model, encoded_conditions, tokenizer)
                output_line = f"[{raw_conditions[0]}] [{raw_conditions[1]}] [{raw_conditions[2]}] [{raw_conditions[3]}] {generated_date}\n"
                outfile.write(output_line)
                count += 1
                if count % 100 == 0: print(f"  ... Processed {count} lines ...")
        print(f"Predictions successfully saved to {args.output}")
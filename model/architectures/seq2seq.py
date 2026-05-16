import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Layer, Reshape

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class Seq2SeqAttention:
    def __init__(self, seq_len, vocab_size, cond_vocab_sizes, hidden_units=128):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.cond_vocab_sizes = cond_vocab_sizes
        self.hidden_units = hidden_units
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        cond_inputs = [Input(shape=(1,), name=f"enc_cond_{i}") for i in range(4)]
        embeds = []
        for i, size in enumerate(self.cond_vocab_sizes):
            emb = Embedding(input_dim=size, output_dim=32)(cond_inputs[i])
            embeds.append(emb)
        x = Concatenate(axis=1)(embeds) 
        encoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(x)
        
        return tf.keras.Model(cond_inputs, [encoder_outputs, state_h, state_c], name="Encoder")

    def build_decoder(self):
        target_token_in = Input(shape=(1,), name="dec_target_in")
        enc_outputs_in = Input(shape=(4, self.hidden_units), name="enc_outputs")
        state_h_in = Input(shape=(self.hidden_units,), name="state_h")
        state_c_in = Input(shape=(self.hidden_units,), name="state_c")
        x = Embedding(input_dim=self.vocab_size, output_dim=32)(target_token_in)
        attention_layer = BahdanauAttention(self.hidden_units)
        context_vector, attention_weights = attention_layer(state_h_in, enc_outputs_in)
        context_vector_expanded = Reshape((1, self.hidden_units))(context_vector)
        x = Concatenate(axis=-1)([context_vector_expanded, x])
        decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True)
        x, state_h, state_c = decoder_lstm(x, initial_state=[state_h_in, state_c_in])
        x = Reshape((-1,))(x)
        out = Dense(self.vocab_size, activation="softmax")(x)
        
        return tf.keras.Model(
            [target_token_in, enc_outputs_in, state_h_in, state_c_in], 
            [out, state_h, state_c, attention_weights], 
            name="Decoder"
        )

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

def seq2seq_inference(encoder, decoder, conditions, tokenizer, max_len=11):
    cond_tensors = [tf.constant([[c]], dtype=tf.float32) for c in conditions]
    
    enc_output, state_h, state_c = encoder(cond_tensors, training=False)
    
    dec_input = tf.constant([[tokenizer.char2idx['<SOS>']]], dtype=tf.float32)
    current_seq = [tokenizer.char2idx['<SOS>']]
    
    for _ in range(max_len):
        predictions, state_h, state_c, _ = decoder(
            [dec_input, enc_output, state_h, state_c], training=False
        )
        
        predicted_id = tf.argmax(predictions[0]).numpy()
        current_seq.append(predicted_id)
        
        if tokenizer.idx2char[predicted_id] == '<EOS>':
            break
            
        dec_input = tf.constant([[predicted_id]], dtype=tf.float32)
        
    return tokenizer.decode_date(current_seq)

def train_seq2seq(X_train, y_train, vocab_size, epochs=30, batch_size=64, hidden_units=128, epoch_callback=None):
    cond_vocab_sizes = [7, 12, 2, 41]
    seq_len = y_train.shape[1]
    
    seq2seq = Seq2SeqAttention(seq_len, vocab_size, cond_vocab_sizes, hidden_units)
    encoder, decoder = seq2seq.encoder, seq2seq.decoder
    
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size)
    
    @tf.function
    def train_step(batch_x, batch_y):
        cond_inputs = [batch_x[:, i:i+1] for i in range(4)]
        loss = 0
        
        with tf.GradientTape() as tape:
            enc_output, state_h, state_c = encoder(cond_inputs, training=True)
            dec_input = tf.expand_dims(batch_y[:, 0], 1)
            
            for t in range(1, seq_len):
                predictions, state_h, state_c, _ = decoder(
                    [dec_input, enc_output, state_h, state_c], training=True
                )
                loss += loss_fn(batch_y[:, t], predictions)
                dec_input = tf.expand_dims(batch_y[:, t], 1)
                
        batch_loss = (loss / int(batch_y.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    print("Starting Seq2Seq training...")
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
            epoch_callback(encoder, decoder, epoch)
            
    return encoder, decoder, losses

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
        
        def monitor_callback(enc, dec, epoch):
            print("  -> Generating sample date for [WED] [JAN] [False] [200] ...")
            generated_date = seq2seq_inference(enc, dec, sample_conditions, tokenizer)
            print(f"  -> Generated: {generated_date}")

        encoder, decoder, losses = train_seq2seq(X_train, y_train, vocab_size, epoch_callback=monitor_callback)
        
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        os.makedirs(weights_dir, exist_ok=True)
        encoder.save_weights(os.path.join(weights_dir, 'seq2seq_encoder.weights.h5'))
        decoder.save_weights(os.path.join(weights_dir, 'seq2seq_decoder.weights.h5'))
        
        plt.plot(losses)
        plt.title('Seq2Seq Training Loss')
        plt.savefig(os.path.join(weights_dir, 'seq2seq_loss.png'))
        print("Training complete. Weights saved.")

    elif args.predict:
        weights_dir = os.path.join(os.path.dirname(__file__), '../weights')
        enc_weights_path = os.path.join(weights_dir, 'seq2seq_encoder.weights.h5')
        dec_weights_path = os.path.join(weights_dir, 'seq2seq_decoder.weights.h5')
        
        cond_vocab_sizes = [7, 12, 2, 41]
        seq_len = 12
        seq2seq = Seq2SeqAttention(seq_len, vocab_size, cond_vocab_sizes, hidden_units=128)
        encoder, decoder = seq2seq.encoder, seq2seq.decoder
        
        dummy_conds = [tf.zeros((1, 1)) for _ in range(4)]
        enc_out, state_h, state_c = encoder(dummy_conds)
        dummy_dec_in = tf.zeros((1, 1))
        decoder([dummy_dec_in, enc_out, state_h, state_c])
        
        encoder.load_weights(enc_weights_path)
        decoder.load_weights(dec_weights_path)
        
        print(f"Reading inputs from {args.input}...")
        with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
            count = 0
            for line in infile:
                line = line.strip()
                if not line: continue
                raw_conditions, _ = tokenizer.parse_line(line)
                encoded_conditions = tokenizer.encode_conditions(raw_conditions)
                generated_date = seq2seq_inference(encoder, decoder, encoded_conditions, tokenizer)
                output_line = f"[{raw_conditions[0]}] [{raw_conditions[1]}] [{raw_conditions[2]}] [{raw_conditions[3]}] {generated_date}\n"
                outfile.write(output_line)
                count += 1
                if count % 100 == 0: print(f"  ... Processed {count} lines ...")
        print(f"Predictions successfully saved to {args.output}")
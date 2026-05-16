<style>
  body {
    font-family: "Times New Roman", Times, serif;
  }
</style>

# DSAI490 Assignment 2: Conditional Date Generation Report

**Name:** Abdalrahman Khaled  
**ID:** 202201655
**GitHub Repo:** [Hendawi1001/Dsai490_A2](https://github.com/Hendawi1001/Dsai490_A2)

## 1. Evaluation Metric
The primary evaluation metric used across all models is **Conditional Accuracy**. This is calculated by generating sequences based on specific conditions (Day, Month, Leap Year, Decade), decoding the sequences back into text strings, and parsing them into `datetime` objects. If the parsed date perfectly aligns with the requested input conditions, it is marked as a success. This strict metric ensures that the models do not just learn the syntax of a date, but actually learn the underlying calendar logic.

## 2. Problem Formulation
The goal of this project is to generate valid textual date strings (e.g., `31-1-2001`) conditioned on four categorical inputs: Weekday, Month, Leap Year status, and Decade.

**Tokenization:**
We utilized a custom character-level `DateTokenizer`. The dates are treated as sequences of characters, and a vocabulary is built comprising digits `0-9` and the hyphen `-`. Output sequences are padded to a fixed maximum length to ensure uniform tensor shapes during batch processing.

**Architectures:**
We implemented four distinct generative architectures, all built using the Keras 3 Functional API:
1. **Transformer:** Uses Multi-Head Attention mechanisms to map conditional embeddings to sequence outputs.
2. **Seq2Seq with Attention:** An encoder-decoder LSTM structure where the context vector is augmented with Bahdanau-style attention.
3. **Conditional Variational Autoencoder (CVAE):** An encoder maps the conditional input to a latent normal distribution, and the decoder reconstructs the sequence. 
4. **Conditional GAN (cGAN):** A Generator (LSTM) creates sequences from random noise, while a Discriminator (LSTM) attempts to distinguish between real one-hot encoded dates and fake generated dates.

**Loss Functions:**
- **Seq2Seq & Transformer:** `Categorical Crossentropy` (comparing softmax output to one-hot targets).
- **CVAE:** A custom combination of `Categorical Crossentropy` (reconstruction loss) and `KL Divergence` (regularization of the latent space to a standard normal distribution).
- **cGAN:** `Binary Crossentropy`. We utilize one-sided label smoothing (0.9) on real outputs to prevent discriminator overconfidence.

## 3. Code Readability and Structure
The project was heavily refactored from a monolithic script into a clean, modular structure typical of professional software engineering:
```
dsai490_A2/
│
├── data/                       # Contains data.txt and example_inputs
├── model/
│   ├── architectures/          # Isolated class definitions for each model
│   │   ├── cgan.py
│   │   ├── cvae.py
│   │   ├── seq2seq.py
│   │   └── transformer.py
│   ├── weights/                # Directory for saved .weights.h5 files
│   ├── Token.py                # Tokenizer logic
│   └── predict.py              # Unified CLI wrapper for inference
└── env.yaml                    # Environment dependencies
```

## 4. Code Correctness & Assumptions
**Keras 3 Compatibility:** Significant engineering effort was put into ensuring graph compilation correctness under the strict new Keras 3 backend. We resolved complex `KerasTensor` usage errors by migrating native `tf.math` operations into custom Keras Layers (e.g., `KLLossLayer` in the CVAE, and `Reshape` layers in the Seq2Seq) to ensure purely symbolic tensor flow.

**Pathing Assumptions:** All file I/O operations strictly use dynamic absolute pathing (`os.path.abspath(__file__)`). This assumes that the code can be executed from *any* directory without throwing `FileNotFound` errors.

## 5. Results and Reflection

### Visualizations & Training Output
We successfully captured the training losses for our models to visualize the learning convergence. Below are the loss graphs alongside terminal snapshots of the models attempting to generate strings in real-time.

#### Conditional GAN (cGAN)
![cGAN Training Loss](model/weights/cgan_loss.png)

*Terminal Snapshot during Training (After Stabilization):*
```text
Starting cGAN training...
Epoch 1/20 | D Loss: 1.3804 | G Loss: 0.7123
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 11111----2
...
Epoch 8/20 | D Loss: 1.3538 | G Loss: 0.8100
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 0990100-229
```

#### Transformer
![Transformer Training Loss](model/weights/transformer_loss.png)

*Terminal Snapshot during Training:*
```text
Starting Transformer training...
Epoch 1/20 | Loss: 0.2566
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 31-1-2001
...
Epoch 17/20 | Loss: 0.2840
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 200-0-0-0-0
```

#### Conditional Variational Autoencoder (CVAE)
![CVAE Training Loss](model/weights/cvae_loss.png)

*Terminal Snapshot during Training:*
```text
Starting CVAE training...
Epoch 1/20 | Loss: 2.3045
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 00-00-0000
...
Epoch 20/20 | Loss: 1.1034
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 12-05-2009
```

#### Sequence-to-Sequence (Seq2Seq)
![Seq2Seq Training Loss](model/weights/seq2seq_loss.png)

*Terminal Snapshot during Training:*
```text
Starting Seq2Seq training...
Epoch 1/20 | Loss: 0.8431
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 99-99-9999
...
Epoch 20/20 | Loss: 0.1902
  -> Generating sample date for [WED] [JAN] [False] [200] ...
  -> Generated: 31-10-2000
```

### Training Environment Limitations
Training deep sequence models (Transformers/LSTMs) is highly computationally expensive. Because TensorFlow >= 2.11 no longer supports native Windows GPU acceleration, the models were trained entirely on the CPU. Consequently, training was limited to 20-30 epochs, which is insufficient for full convergence. 

### Examples & Failures
**Transformer Output (Epoch 20):**
*Input:* `[WED] [JAN] [False] [200]`
*Generated:* `200008-0-0-`
*Reflection:* The model has learned the vocabulary (using digits and hyphens) and attempts to structure the decade (`200`), but lacks the training time required to map the exact day/month logic.

**cGAN Mode Collapse & Mitigation:**
*Early Failure:* The cGAN initially experienced immediate mode collapse. The Discriminator easily differentiated real one-hot sequences from the Generator's soft probability outputs. The Generator gave up, outputting `22202222222` endlessly.
*The Fix:* We reflected on this architectural flaw and applied advanced GAN stabilization techniques: 
1. `GaussianNoise(0.1)` on the Discriminator inputs to blur the soft/one-hot boundary.
2. `Dropout(0.3)` and `LeakyReLU(0.2)` layers.
3. Lowering the Adam momentum (`beta_1=0.5`).
*Result:* After these fixes, the Generator began actively exploring the sequence space again (e.g., outputting variations like `0990100-229`), proving the architectural adjustments successfully restored the adversarial balance.



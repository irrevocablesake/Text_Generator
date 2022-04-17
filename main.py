import tensorflow as tf
import numpy as np
import os
import time

embedding_dim = 256
rnn_units = 1024
BATCH_SIZE = 64
seq_length = 100
BUFFER_SIZE = 10000

class Helper:
    def adjust_Log_level(self,level):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = level

class Parser:
    save_location = "I:/programming/Projects/text generation/dataset/dataset.txt"
    data = ""

    def input_data_from_url(self,file_url):
        path_to_file =  tf.keras.utils.get_file(self.save_location,file_url)

    def read_data(self):
        self.data = open(self.save_location,'rb').read().decode(encoding='utf-8')

    def print_data(self):
        print(self.data)

    def length_data(self):
        return len(self.data)

    def distinct_characters(self):
        return sorted(set(self.data))

    def split_string(self,input):
        input_text = input[:-1]
        target_text = input[1:]
        return input_text, target_text

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

class One_Character_Model(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_character(self, inputs, states=None):
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')

    input_ids = self.ids_from_chars(input_chars).to_tensor()

    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    predicted_logits = predicted_logits + self.prediction_mask

    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    predicted_chars = self.chars_from_ids(predicted_ids)

    return predicted_chars, states


class TextGeneration:
    helper = Helper()
    parser = Parser()
    ids_from_chars=""
    chars_from_ids=""
    model = ""
    dataset = ""
    character_model = ""
    epoch = 0

    def __init__(self, epoch, data_url):
        self.helper.adjust_Log_level('3')
        self.parser.input_data_from_url(data_url)
        self.parser.read_data()

        alphabets = self.parser.distinct_characters()

        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(alphabets), mask_token=None)

        all_ids = self.ids_from_chars(tf.strings.unicode_split(self.parser.data, 'UTF-8'))
        
        self.chars_from_ids = tf.keras.layers.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        chars = self.chars_from_ids(all_ids)

        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        examples_per_epoch = len(self.parser.data)//(seq_length+1)

        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

        self.epoch = epoch

        self.dataset = sequences.map(self.parser.split_string)

        self.dataset = (
            self.dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

        vocab_size = len(alphabets)

        self.model = MyModel(
    vocab_size=len(self.ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

    def trainModel(self):
        for input_example_batch, target_example_batch in self.dataset.take(1):
            example_batch_predictions = self.model(input_example_batch)
        
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)

        tf.exp(example_batch_mean_loss).numpy()

        self.model.compile(optimizer='adam', loss=loss)

        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

        EPOCHS = self.epoch

        fitted_model = self.model.fit(self.dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
        self.character_model = One_Character_Model(self.model, self.chars_from_ids, self.ids_from_chars)

    def generateText(self, prompt, result_length_data):
        states = None
        next_char = tf.constant([prompt])
        result = [next_char]

        for n in range(result_length_data):
            next_char, states = self.character_model.generate_one_character(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        line = result[0].numpy().decode('utf-8')
        parsed_output = line.replace('\n', '<br/>')

        return parsed_output
        
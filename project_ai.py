from __future__ import print_function
import refer
import random
import numpy as np
from keras.models import load_model
import sys
import argparse
seqlen = 40
step = 3
Lyrics_file = "corpus.txt"
epochs = 45
Accurate_lyrics = 1.0
txt = refer.corpus_read(Lyrics_file)
chars = refer.char_extract(txt)
seq, chars_next = refer.create_seq(txt, seqlen, step)
index_char, indices_char = refer.char_index(chars)
X, y = refer.vector(seq, seqlen, chars,index_char,chars_next)
model = refer.model_build(seqlen, chars)
# To Train the model, uncomment this line.
#model.fit(X, y, batch_size=128, nb_epoch=epochs)
#model.save("any_name.h5")
model = load_model("test.h5")  # you can skip training by loading the trained weights
for Accurate_lyrics in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    print()
    print('Variations:', Accurate_lyrics)
    output_generate = ''
    given_input = "All the life she has seen All the meaner"
    given_input = given_input.lower()
    output_generate += given_input
    print('Generated output using seed: "' + given_input + '"')
    sys.stdout.write(output_generate)
    for i in range(500):
        x = np.zeros((1, seqlen, len(chars)))
        for t, char in enumerate(given_input):
            x[0, t, index_char[char]] = 1.
        predictions = model.predict(x, verbose=0)[0]
        next_index = refer.sample_test(predictions, Accurate_lyrics)
        chars_next = indices_char[next_index]
        output_generate += chars_next
        given_input = given_input[1:] + chars_next
        sys.stdout.write(chars_next)
        sys.stdout.flush()
    print()
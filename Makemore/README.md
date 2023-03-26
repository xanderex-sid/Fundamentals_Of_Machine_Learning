# PROJECT: Makemore

* `makemore` takes `one text file as input`, where each line is assumed to be one training thing, and `generates more things like it`.
* It is `an autoregressive character-level language model`, with a wide choice of `models from bigrams, MLP all the way to a Transformer` (exactly as seen in GPT), used library `Pytorch`.

Making Everything From Scratch:
* <a href="https://github.com/RustyGrackle/Fundamentals_Of_Machine_Learning/blob/main/Makemore/Back_Propagation_and_Creating_Neuron_Layers_MLP.ipynb">Back Propagation and Creating Neuron, Layers, MLP</a> : In this Notebook, I created `Backpropagation Algorithm` from scratch and created `Neuron`, `layers`, `Multi-layer Perceptron (MLP)` from scratch (A `mini version` of `Pytorch library` from scratch). This is made so, to understand basics of Neural Network and inside the Pytorch library.


## Different Models Used To Make `Makemore`:

* <a
href="https://github.com/RustyGrackle/Fundamentals_Of_Machine_Learning/blob/main/Makemore/BigramLanguageModelling.ipynb">Makemore Project Using Bigram</a> : In this notebook, I have made `Bigram character level language model` from scratch, and then build `Neural Network` using `Pytorch`, which does the same job as `Bigram model`. we have used total `32,033` words to model this.

* <a href="https://github.com/RustyGrackle/Fundamentals_Of_Machine_Learning/blob/main/Makemore/Character_level_Language_Modeling_Using_MLP.ipynb">Makemore Project Using MLP</a> : In this notebook, I have made 'MLP character level language model' from scratch, and `generated new names using 32,033 names`.

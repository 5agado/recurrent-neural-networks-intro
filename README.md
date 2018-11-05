# Recurrent Neural Networks Intro
Resources, utils and test code for Recurrent Neural Networks (RNN).

## Text Generation
Current main focus and coverage for this repository is around text-generation.

You can find two separate sets of resources, one related to model training and the other related to model serving and consuming (see also [this Medium entry](https://towardsdatascience.com/practical-text-generation-with-tensorflow-serving-3fa5c792605e)).

For training refer to:
* [RNN with Keras - Text Generation](RNN%20with%20Keras%20-%20Text%20Generation.ipynb)
* [RNN Text Generation - Advanced](RNN%20Text%20Generation%20-%20Advanced.ipynb)

In the `src` folder you will find instead implementation for the **consuming middleware**. It includes a basic class responsible for text pre and post-processing, a procedure for text generation (which builds upon multiple model calls and secondary requirements) and a proxy to handle different models.


## Resources
* [Karpathy’s article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)  
* [Crash Course in Recurrent Neural Networks](http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)  
* [Denny Britz tutorials](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)  
* [Understanding LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
* [A noob’s guide to implementing RNN-LSTM using Tensorflow](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/)  
* [Predicting sequences of vectors (regression) in Keras using RNN - LSTM](http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/)

### Dataset
* [Cornell Movie — Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* [dataset_30000_published_crossword_puzzles](https://www.reddit.com/r/datasets/comments/46jol1/dataset_30000_published_crossword_puzzles/)
* [short text corpus](https://github.com/svenvdbeukel/Short-text-corpus-with-focus-on-humor-detection)
* [fortune cookies galore](https://github.com/ianli/fortune-cookies-galore)

## License

Released under version 2.0 of the [Apache License].

[Apache license]: http://www.apache.org/licenses/LICENSE-2.0

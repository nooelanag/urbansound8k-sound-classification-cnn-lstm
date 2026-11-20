# Urban Sound Classification with CNN-LSTM (UrbanSound8K)

Lab session of *Applications of Speech, Audio, Image and Video Processing*
(Master's in Telecommunication Engineering, UC3M).

The goal is to classify urban sounds from the
[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset
(8,732 labeled excerpts of up to 4 s, 10 classes: air conditioner, car horn,
children playing, dog bark, drilling, engine idling, gun shot, jackhammer,
siren and street music).

## Pipeline
1. **Feature extraction:** log-mel spectrograms (hop length 512, window 1024,
   128 mel bands), fixed to 128 frames (about 3 s) by cropping or padding with
   −80 dB. Each spectrogram is a 128 × 128 single-channel "image".
2. **Data augmentation (online):** random right shift of up to 12 frames and
   additive Gaussian noise, each applied with a probability. Validation and test
   data are not augmented.
3. **Model (CNN-LSTM):** three convolutional blocks (32, 64 and 64 filters,
   3×3 kernels, ELU, max-pooling and dropout) that turn the spectrogram into a
   sequence of 5 time steps of 128 features, followed by a 2-layer LSTM
   (hidden size 64) and a dense output layer with log-softmax.
4. **Training:** 15 epochs, batch size 32, with the dataset folds split into
   training, validation and test sets.

## Experiments
Three variants of the "many-to-one" step were compared:
- **Baseline:** keep the output of the last LSTM time step.
- **Mean-pooling:** average the LSTM outputs over the 5 time steps.
- **Bidirectional LSTM:** two LSTMs in opposite directions (output size doubled).

## Results

| Model | Best validation accuracy | Min. validation loss | Test accuracy |
|---|---|---|---|
| CNN-LSTM (last time step) | 72.54% | 0.850 | 65.85% |
| CNN-LSTM + mean-pooling | 70.42% | 0.961 | **68.64%** |
| CNN-BiLSTM | 68.86% | 0.873 | 62.61% |

Observations:
- All models overfit after a few epochs (training accuracy around 95%) and the
  validation curves are noisy.
- Results change between runs (random initialization, minibatch order and
  augmentation): a second run of the baseline gave 67.30% best validation
  accuracy and 66.18% test accuracy. Mean-pooling gives a better test accuracy
  in this run, but the difference is small and not conclusive.
- The bidirectional LSTM did not improve the results. A possible reason is that
  the last time step of the backward direction has only seen the last element
  of the sequence.

## How to run
1. Run the preprocessing notebook in Google Colab to download the dataset and
   compute `us8k_features.pkl` (it can take more than one hour; run it once).
2. Open `LabSession2_Sound_Classification.ipynb` in Colab with a GPU runtime
   and run the cells in order (about 2 minutes per training run on GPU).

The dataset and the extracted features are not included in this repository.

## Notes
The lab statement and the base code were provided by the course instructors.

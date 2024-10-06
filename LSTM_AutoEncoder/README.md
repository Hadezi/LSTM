# LSTM Auto-Encoder (LSTM-AE) implementation in Pytorch
The code implements three variants of LSTM-AE:
1. Regular LSTM-AE for reconstruction tasks (LSTMAE.py)
2. LSTM-AE + Classification layer after the decoder (LSTMAE_CLF.py)
3. LSTM-AE + prediction layer on top of the encoder (LSTMAE_PRED.py)

To test the implementation, we defined three different tasks:

Ship Fuel oil Consumption example (on random uniform data) for sequence prediction by LSTMAE_PRED:
```
python lstm_ae_sfoc.py
```

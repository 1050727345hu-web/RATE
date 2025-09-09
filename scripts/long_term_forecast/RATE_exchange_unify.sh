#export CUDA_VISIBLE_DEVICES=0

model_name=RATE

seq_len=96
down_sampling_layers=6
down_sampling_window=2
learning_rate=0.001
d_model=512
batch_size=8
train_epochs=40
patience=5

python -u run_risk.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size  \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \


python -u run_risk.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size  \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \


python -u run_risk.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size  \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \


python -u run_risk.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size  \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
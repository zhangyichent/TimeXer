export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset \
  --data_path weather/weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --label_len 0 \
  --embed timeF \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 128 \
  --des Exp \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --target CO2

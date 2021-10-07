# aifarm (소달구지)

## Implementation Details.
- Implemented with base_line code
- Used Modified Cross Etropy Loss
   - Cross Entropy Loss (ce)
   - Weight Cross Etropy Loss (w_ce)
   - Focal Loss(focal_loss)
   - Weight Focal Loss (w_focal_loss)  <= Mainly Used
- Top 5 Models Trained with corresponding Config Files.
   - tf_efficientnetv2_l_in21ft1k => train_config_best1
   - tf_efficientnetv2_m_in21ft1k => train_config_best2
   - tf_efficientnet_b6_ns => train_config4
   - tf_efficientnetv2_l_in21ft1k => train_config2
   - swin_large_patch4_window7_224 => train_config3

## Directory.
```bash
├── project_folder
│   ├── config
│   ├── models
│   ├── modules
│   ├── results
│   ├── command.py
│   ├── logger_20210712.py
│   ├── predict.py
│   ├── README.md
│   └── train.py
└── data
    ├── test
    └── train
        ├──Tomato_D01
        ├──Tomato_D01
        ...
        └──Tomato_R01
```
## Train Process
```
python command.py
```
<command.py>
```
...

command_list = [
###

## Best
"python train.py --yml train_config_best1",  # tf_efficientnetv2_m_in21ft1k, bathsize 32 => pub test: 99.03
"python train.py --yml train_config_best2",  # tf_efficientnetv2_l_in21ft1k, bathsize 16 => pub test: 99.22

###
]

...
```
## Example Config File (for model: tf_efficientnetv2_m_in21ft1k)
```
SEED:
  random_seed: 42
DATALOADER:
  num_workers: 8
  shuffle: true
  pin_memory: true
  drop_last: false
TRAIN:
  input_shape: 224
  num_epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  early_stopping_patience: 10
  model: tf_efficientnetv2_m_in21ft1k
  optimizer: adam
  scheduler: msl
  momentum: null
  weight_decay: 5.0e-05
  loss_function: w_focal_loss
  metric_function: null
PERFORMANCE_RECORD:
  column_list:
  - train_serial
  - train_timestamp
  - model_str
  - optimizer_str
  - loss_function_str
  - metric_function_str
  - early_stopping_patience
  - batch_size
  - epoch
  - learning_rate
  - momentum
  - random_seed
  - epoch_index
  - train_loss
  - validation_loss
  - train_score
  - validation_score
  - elapsed_time
```

# Top 2 Models      
## tf_efficientnetv2_m_in21ft1k_20211006215437 
   - train: 0.999549062049062  val: 0.994590695997115
   - loss: w_focal_loss
   - sche: msl
   - optim: adam
   - Test
      - Private: 0.9903
   ![image](https://user-images.githubusercontent.com/55650445/136344916-c683b495-2b96-465a-98b6-80e945d61efa.png)
   ![image](https://user-images.githubusercontent.com/55650445/136344881-9567bd17-3691-4134-b5bd-f26f11cef75a.png)

## tf_efficientnetv2_l_in21ft1k_20211007123012 
   - train: 0.9999098124098124 val: 0.997836278398846
   - loss: ce
   - sche: msl
   - optim: adam
   - Test
      - Private: 0.9922 
   ![image](https://user-images.githubusercontent.com/55650445/136345336-5aa283c0-0c81-4eba-a4a5-b08fd1b20c27.png)
   ![image](https://user-images.githubusercontent.com/55650445/136345371-9fee8179-a369-4eb3-ae8e-fc2cc2d190c4.png)


### Best Score Public: 0.9934

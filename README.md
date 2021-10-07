# aifarm
implemented with base_line code

Best Score Public: 0.9934
1. tf_efficientnetv2_l_in21ft1k_20211007123012 
   - train: 0.9999098124098124 val: 0.997836278398846
   - loss: ce
   - sche: msl
   - optim: adam
   ![image](https://user-images.githubusercontent.com/55650445/136345336-5aa283c0-0c81-4eba-a4a5-b08fd1b20c27.png)
   ![image](https://user-images.githubusercontent.com/55650445/136345371-9fee8179-a369-4eb3-ae8e-fc2cc2d190c4.png)
   - Test
      - Private: 0.9922
2. tf_efficientnetv2_m_in21ft1k_20211006215437 
   - train: 0.999549062049062  val: 0.994590695997115
   - loss: w_focal_loss
   - sche: msl
   - optim: adam
   ![image](https://user-images.githubusercontent.com/55650445/136344916-c683b495-2b96-465a-98b6-80e945d61efa.png)
   ![image](https://user-images.githubusercontent.com/55650445/136344881-9567bd17-3691-4134-b5bd-f26f11cef75a.png)
   - Test
      - Private: 0.9903
    

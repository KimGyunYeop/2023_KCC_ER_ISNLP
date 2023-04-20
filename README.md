# 2022_KCC_휴먼이해 인공지능_ISNLP

## Methodolgy

<h3>Prompt with Audio Feature</h3>  
Audio Feature Encoder를 통해 추출한 Audio Feature를 Prompt learning의 언어모델에서 사용할 수 있도록 구조한 모델  

![2023_kcc_그림](https://user-images.githubusercontent.com/44080708/231979196-05943962-ffd6-41fb-bf57-b0b19678d127.jpg)

<h3>Prompt with Previous Context</h3> 
과거 발화에 대한 문맥정보를 사용하여 현재 발화에 대한 예측성능을 향상시키도록 유도  

![2023_kcc_eq](https://user-images.githubusercontent.com/44080708/231981520-684da9ef-a441-42c1-928b-c899ae9f5aa1.JPG)


## Requirement

    transformers
    pytorch
    torchaudio
    matplotlib
    scikit-learn
    tqdm
    pandas
    numpy

## Usage

Baseline (T5-prompt)  

    python -u main.py \
        --result_path {$result folder name} 
        

Prompt with Audio Feature 

    python -u main.py\
        --result_path {$result folder name} \
        --T5AFModel \
        --AF_num {$l} 
        

Prompt with Previous Context

    python -u main.py\
        --result_path {$result folder name} \
        --prev_turn_loss \
        --prev_turn {$k} \
        --d {$d}
        

Proposed model

    python -u main.py\
        --result_path {$result folder name} \
        --prev_turn_loss \
        --prev_turn {$k} \
        --d {$d} \
        --T5AFModel \
        --AF_num {$l} 
        
        
## Dataset
/datasets 폴더에 KEMDy20을 해제하면 동일한 구조를 얻을 수 있음  
논문의 실험에서는 30개의 대화 세션을 학습용으로 10개의 대화세션을 평가용으로 사용하였음  

    /datasets
       └——————annotation
       └——————EDA
       └——————IBI
       └——————TEMP
       └——————wav
         
학습용 대화 세션 목록 : ['26', '33', '13', '39', '15', '14', '03', '38', '35', '18', '40', '01', '30', '06', '17', '36', '08', '16', '23', '21', '27', '10', '34', '32', '29', '07', '37', '28', '20', '25']  
평가용 대화 세션 목록 : ['19', '12', '22', '31', '04', '09', '24', '11', '02', '05']
        
## Performance

![2023_kcc_eq](<img width="952" alt="image" src="https://user-images.githubusercontent.com/64178197/233313211-9dbb4af4-848d-42ce-9005-48b3caa6e15b.png">)

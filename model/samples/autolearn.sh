#!/bin/bash

### ========================================================
### 3월 15일 주말동안 돌려놓을거
### ========================================================

# cd test_special_token
# echo Testing special token..
# python3 test_special_token.py
# cd ..
# sleep 5


# cd test_question_prompt
# echo Testing prompt..
# python3 test_question_prompt.py
# cd ..
# sleep 5


# cd test_kcELECTRA
# echo Testing KoElectra..
# python3 test_kcELECTRA.py --data_dir ../data_split/Training_행복그림1.csv --output_dir ./output_행복긍정_special --epoch 5 --processing_type special_token --eval_steps 87
# python3 test_kcELECTRA.py --data_dir ../data_split/Training_행복그림1.csv --output_dir ./output_행복긍정_prompt --epoch 5 --processing_type prompt --eval_steps 87
# python3 test_kcELECTRA.py --data_dir ../data_split/Training_12345.csv --output_dir ./output_alldata_special --epoch 10 --processing_type special_token --eval_steps 218
# python3 test_kcELECTRA.py --data_dir ../data_split/Training_12345.csv --output_dir ./output_alldata_prompt --epoch 10 --processing_type prompt --eval_steps 218

# python3 test_kcELECTRA.py --data_dir ../data_split/Training_12345.csv --epoch 10 --processing_type prompt --eval_steps 1

# cd ..
# sleep 5


### ========================================================
### 3월 18일
### ========================================================

# cd test_kcELECTRA
# echo Testing KoElectra..
# python3 test_kcELECTRA_train.py --data_dir ../data_split/Training_불행그림3.csv --output_dir ./output_불행그림3_special --epoch 5 --processing_type special_token --eval_steps 87
# python3 test_kcELECTRA_train.py --data_dir ../data_split/Training_불행그림3.csv --output_dir ./output_불행그림3_prompt --epoch 5 --processing_type prompt --eval_steps 87

# python3 test_kcELECTRA_train.py --data_dir ../data_split/Training_12345_PCA30.csv --output_dir ./output_PCA30_prompt --epoch 5 --processing_type prompt --eval_steps 218 --addi_feat_size 30



### ========================================================
### 3월 19일 낮
### ========================================================

# Validation code

# cd test_kcELECTRA

# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_prompt/checkpoint-4360/ --data_dir ../data_split/Validation_행복.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_prompt/checkpoint-4360/ --data_dir ../data_split/Validation_불행.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_prompt/checkpoint-4360/ --data_dir ../data_split/Validation_그림1.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_prompt/checkpoint-4360/ --data_dir ../data_split/Validation_그림2.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_prompt/checkpoint-4360/ --data_dir ../data_split/Validation_그림3.csv --processing_type prompt

# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_special/checkpoint-4360/ --data_dir ../data_split/Validation_행복.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_special/checkpoint-4360/ --data_dir ../data_split/Validation_불행.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_special/checkpoint-4360/ --data_dir ../data_split/Validation_그림1.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_special/checkpoint-4360/ --data_dir ../data_split/Validation_그림2.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path ./output_alldata_special/checkpoint-4360/ --data_dir ../data_split/Validation_그림3.csv --processing_type special_token


# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_prompt/checkpoint-870/ --data_dir ../data_split/Validation_행복.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_prompt/checkpoint-870/ --data_dir ../data_split/Validation_그림1.csv --processing_type prompt

# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_special/checkpoint-870/ --data_dir ../data_split/Validation_행복.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_special/checkpoint-870/ --data_dir ../data_split/Validation_그림1.csv --processing_type special_token

# echo 행복긍정 prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_prompt/checkpoint-870/ --data_dir ../data_split/Validation_행복.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_prompt/checkpoint-870/ --data_dir ../data_split/Validation_그림1.csv --processing_type prompt

# echo 행복긍정 special
# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_special/checkpoint-870/ --data_dir ../data_split/Validation_행복.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path ./output_행복긍정_special/checkpoint-870/ --data_dir ../data_split/Validation_그림1.csv --processing_type special_token

# echo 행복불행 prompt
# python3 test_kcELECTRA_infer.py --model_path /home/ncp/workspace/blocks1/test/test_question_prompt/output_KcELECTRA-base/checkpoint-200 --data_dir ../data_split/Validation_행복.csv --processing_type prompt
# python3 test_kcELECTRA_infer.py --model_path /home/ncp/workspace/blocks1/test/test_question_prompt/output_KcELECTRA-base/checkpoint-200 --data_dir ../data_split/Validation_불행.csv --processing_type prompt

# echo 행복불행 special
# python3 test_kcELECTRA_infer.py --model_path /home/ncp/workspace/blocks1/test/test_special_token/output_KcELECTRA-base/checkpoint-280 --data_dir ../data_split/Validation_행복.csv --processing_type special_token
# python3 test_kcELECTRA_infer.py --model_path /home/ncp/workspace/blocks1/test/test_special_token/output_KcELECTRA-base/checkpoint-280 --data_dir ../data_split/Validation_불행.csv --processing_type special_token


### lr 테스트
# cd test_kcELECTRA
# echo Testing KoElectra..
# python3 test_kcELECTRA_train.py --data_dir ../data_split/Training_불행그림3.csv --output_dir ./output_불행그림3_special_lr4 --epoch 2 --processing_type special_token --eval_steps 70 --lr 2e-4
# python3 test_kcELECTRA_train.py --data_dir ../data_split/Training_불행그림3.csv --output_dir ./output_불행그림3_special_lr3 --epoch 2 --processing_type special_token --eval_steps 70 --lr 2e-3






### ========================================================
### 3월 19일 저녁
### ========================================================


### 질문별 모델 가중치 생성 (10시간 소요 예정)
# cd test_each
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_행복.csv --output_dir ./output_행복 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_불행.csv --output_dir ./output_불행 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_그림1.csv --output_dir ./output_그림1 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_그림2.csv --output_dir ./output_그림2 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_그림3.csv --output_dir ./output_그림3 --epoch 5 --eval_steps 50

# 아래 스크립트와 같이 각 모델 가중치에 대해서 학습한 질문대로 Validation set 점수 (recall, f1, accuracy, precision...) 확인 바랍니다.
# each 코드를 별도로 만든 건 질문 구분을 없애기 위해서입니다 (processing_type)
# python3 each_kcELECTRA_infer.py --model_path ./output_행복/checkpoint-???/ --data_dir ../data_split/Validation_행복.csv

## 인수
# 데이터 정규화를 했을 때 성능 변화 관찰
# echo "===========================================running 불행 standard==========================================="
# python3 each_kcELECTRA_train.py --standardized True --data_dir ../data_split/Training_불행_standard.csv --output_dir ./output_불행_standard --epoch 5 --eval_steps 50
# 나이 정보 사용 여부에 따른 성능 변화 관찰
# echo "===========================================running 불행 noage==========================================="
# python3 each_kcELECTRA_train.py --use_age False --data_dir ../data_split/Training_불행.csv --output_dir ./output_불행_noage --epoch 5 --eval_steps 50
# 텍스트 전체를 사용했을 때의 성능 변화 관찰 :: infer 돌려봐야함
# python3 each_kcELECTRA_train.py --allow_overflow True --data_dir ../data_split/Training_불행.csv --output_dir ./output_불행_overflow --epoch 5 --eval_steps 50


### 후보 모델들 인적정보 추가하여 불행 질문에 대한 성능 테스트
# cd ..
# cd test_personal_info
# python3 personal_test.py
# 해당 폴더에서 output 모델 kcbert-base, funnel-kor-base 결과 확인 바랍니다
# 두 모델이 electra보다 성능이 좋아서 KcELECTRA 외의 후보로 사용하기 좋을 것 같아 테스트해보았습니다.



### ========================================================
### 3월 21일
### ========================================================

# 문제 개별 가중치 학습 다시 돌리기
# cd test_each
# echo Training Each KoElectra..
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_행복.csv --output_dir ./output_행복 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_불행.csv --output_dir ./output_불행 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_그림1.csv --output_dir ./output_그림1 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_그림2.csv --output_dir ./output_그림2 --epoch 5 --eval_steps 50
# python3 each_kcELECTRA_train.py --data_dir ../data_split/Training_그림3.csv --output_dir ./output_그림3 --epoch 5 --eval_steps 50
# cd ..
# sleep 5

# # 아키텍쳐 2중 레이어 재실험 (2중 레이어에 Relu 추가, abs > +1 > log)
# cd test_arch_layer2_again
# echo Training double-layered KoElectra..
# python3 arch_layer2_test.py --eval_steps 30
# # python3 arch_layer2_test.py --model_path ../../models/kcbert-base --output_dir ./output_kcbert --eval_steps 30
# # python3 arch_layer2_test.py --model_path ../../models/funnel-kor-base --output_dir ./output_funnel --eval_steps 30
# cd ..
# sleep 5


# # Freeze 재실험 (2중 레이어 사용)
# cd test_arch_freeze_again
# echo Training double-layered freeze KoElectra..
# python3 arch_freeze_test.py --eval_steps 30
# cd ..
# sleep 5

# echo Done Training!


### ========================================================
### 3월 22일
### ========================================================
cd test_kcELECTRA
echo prompt_불행
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_prompt/checkpoint-3270/" --data_dir ../data_split/Validation_불행.csv
echo prompt_그림1
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_prompt/checkpoint-3270/" --data_dir ../data_split/Validation_그림1.csv
echo prompt_그림2
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_prompt/checkpoint-3270/" --data_dir ../data_split/Validation_그림2.csv
echo prompt_그림3
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_prompt/checkpoint-3270/" --data_dir ../data_split/Validation_그림3.csv

echo prompt_행복
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_special/checkpoint-3270/" --data_dir ../data_split/Validation_행복.csv --processing_type special_token
echo prompt_불행
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_special/checkpoint-3270/" --data_dir ../data_split/Validation_불행.csv --processing_type special_token
echo prompt_그림1
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_special/checkpoint-3270/"  --data_dir ../data_split/Validation_그림1.csv --processing_type special_token
echo prompt_그림2
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_special/checkpoint-3270/"  --data_dir ../data_split/Validation_그림2.csv --processing_type special_token
echo prompt_그림3
python3 test_kcELECTRA_infer.py --model_path "../test_kcELECTRA/output_alldata_special/checkpoint-3270/"  --data_dir ../data_split/Validation_그림3.csv --processing_type special_token

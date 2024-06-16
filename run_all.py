import os

# 定义所有输入变量
#model_name_or_path = "./model/google_bert_6L"
model_name_or_path = "./model/tinybert_huawei"
teacher_model = "./model/google_bert_base_uncased"

#task_names = ["sst2","qnli","mrpc","rte","cola","wnli","stsb","qqp","mnli","ax"]
task_names = ["mnli","qqp"]#,"qnli","wnli","mrpc","rte"]#qqp与mnli数据集最大，最后再训练.
eval_steps = 1000
num_train_epochs = 6

per_device_train_batch_size = 32
per_device_eval_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 3e-05
t_learning_rate = 1e-05
alpha_kd = 0.9
temperature = 1.0

do_train = True
do_eval = True
train_teacher = True

# 构建output_dir
for task_name in task_names:
    output_dir = f"./output/cross_new_distan_tinybert_big_dataset/99ot_lgtm_{task_name}_{num_train_epochs}_correct_MINE_train_step"
    output_dir = os.path.join(".", output_dir)
    print('lr::',learning_rate)
    print('t_lr::', t_learning_rate)
    
    # 构建命令
    command = f"python run_glue_cross_new_distan_new_split.py --model_name_or_path {model_name_or_path_all[task_name]} \
        --teacher_model {teacher_model} \
        --task_name {task_name} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --per_device_eval_batch_size {per_device_eval_batch_size} \
        --learning_rate {learning_rate} \
        --t_learning_rate {t_learning_rate} \
        --alpha_kd {alpha_kd} \
        --temperature {temperature} \
        --num_train_epochs {num_train_epochs} \
        --output_dir {output_dir} \
        --eval_steps {eval_steps} \
        --do_train {do_train} \
        --do_eval {do_eval} \
        --train_teacher {train_teacher} \
        --overwrite_output_dir "
    
    # 运行命令
    os.system(command)

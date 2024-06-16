import os

# 定义所有输入变量
#model_name_or_path = "google/bert_uncased_L-6_H-768_A-12"
teacher_model = "bert-base-uncased"

#task_names = ["qnli","qqp","mrpc","rte","cola","wnli","stsb","ax"]
#task_names = ["sst2","mrpc","rte","cola","wnli","qnli",'mnli','qqp','stsb']#qqp与mnli数据集最大，最后再训练，stsb需要修改alpha_kd才能训练，明天再说
#task_names=['mnli','qqp','stsb']
task_names=["mnli","qqp"]#,"qqp"]#,'sst2','qnli']
eval_steps = 30
num_train_epochs = 6
k_label = 3

per_device_train_batch_size = 32
per_device_eval_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 3e-05
t_learning_rate = 3e-05
alpha_kd = 0.9
temperature = 1.0
t_alpha_kds=[0.3]

do_train = True
do_eval = True
do_predict = True
train_teacher = True


# 构建output_dir
for t_alpha_kd in t_alpha_kds:
    for task_name in task_names:
        output_dir = f"./output/cross_new_distan_tinybert_big_dataset/99ot_lgtm_{task_name}_{num_train_epochs}_correct_MINE_train_step"
        output_dir = os.path.join(".", output_dir)
        model_name_or_path = output_dir
        # 构建命令
        command = f"python run_glue_cross_new_distan_new_split.py --model_name_or_path {model_name_or_path} \
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
            --do_predict {do_predict} \
            --train_teacher {train_teacher} \
            --overwrite_output_dir "
        
        # 运行命令
        os.system(command)

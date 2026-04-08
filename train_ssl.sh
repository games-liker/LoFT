# data="imagenet_lt"
# data="places_lt"
# data="inat2018"
# data="cifar100_ir100"
data="smallimagenet"
# model="openclip_vit_b16"
model="clip_vit_b16"
# model="metaclip_vit_b16"
lambda_u=3.0
lambda_uc=1.0
mu=7
lr=0.01
threshold=0.6
ood_threshold=0.3
batch_size=16
micro_batch_size=16
num_max=50
num_max_u=400
imb_ratio_label=10
imb_ratio_unlabel=10
flag_reverse_LT=0
img_size=64
total_steps=10000
eval_step=100
num_epochs=100
is_open=True  # 设置为True时,会将unlabeled dataset与COCO dataset concat
ood_root=""  # COCO数据集路径
export CUDA_VISIBLE_DEVICES=2
nohup python main_ssl.py \
-d ${data} \
-m ${model} \
total_steps ${total_steps} \
eval_step ${eval_step} \
num_epochs ${num_epochs} \
num_max ${num_max} \
num_max_u ${num_max_u} \
batch_size ${batch_size} \
imb_ratio_label ${imb_ratio_label} \
imb_ratio_unlabel ${imb_ratio_unlabel} \
flag_reverse_LT ${flag_reverse_LT} \
img_size ${img_size} \
micro_batch_size ${micro_batch_size} \
lambda_u ${lambda_u} \
lambda_uc ${lambda_uc} \
lr ${lr} \
threshold ${threshold} \
ood_threshold ${ood_threshold} \
mu ${mu} \
is_open ${is_open} \
ood_root "${ood_root}" \
output_dir "test" \
adaptformer True \
>> output/test.log 2>&1 &
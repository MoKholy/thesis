seed=42
dataset="correctly_labeled"
stratify=true
test_size=0.1
val_size=0.1
batch_size=32
num_workers=3
shuffle=true
pin_memory=true
n_hidden="64 128 256"
dropout=0.3
output_size=12
do_train=false
epochs=100
lr=0.001
final_dim=128
loss_lamda_val=1.2
eps=2.4
optimizer="SGD"
momentum=0.9
weight_decay=0.0005
scheduler="StepLR"
step_size=30
nesterov=true
model_name="testingparsing"

python main.py --seed $seed --dataset $dataset --stratify $stratify --test_size $test_size --val_size $val_size --batch_size $batch_size --num_workers $num_workers --shuffle $shuffle --pin_memory $pin_memory --n_hidden $n_hidden --dropout $dropout --output_size $output_size --do_train $do_train --epochs $epochs --lr $lr --final_dim $final_dim --loss_lamda_val $loss_lamda_val --eps $eps --optimizer $optimizer --momentum $momentum --weight_decay $weight_decay --scheduler $scheduler --step_size $step_size --nesterov $nesterov --model_name $model_name

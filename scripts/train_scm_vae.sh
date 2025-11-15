NUM_MACHINES=1
NUM_LOCAL_GPUS=2
MACHINE_RANK=0
MAIN_MACHINE_PROT="29501"  # fill your machine port here

EXEC_FILE="src/train_scm_vae.py"
CONFIG_FILE="configs/vae_sd15.yaml"
TAG="train_scm_vae_wconf"

OUTPUT_FOLDER="./out"

VAE_PRETRAINED_FOLDER="/data-nas/experiments/zhenqing/cache/stable-diffusion-2-1"
VGGNET_PRETRAINED_MODEL_PATH="/data-nas/experiments/zhenqing/cache/lpips/vgg16-397923af.pth"
SPATIALGEN_DATASET_FOLDER="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k"
SPATIALGEN_TRAIN_SPLIT_FILE="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/8k_perspective_trains.txt"

export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

mkdir -p /root/.cache/torch/hub/checkpoints;
cp $VGGNET_PRETRAINED_MODEL_PATH /root/.cache/torch/hub/checkpoints/;

export CUDA_LAUNCH_BLOCKING=1

# accelerate launch  --multi_gpu --mixed_precision no \
#     --num_machines $NUM_MACHINES \
#     --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
#     --machine_rank $MACHINE_RANK \
#     --main_process_port $MAIN_MACHINE_PROT \
#     ${EXEC_FILE} \
#         --config_file ${CONFIG_FILE} \
#         --tag ${TAG} \
#         --pin_memory \
#         --use_grad_loss \
#         --allow_tf32 \
# $@

# single GPU version
accelerate launch  --mixed_precision no \
    --config_file configs/accelerate-train.yaml \
    ${EXEC_FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --use_grad_loss \
        --predict_conf \
        --allow_tf32 \
        --output_dir $OUTPUT_FOLDER \
        opt.spatialgen_data_dir=$SPATIALGEN_DATASET_FOLDER \
        opt.train_split_file=$SPATIALGEN_TRAIN_SPLIT_FILE \
        opt.input_res=512 \
        opt.num_input_views=1 \
        opt.num_views=2 \
        opt.use_scene_coord_map=true \
        opt.use_metric_depth=false \
        opt.pretrained_model_name_or_path=$VAE_PRETRAINED_FOLDER \
        opt.vggnet_pretrained_model_path=$VGGNET_PRETRAINED_MODEL_PATH \
$@


# multi GPU version
# accelerate launch --mixed_precision fp16 \
#     --num_machines $NUM_MACHINES \
#     --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
#     --machine_rank $MACHINE_RANK \
#     --main_process_port $MAIN_MACHINE_PROT \
#             src/train_scm_vae.py \
#             --config_file $CONFIG_FILE \
#             --tag $TAG \
#             --pin_memory \
#             --use_grad_loss \
#             --predict_conf \
#             --allow_tf32 \
#             --output_dir $OUTPUT_FOLDER \
#             opt.spatialgen_data_dir=$SPATIALGEN_DATASET_FOLDER \
#             opt.train_split_file=$SPATIALGEN_TRAIN_SPLIT_FILE \
#             opt.input_res=512 \
#             opt.num_input_views=1 \
#             opt.num_views=2 \
#             opt.use_scene_coord_map=true \
#             opt.use_metric_depth=false \
#             opt.pretrained_model_name_or_path=$VAE_PRETRAINED_FOLDER \
#             opt.vggnet_pretrained_model_path=$VGGNET_PRETRAINED_MODEL_PATH 
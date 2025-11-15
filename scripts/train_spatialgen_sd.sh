NUM_MACHINES=1
NUM_LOCAL_GPUS=4
MACHINE_RANK=0
MAIN_MACHINE_PROT="29501"  # fill your machine port here

EXEC_FILE="src/train_spatialgen_sd.py"
CONFIG_FILE="configs/spatialgen_sd21.yaml"
TAG="train_spatialgen_sd21"

OUTPUT_FOLDER="./out"

# SD21_PRETRAINED_FOLDER="/data-nas/experiments/zhenqing/cache/stable-diffusion-2-1"
# VGGNET_PRETRAINED_MODEL_PATH="/data-nas/experiments/zhenqing/cache/lpips/vgg16-397923af.pth"
# SPATIALGEN_DATASET_FOLDER="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k"
# SPATIALGEN_TRAIN_SPLIT_FILE="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/8k_perspective_trains.txt"
SD21_PRETRAINED_FOLDER="/alluxio/training/experiments/zhenqing/spatialgen-publish/spatialgen/pretrained_ckpts/spatialgen-1.0"
VGGNET_PRETRAINED_MODEL_PATH="/alluxio/training/experiments/zhenqing/cache/lpips/vgg16-397923af.pth"
SPATIALGEN_DATASET_FOLDER="/alluxio/training/dataset/qunhe/PanoRoom/roomverse_data/processed_data_spiral_randfov"
SPATIALGEN_TRAIN_SPLIT_FILE="/alluxio/training/dataset/qunhe/PanoRoom/roomverse_data/final_57k_perspective_trains.txt"
PRETRAINED_TINY_VAE_FOLDER=/alluxio/training/experiments/zhenqing/spatialgen-publish/spatialgen/pretrained_ckpts/tinyvae-ckpt-047000;

export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

mkdir -p /root/.cache/torch/hub/checkpoints;
cp $VGGNET_PRETRAINED_MODEL_PATH /root/.cache/torch/hub/checkpoints/;

export CUDA_LAUNCH_BLOCKING=1

# single gpu version
# accelerate launch --mixed_precision fp16 \
#     --num_machines $NUM_MACHINES \
#     --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
#     --machine_rank $MACHINE_RANK \
#     --main_process_port $MAIN_MACHINE_PROT \
#     ${EXEC_FILE} \
#         --config_file ${CONFIG_FILE} \
#         --tag ${TAG} \
#         --pin_memory \
#         --use_tiny_vae \
#         --use_scm_conf_map \
#         --use_deepspeed \
#         --allow_tf32 \
# $@

# multi-gpu version
accelerate launch --mixed_precision fp16 \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    --main_process_port $MAIN_MACHINE_PROT \
    ${EXEC_FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --use_grad_loss \
        --use_tiny_vae \
        --use_scm_conf_map \
        --allow_tf32 \
        --output_dir $OUTPUT_FOLDER \
        opt.spatialgen_data_dir=$SPATIALGEN_DATASET_FOLDER \
        opt.train_split_file=$SPATIALGEN_TRAIN_SPLIT_FILE \
        opt.input_res=512 \
        opt.num_input_views=1 \
        opt.num_views=8 \
        opt.prediction_type=v_prediction \
        opt.use_layout_prior=true  \
        opt.use_scene_coord_map=true \
        opt.use_metric_depth=false  \
        opt.input_concat_binary_mask=true  \
        opt.input_concat_warpped_image=true \
        opt.pretrained_model_name_or_path=$SD21_PRETRAINED_FOLDER \
        opt.vggnet_pretrained_model_path=$VGGNET_PRETRAINED_MODEL_PATH \
$@


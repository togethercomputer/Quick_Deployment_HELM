cd /home/fm/dev/Quick_Deployment_HELM

# ./together start >> glm_together.log &

MODEL_TYPE="glm-130b"
CHECKPOINT_PATH="/root/fm/models/glm-130b-sat"
MP_SIZE=8
SEED=1234
MAX_OUTPUT_LENGTH=128
MIN_GEN_LENGTH=0
# BaseStrategy args
TEMP=0.9
TOPK=1
TOPP=0


MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --num-layers 70 \
            --hidden-size 12288 \
            --inner-hidden-size 32768 \
            --vocab-size 150528 \
            --num-attention-heads 96 \
            --max-sequence-length 256 \
            --tokenizer-type icetk-glm-130B \
            --layernorm-order post \
            --load ${CHECKPOINT_PATH} \
            --skip-init \
            --fp16 \
            --seed $SEED \
            --upload-token $S3TOKEN \
            --mode inference \
            --sampling-strategy BeamSearchStrategy \
            --out-seq-length $MAX_OUTPUT_LENGTH \
            --min-gen-length $MIN_GEN_LENGTH \
            --temperature $TEMP \
            --top_k $TOPK \
            --top_p $TOPP"


python3 -m torch.distributed.launch --nproc_per_node $MP_SIZE serving_dist_glm.py $MODEL_ARGS





GLOO_SOCKET_IFNAME=eno1

HOSTNAME=mi250-018
BATCH_SIZE=8

PP_SIZE=1
RING_SIZE=4
ULYSSES_SIZE=8

total=$((PP_SIZE * RING_SIZE * ULYSSES_SIZE))
if (( total % 8 != 0 )); then
    echo "Error: (PP_SIZE * RING_SIZE * ULYSSES_SIZE) must be divisible by 8." >& 2
    exit 1
fi
NNODES=$(( total / 8 ))

ISL=32768
DATE=$(date +%Y%m%d_%H%M%S)

# Set NUM_ITERATIONS based on PP_SIZE
if [ ${PP_SIZE} -eq 1 ]; then
    NUM_ITERATIONS=10
else
    NUM_ITERATIONS=100
fi

LOG_FILE="pp_${PP_SIZE}_r${RING_SIZE}_u${ULYSSES_SIZE}_rank${NODE_RANK}_${ISL}_bsz${BATCH_SIZE}_${DATE}.log"

torchrun --nnodes ${NNODES} --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint ${HOSTNAME}:27000 --node-rank ${NODE_RANK} --role rank measure.py --model /home/share/model/gpt-oss-120b --pp-size ${PP_SIZE} --ring-size ${RING_SIZE} --ulysses-size ${ULYSSES_SIZE} --input-length ${ISL} --num-iterations ${NUM_ITERATIONS} --batch-size ${BATCH_SIZE} 2>&1 | tee logs/${LOG_FILE}

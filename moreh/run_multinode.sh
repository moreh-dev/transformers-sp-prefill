torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint mi250-018:29000 --node-rank 0 --role rank moreh/measure.py --model /model/gpt-oss-120b --pp-size 2 --sp-size 8

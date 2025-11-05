torchrun --nproc-per-node 8 --local-ranks-filter 7 measure.py --model /model/gpt-oss-120b --pp-size 8 --input-length 8192 --num-iterations 10 --output-lengths 1 --pp-split-points 3 8 13 18 23 28 33

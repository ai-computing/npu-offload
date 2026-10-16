#!/bin/bash
cd /home/user/workspace/npu_offload/solar_hybrid
P="서울에서 주말에 가볼 만한 곳을 다섯 군데 추천하고 각각 이유를 설명해줘."
for SW in 0 2; do
  python3 runtime.py --mode hybrid --devs rngd:0,rngd:1 --slots 20,26 --chat --prompt "$P" --n 32 --warmup 1 --swap $SW --cpu-threads 16 > ../logs/full_gather_swap$SW.log 2>&1
done
echo ALLDONE

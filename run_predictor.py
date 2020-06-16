import os, sys
if __name__ == '__main__':
  cmd = 'python ./base_predictor.py \
                    --dir /home/xuechao.shi/FER/code/code_35/FER_detector_lmks_tools/image/ \
                    --save ./test.txt \
                    --select_type all'
  print(cmd)
  os.system('{}'.format(cmd))
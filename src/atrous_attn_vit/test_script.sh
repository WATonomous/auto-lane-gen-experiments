python vit-imagenet.py --seed 55 --epoch 30 --patchsize 112 --atrousAttn 2>&1 |& tee vit_atrous1.txt
python vit-imagenet.py --seed 55 --epoch 30 --patchsize 56 --atrousAttn 2>&1 |& tee vit_atrous2.txt
python vit-imagenet.py --seed 55 --epoch 30 --patchsize 32 --atrousAttn 2>&1 |& tee vit_atrous3.txt
# python vit-imagenet.py --seed 55 --epoch 30 --patchsize 32  2>&1 |& tee vit_reg2.txt
# python vit-imagenet.py --seed 55 --epoch 30 --patchsize 32 --atrousAttn 2>&1 |& tee vit_atrous2.txt

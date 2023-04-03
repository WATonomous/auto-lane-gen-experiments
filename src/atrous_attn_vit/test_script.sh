python vit-imagenet.py --seed 42 2>&1 |& tee vit_reg1.txt
python vit-imagenet.py --seed 42 --atrousAttn 2>&1 |& tee vit_atrous1.txt
python vit-imagenet.py --seed 11 2>&1 |& tee vit_reg2.txt
python vit-imagenet.py --seed 11 --atrousAttn 2>&1 |& tee vit_atrous2.txt
python vit-imagenet.py --seed 2023 2>&1 |& tee vit_reg3.txt
python vit-imagenet.py --seed 2023 --atrousAttn 2>&1 |& tee vit_atrous3.txt

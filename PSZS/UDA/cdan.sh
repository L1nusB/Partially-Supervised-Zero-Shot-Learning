CUDA_VISIBLE_DEVICES=0 python PSZS/Baseline/cdan.py /net/vid-ssd1/storage/deeplearning/users/lin21700/datasets/CompCars -d CompCarsModel -s w_sv_comb -t svTot -a resnet50 --seed 0 -i 2170 --epochs 20 --log /net/vid-ssd1/storage/deeplearning/users/lin21700/results/CDAN
CUDA_VISIBLE_DEVICES=0 python PSZS/Baseline/cdan.py /net/vid-ssd1/storage/deeplearning/users/lin21700/datasets/CompCars -d CompCarsModel -s w_sv_comb -t svShr -a resnet50 --seed 0 -i 2170 --epochs 20 --log /net/vid-ssd1/storage/deeplearning/users/lin21700/results/CDAN
CUDA_VISIBLE_DEVICES=0 python PSZS/Baseline/cdan.py /net/vid-ssd1/storage/deeplearning/users/lin21700/datasets/CompCars -d CompCarsModel -s w_sv_comb -t svNov -a resnet50 --seed 0 -i 2170 --epochs 20 --log /net/vid-ssd1/storage/deeplearning/users/lin21700/results/CDAN
CUDA_VISIBLE_DEVICES=0 python PSZS/Baseline/cdan.py /net/vid-ssd1/storage/deeplearning/users/lin21700/datasets/CompCars -d CompCarsModel -s wTot -t svTot -a resnet50 --seed 0 -i 2170 --epochs 20 --log /net/vid-ssd1/storage/deeplearning/users/lin21700/results/CDAN
CUDA_VISIBLE_DEVICES=0 python PSZS/Baseline/cdan.py /net/vid-ssd1/storage/deeplearning/users/lin21700/datasets/CompCars -d CompCarsModel -s wTot -t svShr -a resnet50 --seed 0 -i 2170 --epochs 20 --log /net/vid-ssd1/storage/deeplearning/users/lin21700/results/CDAN
CUDA_VISIBLE_DEVICES=0 python PSZS/Baseline/cdan.py /net/vid-ssd1/storage/deeplearning/users/lin21700/datasets/CompCars -d CompCarsModel -s wTot -t svNov -a resnet50 --seed 0 -i 2170 --epochs 20 --log /net/vid-ssd1/storage/deeplearning/users/lin21700/results/CDAN
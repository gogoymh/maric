python3 main.py --datasets skin_cancer --methods maric --vlm llava-7b --batch_size 32 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/skin_cancer_maric_llava-7b.log 2>&1

python3 main.py --datasets skin_cancer --methods maric --vlm llava-13b --batch_size 16 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/skin_cancer_maric_llava-13b.log 2>&1




python3 main.py --datasets weather --methods maric --vlm llava-7b --batch_size 32 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/weather_maric_llava-7b.log 2>&1

python3 main.py --datasets weather --methods maric --vlm llava-13b --batch_size 16 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/weather_maric_llava-13b.log 2>&1




python3 main.py --datasets cifar10 --methods maric --vlm llava-7b --batch_size 32 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/cifar10_maric_llava-7b.log 2>&1

python3 main.py --datasets cifar10 --methods maric --vlm llava-13b --batch_size 16 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/cifar10_maric_llava-13b.log 2>&1




python3 main.py --datasets oodcv --methods maric --vlm llava-7b --batch_size 32 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/oodcv_maric_llava-7b.log 2>&1

python3 main.py --datasets oodcv --methods maric --vlm llava-13b --batch_size 16 --gpu_ids 0 1 2 3 --log_samples 100 > experiments/oodcv_maric_llava-13b.log 2>&1
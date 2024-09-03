#!/bin/bash -l
#SBATCH --chdir /scratch/izar/challier/project-m3-2024-chatbots-r-us/model/quantization/
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 00:30:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

echo STARTING AT `date`
echo IN `pwd`
nvidia-smi

# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq"

python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-None_qt-fp4" --nb_bits 4 --quant_type "fp4" --double "False"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-None_qt-nf4" --nb_bits 4 --quant_type "nf4" --double "False"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-bf16_qt-fp4" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "fp4" --double "False"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-bf16_qt-nf4" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "nf4" --double "False"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-None_qt-fp4_double" --nb_bits 4 --quant_type "fp4" --double "True"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-None_qt-nf4_double" --nb_bits 4 --quant_type "nf4" --double "True"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-bf16_qt-fp4_double" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "fp4" --double "True"
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant4_cd-bf16_qt-nf4_double" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "nf4" --double "True" 
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant8_thresh-6" --nb_bits 8 --int8_threshold 6
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant8_thresh-6" --nb_bits 8 --int8_threshold 4
python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-sciq-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2-sciq_quant8_thresh-6" --nb_bits 8 --int8_threshold 8

# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-None_qt-fp4" --nb_bits 4 --quant_type "fp4" --double "False"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-None_qt-nf4" --nb_bits 4 --quant_type "nf4" --double "False"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-bf16_qt-fp4" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "fp4" --double "False"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-bf16_qt-nf4" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "nf4" --double "False"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-None_qt-fp4_double" --nb_bits 4 --quant_type "fp4" --double "True"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-None_qt-nf4_double" --nb_bits 4 --quant_type "nf4" --double "True"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-bf16_qt-fp4_double" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "fp4" --double "True"
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant4_cd-bf16_qt-nf4_double" --nb_bits 4 --compute_dtype "bfloat16" --quant_type "nf4" --double "True" 
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant8_thresh-6" --nb_bits 8 --int8_threshold 6
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant8_thresh-6" --nb_bits 8 --int8_threshold 4
# python3 -u evaluation_separate_mcqa_quantization.py --model_path "../checkpoints/flan-t5-large_mcqa-ai2-LoRA-merged" --model_name "flan-t5-large_mcqa-LoRA-merged-mcqa-ai2_quant8_thresh-6" --nb_bits 8 --int8_threshold 8

echo FINISHED at `date`
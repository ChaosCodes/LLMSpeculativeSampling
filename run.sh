python byte_llama.py \
    --input "In May, Singapore athletes did the nation proud by hauling home 51 gold, 43 silver and 64 bronze medals from the Southeast Asia (Sea) Games in Cambodia." \
    --target_model_name huggyllama/llama-7b \
    --approx_model_name PY007/ByteLlama-230M-preview \
    --gamma 20 \
    --benchmark -v

import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)
pipe = HiDreamImageEditingPipeline.from_pretrained(
    "HiDream-ai/HiDream-E1-Full",
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16,
)
test_image = Image.open("assets/test_1.png")
test_image = test_image.resize((768, 768))
pipe = pipe.to("cuda", torch.bfloat16)
image = pipe(
    prompt = 'Editing Instruction: Convert the image into a Ghibli style. Target Image Description: A person in a light pink t-shirt with short dark hair, depicted in a Ghibli style against a plain background.',
    negative_prompt = "low resolution, blur",
    image = test_image,
    guidance_scale=5.0,
    image_guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(3),
).images[0]
image.save("output.jpg")

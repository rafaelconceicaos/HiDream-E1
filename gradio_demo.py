import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from instruction_refinement import refine_instruction
import gradio as gr

# Load models
print("Loading models...")
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
pipe = pipe.to("cuda", torch.bfloat16)
print("Models loaded successfully!")

def process_image(input_image, instruction, seed, guidance_scale, image_guidance_scale, steps, refine_instruction_option):
    
    original_width, original_height = input_image.size
    
    # Resize the input image to 768x768
    input_image = input_image.resize((768, 768))
    
    # Refine the instruction if option is selected
    if refine_instruction_option:
        refined_instruction = refine_instruction(src_image=input_image, src_instruction=instruction)
    else:
        refined_instruction = instruction
    
    # Generate image
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=refined_instruction,
        negative_prompt="low resolution, blur",
        image=input_image,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    
    # Resize the result to the original size
    result = result.resize((original_width, original_height))
    
    return result, refined_instruction

# Create Gradio interface
with gr.Blocks(title="HiDream Image Editor") as demo:
    gr.Markdown("# HiDream Image Editing")
    gr.Markdown("Upload an image and provide an instruction to edit it using the HiDream model.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            instruction = gr.Textbox(label="Editing Instruction", placeholder="e.g., convert the image into a Ghibli style")
            refine_instruction_option = gr.Checkbox(label="Refine Instruction", value=True, info="Use VLM to refine your instruction")
            
            with gr.Accordion("Advanced Settings", open=True):
                seed = gr.Slider(minimum=0, maximum=10000, step=1, value=3, label="Seed")
                gr.Markdown("*Note: You might need to try different seeds to get satisfying results.*")
                guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=5.0, label="Instruction Following Strength")
                image_guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Image Preservation Strength")
                gr.Markdown("*Note: For style changes, use higher image preservation strength (e.g., 3.0-4.0). For local edits like adding, deleting, replacing elements, use lower image preservation strength (e.g., 2.0-3.0).*")
                steps = gr.Slider(minimum=10, maximum=50, step=1, value=28, label="Steps")
            submit_btn = gr.Button("Generate")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image", format="png")
            refined_instruction_text = gr.Textbox(label="Refined Instruction", interactive=False)
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, instruction, seed, guidance_scale, image_guidance_scale, steps, refine_instruction_option],
        outputs=[output_image, refined_instruction_text]
    )
    
    gr.Examples(
        examples=[
            ["assets/test_1.png", "convert the image into a Ghibli style",3, 5, 4, 28, True],
            ["assets/test_1.png", "change the image into Disney Pixar style",3, 5, 4, 28, True],
            ["assets/test_1.png", "turn to sketch style",3, 5, 4, 28, True],
            ["assets/test_1.png", "add a sunglasses to the girl",3, 5, 2, 28, True],
            ["assets/test_1.png", "change the background to a sunset",3, 5, 2, 28, True],
            ["assets/test_2.jpg", "convert this image into a ink sketch image",3, 5, 2, 28, True],
            ["assets/test_2.jpg", "add butterfly'",3, 5, 2, 28, True],
            ["assets/test_2.jpg", "remove the wooden sign'",3, 5, 2, 28, True],
        ],
        fn=process_image,
        inputs=[input_image, instruction, seed, guidance_scale, image_guidance_scale, steps, refine_instruction_option],
        outputs=[output_image, refined_instruction_text],
        cache_examples=True,
        cache_mode="lazy",
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

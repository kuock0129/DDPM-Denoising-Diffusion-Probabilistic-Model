import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

repo_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")
pipe = pipe.to("mps")  # Apple GPU (Metal)


prompt = "Iron Man riding a bicycle, highly detailed, beautiful HD quality, digital painting, artstation, smooth, illustration, cinematic lighting" #@param {type:'string'}

image = pipe(prompt).images[0]
image
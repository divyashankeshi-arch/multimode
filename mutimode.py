from transformers import VisionEncoderDecoderModel
from transformers import ViTImageProcessor
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import torch
image_path = r"pictures\images.jpeg"
image=Image.open(image_path).convert("RGB")
plt.imshow(image)
plt.axis("off")
plt.show()
model_name= "nlpconnect/vit-gpt2-image-captioning"
model=VisionEncoderDecoderModel.from_pretrained(model_name)
processor=ViTImageProcessor.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)
device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
pixel_values=processor(images=image, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    outputs_=model.generate(
        pixel_values, 
        max_length=30,
        num_beams=4,
    )
    caption=tokenizer.decode(outputs_ideas[0], skip_special_tokens=True)
    print("Caption:", caption)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(caption)
    plt.axis("off")
    plt.show()

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# device = "cpu" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# # Ensure the tokenizer uses the correct pad token ID
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# with torch.no_grad():
#     encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

#     model_inputs = encodeds.to(device)
#     model.to(device)

#     generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
#     decoded = tokenizer.batch_decode(generated_ids)
#     print(decoded[0])

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
except Exception as e:
    print(f"Failed to load model or tokenizer with ID {model_id}. Error: {e}")
else:
    text = "Hello my name is"
    with torch.no_grad():
        try:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        except Exception as e:
            print(f"Error during text generation or decoding. Error: {e}")

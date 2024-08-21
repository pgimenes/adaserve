"""
This prompt benchmark request dataset with openai compatiable api
The script currently preprocess all request with given tokenizer
and send the request to the server with the given timestamp.

Caveat: all request prompt had to be pre-generated due to prompt length calcualtion

"""

import argparse
import logging
import time
import asyncio
import aiohttp
from transformers import AutoTokenizer
import csv
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

LOG_INTERVAL = 1000
DATASET_PATH = "src/ada/datasets/AzureLLMInferenceTrace_conv_parsed.csv"
PROMPT_PATH = "experiments/prompt.txt"
HOST_URL = "http://localhost:8000/v1"
API_KEY = "test"
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
openai_request_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# preload the tokenizer with auto tokenizer and preload the place holder prompt
tokenizer = AutoTokenizer.from_pretrained(MODEL)
with open(PROMPT_PATH, "r") as f:
    prompt = f.read()

# load the dataset csv file from dataset path
data = []
with open(DATASET_PATH, "r") as f:
    reader = csv.reader(f)
    # discard the header for the first row
    next(reader)
    
    for index, row in enumerate(reader):
        timestamp = float(row[1])
        input_token_length = row[2]
        output_token_length = row[3]
        data.append((index, timestamp, input_token_length, output_token_length))
response_times = [None]*len(data)

def prompt_to_token(tokenizer, prompt):
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
    return tokenized_prompt

def  token_to_prompt(tokenizer, token):
    prompt = tokenizer.decode(token)
    return prompt

def generate_fixed_length_input(token_length, prompt):
    # truncate the given prompt to fixed length prompt
    tokenized_prompt = prompt_to_token(tokenizer, prompt)
    tokenized_prompt = tokenized_prompt[:, :token_length]
    fixed_length_prompt = token_to_prompt(tokenizer, tokenized_prompt)
    return fixed_length_prompt


async def send_request(session, index, timestamp, input_prompt, output_token_length):
    await asyncio.sleep(timestamp)  # Wait until the precise timestamp
    if index % LOG_INTERVAL == 0:
        logger.info(f"Processing request {index}")
    start_time = time.time()
    
    url = HOST_URL + "/chat/completions"
    headers = openai_request_headers
    # the payload is the minimal tested example working on vllm
    payload = {
        "model": MODEL,
        "prompt": input_prompt,
        "max_tokens": output_token_length,
        "temperature": 0.0,
    }

    async with session.post(url, json=payload, headers=headers) as response:
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        result = await response.json()  # Get the response as JSON
        response_times[index] = elapsed_time # Store the response time 
        # check the output token length from the response
        usage = result["usage"]
        completion_tokens = usage["completion_tokens"]
        if completion_tokens != output_token_length:
            logger.error(f"Expected {output_token_length} tokens, got {completion_tokens} tokens")
        else:
            logger.debug(f"Expected {output_token_length} tokens, got {completion_tokens} tokens")
        
async def main():
    logger.info("Starting the benchmark")
    # preprocess all the request with fixed length prompt with tqdm
    fixed_length_prompts = []
    for index, timestamp, input_token_length, output_token_length in tqdm(data):
        fixed_length_prompt = generate_fixed_length_input(input_token_length, prompt)
        fixed_length_prompts.append(fixed_length_prompt)
    
    logger.info("Preprocessed all requests")
    
    logger.info("Sending requests")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, timestamp, input_token_length,output_token_length in data:
            task = send_request(session, index, timestamp, prompt)
            tasks.append(task)
        await asyncio.gather(*tasks)

    logger.info("All requests finished, dumping response times to csv file")
    # dump the response time to csv file
    with open("response_times.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "timestamp", "input_token_length", "output_token_length", "response_time"])
        for index, timestamp, input_token_length, output_token_length in data:
            response_time = response_times[index]
            writer.writerow([index, timestamp, input_token_length, output_token_length, response_time])
    
# Run the main function
asyncio.run(main())
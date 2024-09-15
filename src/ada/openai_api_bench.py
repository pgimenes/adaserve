"""
This prompt benchmark request dataset with openai compatiable api
The script currently preprocess all request with given tokenizer
and send the request to the server with the given timestamp.

Caveat: all request prompt had to be pre-generated due to prompt length calcualtion

"""

import logging
import time
import asyncio
import aiohttp
from transformers import AutoTokenizer
import csv
from tqdm import tqdm
import re


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

LOG_INTERVAL = 1
DATASET_PATH = "datasets/AzureLLMInferenceTrace_conv_parsed.csv"
PROMPT_PATH = "experiments/prompt.txt"
HOST_URL = "http://localhost:8000"
API_KEY = "test"
MODEL = "/home/pedrogimenes/huggingface/unsloth/Meta-Llama-3.1-8B-Instruct"
# MODEL = "/home/pedrogimenes/huggingface/google/gemma-2-2b-it"
# MODEL = "/home/pedrogimenes/huggingface/google/gemma-2-9b-it"
openai_request_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}
MAX_REQUESTS = 100000
timeout = aiohttp.ClientTimeout(
    total=60*60*24,
    connect=60*60*24,
    sock_read=60*60*24,
    sock_connect=60*60*24,
    ceil_threshold=60*60*24,
)

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
        if index >= MAX_REQUESTS:
            break

        timestamp = float(row[1])
        input_token_length = int(row[2])
        output_token_length = int(row[3])
        data.append((index, timestamp, input_token_length, output_token_length))

JCT_LIST = [None] * len(data)
TTFT_LIST = [None] * len(data)
TBT_LIST = [None] * len(data)

def prompt_to_token(tokenizer, prompt):
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
    return tokenized_prompt

def token_to_prompt(tokenizer, token):
    prompt = tokenizer.decode(token[0])
    return prompt

def generate_fixed_length_input(token_length, prompt):
    # truncate the given prompt to fixed length prompt
    tokenized_prompt = prompt_to_token(tokenizer, prompt)
    tokenized_prompt = tokenized_prompt[:, :token_length]
    fixed_length_prompt = token_to_prompt(tokenizer, tokenized_prompt)
    return fixed_length_prompt


async def send_request(session, index, timestamp, input_prompt, output_token_length):
    attempt = 0
    max_tries = 5
    while True:
        try:
            # Only wait for the right timestamp on the first attempt
            if attempt == 0:
                await asyncio.sleep(timestamp)  # Wait until the precise timestamp
            
            if index % LOG_INTERVAL == 0:
                logger.debug(f"Processing request {index}")
            
            payload = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": input_prompt,
                    }
                ],
                "max_tokens": output_token_length,
                "temperature": 0.0,
                "stream": True,
            }

            url = HOST_URL + "/v1/chat/completions"
            headers = openai_request_headers
            start_time = time.time()
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:

                collected_chunks = []
                chunk_times = []
                async for chunk in response.content.iter_any():
                    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
                    chunk_times.append(chunk_time)
                    collected_chunks.append(chunk)  # save the event response

                tbts = []
                for i in range(1, len(chunk_times)):
                    tbts.append(chunk_times[i] - chunk_times[i - 1])
                
                jct = chunk_times[-1]  # job completion time
                ttft = chunk_times[0]

                if len(collected_chunks) == 1:
                    tbt = 0
                    logger.warning(f"Request {index} returned only one chunk although output length should be {output_token_length}.")
                else:
                    tbt = sum(tbts) / len(tbts)  # time between tokens
                

                JCT_LIST[index] = jct
                TTFT_LIST[index] = ttft
                TBT_LIST[index] = tbt

                # Usually this equals output_token_length + 2 due to role chunk at the beginning and DONE chunk at the end
                # Sometimes this equals output_token_length + 1 when the role chunk is merged with the first content chunk
                # Sometimes it's less when many tokens are merged into one chunk
                # assert len(collected_chunks) >= output_token_length, f"Expected {output_token_length} tokens, got {len(collected_chunks)}"

            logger.info(f"Finished request {index} with JCT: {jct}, TTFT: {ttft}, TBT: {tbt}")
            break
        except Exception as e:
            logger.error(f"Error in request {index}. Attempt: {attempt+1}/{max_tries}. Error: {e}")

            if attempt >= max_tries:
                logger.error(f"Failed to send request {index} after {max_tries} attempts. Skipping request.")
                break
            else:
                # Try again...
                attempt += 1

async def main():
    logger.info("Starting the benchmark")
    
    # Preprocess all the request with fixed length prompt with tqdm
    fixed_length_prompts = []
    for index, timestamp, input_token_length, output_token_length in tqdm(data):
        fixed_length_prompt = generate_fixed_length_input(input_token_length, prompt)
        fixed_length_prompts.append(fixed_length_prompt)
    logger.info("Preprocessed all requests")

    # Send requests
    logger.info("Sending requests")
    START_TIME = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for index, timestamp, input_token_length, output_token_length in data:
            task = send_request(session, index, timestamp, fixed_length_prompts[index], output_token_length)
            tasks.append(task)
    
        # Wait for all requests to finish
        await asyncio.gather(*tasks)
    END_TIME = time.time()
    ELAPSED = END_TIME - START_TIME

    # Calculate throughput
    prompt_throughput = sum([input_token_length for _, _, input_token_length, _ in data]) / ELAPSED
    gen_throughput = sum([output_token_length for _, _, _, output_token_length in data]) / ELAPSED

    # Dump the response time to csv file
    logger.info("All requests finished, dumping response times to csv file")
    with open(f"out.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "timestamp", "input_token_length", "output_token_length", "jct", "ttft", "tbt"])

        for index, timestamp, input_token_length, output_token_length in data:
            jct = JCT_LIST[index]
            ttft = TTFT_LIST[index]
            tbt = TBT_LIST[index]

            writer.writerow([index, timestamp, input_token_length, output_token_length, jct, ttft, tbt])

    # calculate average JCT, TTFT, TBT
    jct_avg = sum(JCT_LIST) / len(JCT_LIST)
    ttft_avg = sum(TTFT_LIST) / len(TTFT_LIST)
    tbt_avg = sum(TBT_LIST) / len(TBT_LIST)

    logger.info(f"Total elapsed time: {ELAPSED}")
    logger.info(f"Average JCT: {jct_avg}")
    logger.info(f"Average TTFT: {ttft_avg}")
    logger.info(f"Average TBT: {tbt_avg}")

    logger.info(f"Prompt Throughput: {prompt_throughput}")
    logger.info(f"Generation Throughput: {gen_throughput}")
    
# Run the main function
asyncio.run(main())
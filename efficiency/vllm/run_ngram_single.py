import argparse
import os
import random
import signal
import subprocess
import sys
import time
from itertools import product

import requests


def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"


def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"


def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"


def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"


def check_server_status(base_port):
    """Check if the server is running."""
    server_ready = False
    max_attempts = 300
    attempt = 0
    while not server_ready and attempt < max_attempts:
        try:
            # Try to hit an endpoint provided by the OpenAI API protocol.
            response = requests.get(f"http://localhost:{base_port}/health")
            if response.status_code == 200:
                server_ready = True
                print(g_str("Server is up and running! ") + f"Took {attempt} secs.")
                break
        except requests.exceptions.ConnectionError:
            pass
        attempt += 1
        if attempt % 10 == 0:
            print(
                y_str("Waiting for server to start... ") + f"({attempt}/{max_attempts})"
            )
        time.sleep(1)
    if not server_ready:
        print(r_str("Server did not start in time. Exiting."))

    return server_ready


dataset_datapath_list = [
    ["hf", "livecodebench/code_generation_lite"],
    ["hf", "AI-MO/aimo-validation-aime"],
    ["hf", "HuggingFaceH4/MATH-500"],
    ["hf", "Idavidrein/gpqa"],
]

temperature_list = [0.0, 0.6]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM server and benchmark client."
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=32,
        help="Number of requests to send to the server.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        required=True,
        help="Base port for the server.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name to use.",
    )
    parser.add_argument(
        "--enable-ngram",
        action="store_true",
        help="Enable ngram model.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="Number of speculative tokens.",
    )
    args = parser.parse_args()

    tp, model = args.tp_rank, args.model_name
    base_port = args.base_port
    num_requests = args.num_requests

    # basename of model
    model_basename = os.path.basename(model)
    args.output_dir = f"{args.output_dir}/{model_basename}"
    os.makedirs(args.output_dir, exist_ok=True)

    test_success = 1
    # Run the benchmark vLLM server
    if args.enable_ngram:
        assert args.num_speculative_tokens > 0, "num_speculative_tokens must be > 0"
        spec_config = f"""
        {{
            "model": "ngram",
            "prompt_lookup_max": 7,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": {args.num_speculative_tokens}
        }}
        """
        server_cmd = (
            f"VLLM_USE_V1=1 vllm serve {model} --swap-space 32 --disable-log-requests "
            f"--port {base_port} "
            f"--speculative-config '{spec_config}'"
        )
        server_name = f"{model_basename}_ngram-{args.num_speculative_tokens}"

    else:
        server_cmd = (
            f"VLLM_USE_V1=1 vllm serve {model} --swap-space 32 --disable-log-requests "
            f"--port {base_port}"
        )
        server_name = f"{model_basename}_full"

    print(g_str("Running server command: ") + server_cmd)

    server_stdout = open(
        os.path.join(args.output_dir, f"{server_name}_server.stdout"), "w"
    )
    server = subprocess.Popen(
        server_cmd,
        shell=True,
        stdout=server_stdout,
        stderr=server_stdout,
        preexec_fn=os.setsid,
    )
    print(g_str(f"{server_name} is running with PID: ") + str(server.pid))

    server_status = check_server_status(base_port)
    if not server_status:
        print(r_str("Server failed to start. Exiting."))
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait()
        test_success = 0
        sys.exit(test_success)
    time.sleep(5)

    for data_datapath, temperature in product(dataset_datapath_list, temperature_list):
        dataset, datapath = data_datapath

        dataset_basename = os.path.basename(datapath)
        client_name = f"{server_name}_{dataset_basename}_{num_requests}_{temperature}"
        os.makedirs(os.path.join(args.output_dir, dataset_basename), exist_ok=True)

        client_cmd = (
            f"VLLM_USE_V1=1 "
            f"python3 ../../3rdparty/vllm/benchmarks/benchmark_serving.py "
            f"--port {base_port} "
            f"--model {model} "
            f"--dataset-name {dataset} "
            f"--dataset-path {datapath} "
            f"--num-prompts {num_requests} "
            f"--temperature {temperature} "
            f"--save-result --save-detailed "
            f"--result-dir {args.output_dir}/{dataset_basename} "
            f"--result-filename {client_name}.jsonl "
        )

        client_stdout = open(
            os.path.join(
                f"{args.output_dir}/{dataset_basename}", f"{client_name}_client.stdout"
            ),
            "w",
        )

        print(g_str("Running client command: ") + client_cmd)
        client = subprocess.Popen(
            client_cmd, shell=True, stdout=client_stdout, stderr=client_stdout
        )
        print(g_str(f"{client_name} Client is running with PID: ") + str(client.pid))

        try:
            stdout, stderr = client.communicate()
        except subprocess.TimeoutExpired:
            print(r_str(f"{client_name} Client timed out. Terminating..."))
            client.kill()
            # stdout, stderr = client.communicate()
        print(g_str(f"{client_name} Client finished."))

    # Terminate the server
    print(g_str(f"Terminating server {server_name}..."))
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    server.wait()
    print(g_str(f"{server_name} Server terminated."))

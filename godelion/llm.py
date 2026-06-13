"""
Godelion LLM Client Factory — multi-provider LLM interface.

Supported providers: Anthropic, OpenAI, DeepSeek, OpenRouter,
Bedrock, Vertex AI, Ollama, vLLM, LM Studio, and any
OpenAI-compatible endpoint.
"""
import json
import os
import re
from typing import Any, Optional, Tuple

import anthropic
import backoff
import openai

from godelion.config import config

MAX_OUTPUT_TOKENS = config.get("llm", "api", "max_tokens", default=4096)
MAX_RETRIES = config.get("llm", "api", "max_retries", default=5)
RETRY_MAX_TIME = config.get("llm", "api", "retry_max_time", default=300)

AVAILABLE_LLMS = [
    # Anthropic
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    # OpenAI
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    # OpenRouter
    "llama3.1-405b",
    # Bedrock
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Vertex AI
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    # DeepSeek
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    # Local providers
    "ollama/deepseek-coder-v2",
    "ollama/qwen2.5-coder:32b",
    "vllm/deepseek-coder-v2",
    "lm-studio/local-model",
]


def _get_local_config() -> dict:
    """Get local model configuration from config."""
    return {
        "enabled": config.get("local", "enabled", default=False),
        "provider": config.get("local", "provider", default="ollama"),
        "base_url": config.get("local", "base_url", default="http://localhost:11434/v1"),
        "api_key": config.get("local", "api_key", default="not-needed"),
    }


def create_client(model: str) -> Tuple[Any, str]:
    """
    Create and return an LLM client based on the specified model.

    Supports: Anthropic, OpenAI, DeepSeek, OpenRouter, Bedrock, Vertex AI,
    Ollama, vLLM, LM Studio, and custom OpenAI-compatible endpoints.
    """
    local_cfg = _get_local_config()

    # Local providers (Ollama, vLLM, LM Studio, custom)
    if model.startswith("ollama/") or (local_cfg["enabled"] and local_cfg["provider"] == "ollama" and "ollama" not in model):
        if model.startswith("ollama/"):
            model_name = model.split("/", 1)[1]
        else:
            model_name = local_cfg.get("coding_model", model)
        base_url = local_cfg["base_url"]
        print(f"Using Ollama with model {model_name} at {base_url}")
        return openai.OpenAI(base_url=base_url, api_key=local_cfg["api_key"]), model_name

    if model.startswith("vllm/"):
        model_name = model.split("/", 1)[1]
        base_url = local_cfg.get("base_url", "http://localhost:8000/v1")
        print(f"Using vLLM with model {model_name} at {base_url}")
        return openai.OpenAI(base_url=base_url, api_key=local_cfg["api_key"]), model_name

    if model.startswith("lm-studio/") or model.startswith("lm_studio/"):
        model_name = model.split("/", 1)[1]
        base_url = local_cfg.get("base_url", "http://localhost:1234/v1")
        print(f"Using LM Studio with model {model_name} at {base_url}")
        return openai.OpenAI(base_url=base_url, api_key=local_cfg["api_key"]), model_name

    # Claude via Anthropic API
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model

    # Claude via Bedrock
    if model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
        return client, client_model

    # Claude via Vertex AI
    if model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model

    # OpenAI models
    if 'gpt' in model or model.startswith("o1-") or model.startswith("o3-"):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model

    # DeepSeek
    if model.startswith("deepseek-"):
        print(f"Using DeepSeek API with {model}.")
        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url="https://api.deepseek.com",
        )
        return client, model

    # OpenRouter (Llama, etc.)
    if model == "llama3.1-405b" or model.startswith("llama3.1-"):
        llama_size = model.split("-")[-1]
        actual_model = f"meta-llama/llama-3.1-{llama_size}-instruct"
        print(f"Using OpenRouter with {actual_model}.")
        client = openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
        return client, actual_model

    # Fallback to local if enabled
    if local_cfg["enabled"]:
        base_url = local_cfg["base_url"]
        model_name = local_cfg.get("coding_model", model)
        print(f"Using local provider ({local_cfg['provider']}) with model {model_name} at {base_url}")
        return openai.OpenAI(base_url=base_url, api_key=local_cfg["api_key"]), model_name

    raise ValueError(f"Model {model} not supported. Configure a local model in config.local.yaml or use a supported API model.")


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_time=RETRY_MAX_TIME)
def get_batch_responses_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[list] = None,
    temperature: float = 0.75,
    n_responses: int = 1,
) -> Tuple[list, list]:
    if msg_history is None:
        msg_history = []

    is_openai_compat = not ("claude" in model or model.startswith("bedrock") or model.startswith("vertex_ai"))
    is_o_series = model.startswith("o1-") or model.startswith("o3-")

    if is_o_series:
        new_msg_history = msg_history + [{"role": "user", "content": system_message + msg}]
        response = client.chat.completions.create(
            model=model,
            messages=new_msg_history,
            temperature=1,
            n=n_responses,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [new_msg_history + [{"role": "assistant", "content": c}] for c in content]

    elif is_openai_compat:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_message}, *new_msg_history],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [new_msg_history + [{"role": "assistant", "content": c}] for c in content]

    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(msg, client, model, system_message, msg_history=msg_history, temperature=temperature)
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        _print_debug(content, new_msg_history)

    return content, new_msg_history


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=RETRY_MAX_TIME,
)
def get_response_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[list] = None,
    temperature: float = 0.7,
) -> Tuple[str, list]:
    if msg_history is None:
        msg_history = []

    is_claude = "claude" in model
    is_bedrock = model.startswith("bedrock")
    is_vertex = model.startswith("vertex_ai")
    is_openai = 'gpt' in model or model.startswith("o1-") or model.startswith("o3-")
    is_deepseek = model.startswith("deepseek-")
    is_llama = model.startswith("llama3.1-") or model.startswith("meta-llama")

    # Claude API
    if is_claude or is_bedrock or is_vertex:
        new_msg_history = msg_history + [{"role": "user", "content": [{"type": "text", "text": msg}]}]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [{"role": "assistant", "content": [{"type": "text", "text": content}]}]

    # OpenAI o-series
    elif model.startswith("o1-") or model.startswith("o3-"):
        new_msg_history = msg_history + [{"role": "user", "content": system_message + msg}]
        response = client.chat.completions.create(
            model=model,
            messages=new_msg_history,
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    # OpenAI GPT, DeepSeek, Ollama, vLLM, LM Studio, custom (all OpenAI-compatible)
    elif is_openai or is_deepseek or is_llama or True:  # fallback: try OpenAI-compatible
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": system_message}, *new_msg_history],
            "temperature": temperature,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "n": 1,
            "stop": None,
        }
        if is_openai and not is_deepseek:
            kwargs["seed"] = 0

        try:
            response = client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if "maximum context length" in str(e).lower() or "too long" in str(e).lower():
                # Try truncating context
                truncated = _truncate_context(new_msg_history, system_message)
                kwargs["messages"] = [{"role": "system", "content": system_message}, *truncated]
                response = client.chat.completions.create(**kwargs)
            else:
                raise

        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

        # Capture reasoning content if available (DeepSeek Reasoner, some local models)
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            content = f"[Reasoning]\n{response.choices[0].message.reasoning_content}\n\n[Response]\n{content}"

    if print_debug:
        _print_debug([content], [new_msg_history])

    return content, new_msg_history


def _truncate_context(msg_history: list, system_message: str, max_chars: int = 150000) -> list:
    """Truncate message history to fit within context window."""
    total = len(system_message)
    truncated = []
    for msg in reversed(msg_history):
        msg_len = len(str(msg.get("content", "")))
        if total + msg_len > max_chars:
            truncated.insert(0, {"role": "system", "content": "[Previous messages truncated due to context limit]"})
            break
        truncated.insert(0, msg)
        total += msg_len
    return truncated


def extract_json_between_markers(llm_output: str) -> Optional[dict]:
    """Extract JSON object from LLM output between ```json ... ``` markers or any JSON-like content."""
    if not llm_output:
        return None

    # Try JSON code block first
    inside = False
    json_lines = []
    for line in llm_output.split('\n'):
        stripped = line.strip()
        if stripped.startswith("```json"):
            inside = True
            continue
        if inside and stripped.startswith("```"):
            break
        if inside:
            json_lines.append(line)

    if json_lines:
        json_string = "\n".join(json_lines).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            try:
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                pass

    # Fallback: find any JSON object in the text
    fallback_pattern = r"\{.*?\}"
    for candidate in re.findall(fallback_pattern, llm_output, re.DOTALL):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
            try:
                return json.loads(candidate_clean)
            except json.JSONDecodeError:
                continue

    return None


def _print_debug(content: list, msg_history: list):
    print()
    print("*" * 20 + " LLM START " + "*" * 20)
    for j, msg in enumerate(msg_history[0] if msg_history else []):
        print(f'{j}, {msg["role"]}: {msg["content"]}')
    print(content)
    print("*" * 21 + " LLM END " + "*" * 21)
    print()

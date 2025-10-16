#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM ChatCompletion API advanced client
Provides session management and interactive mode
"""

import argparse
import json
import requests
import time
import openai
import traceback
import os
from typing import Dict, Any, List, Optional


class ChatSession:
    """Chat session class, manages conversation history"""

    def __init__(self, system_message: Optional[str] = None):
        """Initialize chat session"""
        self.messages = []
        if system_message:
            self.add_message("system", system_message)

    def add_message(self, role: str, content: str) -> None:
        """Add message"""
        assert role in ["system", "user", "assistant", "tool"], f"Role must in [system, user, assistant, tool]"
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        """Clear session history"""
        self.messages = []


class OpenAIClient:
    """
    A simple wrapper around the OpenAI Python SDK for chat completions with retry support.
    """
    def __init__(self, api_key: str = None, api_base: str = None, organization: str = None, model: str = "gpt-4o", max_tokens: int = 4096, system_prompt: str = "You are a helpful assistant."):
        """
        Initialize the client and conversation context.

        :param api_key: Your OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        :param organization: (Optional) Your OpenAI organization ID.
        :param model: The default model to use for chat completions.
        :param max_tokens: The maximum number of tokens for each completion.
        :param system_prompt: System-level instruction for the assistant.
        """
        assert api_key, 'Must provide api key'
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        if api_base:
            openai.base_url = api_base

        self.model = model
        self.max_tokens = max_tokens
    def chat_sync(self,
                  system_prompt='You are a helpful assistant.',
                  user_prompt: str = '',
                  model: str = None,
                  max_tokens: int = None,
                  temperature: float = 0.1,
                  return_raw: bool = False):
        """
        Send a message and get a response.

        :param user_prompt: The user's message text.
        :param model: (Optional) Model name to override the default.
        :param max_tokens: (Optional) Max tokens override.
        :param temperature: Sampling temperature.
        :param return_raw: If True, return the full API response.
        :return: A tuple of (reply, full_response) if return_raw, otherwise reply.
        """
        # Append user message
        messages = [
            {
                'content': system_prompt,
                'role': 'system',
            },
            {
                'content': user_prompt,
                'role': 'user',
            },
        ]

        # Select model and max_tokens
        m = model or self.model
        mt = max_tokens or self.max_tokens

        # Call OpenAI API
        from openai import OpenAI

        openai.api_key = os.environ.get('OPENAI_API_KEY', None)
        openai.base_url = os.environ.get('OPENAI_API_BASE', None)

        response = openai.chat.completions.create(
            model=m,
            messages=messages,
            max_tokens=mt,
            temperature=temperature
        )

        # Extract assistant reply
        reply = response.choices[0].message.content

        if return_raw:
            return reply, response
        return reply

    def chat_sync_retry(self,
                        system_prompt = 'You are a helpful assistant.',
                        user_prompt: str = '',
                        model: str = None,
                        max_retry: int = 5,
                        **kwargs):
        """
        Retry chat on failure.

        :param user_prompt: The user's message.
        :param model: (Optional) Model name to override.
        :param max_retry: Number of retry attempts.
        :param kwargs: Additional args for chat_sync.
        :return: A tuple of (reply, full_response) if return_raw, otherwise reply.
        """
        for attempt in range(max_retry):
            try:
                return self.chat_sync(system_prompt, user_prompt, model=model, **kwargs)
            except Exception as e:
                traceback.print_exc()
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None
    ###  new add
    def batch_chat_sync(self, 
                        batch_user_prompts: List[str],
                        system_prompt: str = 'You are a helpful assistant.',
                        model: str = None,
                        max_tokens: int = None,
                        temperature: float = 0.1,
                        max_retry: int = 5) -> List[Optional[str]]:
        """
        Batch chat with retry support.
        
        :param batch_user_prompts: List of user prompts to process.
        :param system_prompt: System prompt for all requests.
        :param model: (Optional) Model name to override the default.
        :param max_tokens: (Optional) Max tokens override.
        :param temperature: Sampling temperature.
        :param max_retry: Number of retry attempts for each request.
        :return: List of responses, None for failed requests.
        """
        # OpenAI API does not support true batch inference, use concurrent requests instead
        return self._batch_chat_sync_concurrent(
            batch_user_prompts, system_prompt, model, max_tokens, temperature, max_retry
        )

    def _batch_chat_sync_concurrent(self, 
                                   batch_user_prompts: List[str],
                                   system_prompt: str = 'You are a helpful assistant.',
                                   model: str = None,
                                   max_tokens: int = None,
                                   temperature: float = 0.1,
                                   max_retry: int = 5) -> List[Optional[str]]:
        """
        Batch chat with concurrent requests for OpenAI API.
        """
        import concurrent.futures
        
        def single_request_with_idx(idx_prompt):
            idx, prompt = idx_prompt
            return idx, self.chat_sync_retry(
                system_prompt=system_prompt,
                user_prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retry=max_retry
            )
        
        results = [None] * len(batch_user_prompts)
        # Use thread pool for concurrent requests, preserve order
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(batch_user_prompts))) as executor:
            future_to_idx = {executor.submit(single_request_with_idx, (i, prompt)): i for i, prompt in enumerate(batch_user_prompts)}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Batch request error: {e}")
                    idx = future_to_idx[future]
                    results[idx] = None
        
        return results

class ChatClient:
    """Chat client class"""

    def __init__(self, server_url: str = "http://localhost:8000", model: str = "Qwen2.5-7B-Instruct"):
        """Initialize client"""
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.session = ChatSession()

    def check_health(self) -> bool:
        """Check server health status"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("status") == "ok" and health_data.get("model_loaded")
            return False
        except Exception:
            return False

    def wait_for_server(self, max_retries: int = 10, retry_interval: int = 2) -> bool:
        """Wait for server to be ready"""
        for i in range(max_retries):
            if self.check_health():
                return True
            print(f"Server is not ready, retry after {retry_interval} seconds ({i + 1}/{max_retries})...")
            time.sleep(retry_interval)
        return False

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048,
             temperature: float = 0, top_p: float = 0.9, top_k: int = 50) -> Optional[Dict[str, Any]]:
        """Send chat request"""
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed: HTTP {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None

    def batch_chat(self, batch_messages: List[List[Dict[str, str]]], max_tokens: int = 2048,
                   temperature: float = 0, top_p: float = 0.9, top_k: int = 50) -> List[Optional[Dict[str, Any]]]:
        """Send batch chat requests - real batch inference"""
        
        # First try vLLM's batch API format
        batch_url = f"{self.server_url}/v1/chat/batch"
        batch_result = self._try_batch_api(batch_url, batch_messages, max_tokens, temperature, top_p, top_k)
        if batch_result:
            return batch_result

        # Fallback to concurrent requests
        print("Batch inference not supported, falling back to concurrent requests")
        return self._batch_chat_concurrent(batch_messages, max_tokens, temperature, top_p, top_k)

    def _try_batch_api(self, url: str, batch_messages: List[List[Dict[str, str]]], max_tokens: int, 
                      temperature: float, top_p: float, top_k: int) -> Optional[List[Optional[Dict[str, Any]]]]:
        """Try using the dedicated batch API"""
        headers = {"Content-Type": "application/json"}
        
        # vLLM batch API format
        batch_requests = []
        for i, messages in enumerate(batch_messages):
            batch_requests.append({
                "id": str(i),
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            })
        
        batch_data = {
            "requests": batch_requests
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(batch_data))
            if response.status_code == 200:
                response_data = response.json()
                
                if "responses" in response_data:
                    # Sort by ID to ensure correct order
                    responses = sorted(response_data["responses"], key=lambda x: int(x.get("id", 0)))
                    results = []
                    
                    for i, resp in enumerate(responses):
                        if "choices" in resp and len(resp["choices"]) > 0:
                            results.append({
                                "choices": resp["choices"],
                                "usage": resp.get("usage", {})
                            })
                        else:
                            results.append(None)
                    
                    return results
        except Exception as e:
            print(f"Batch API request failed: {e}")
        
        return None


    def _batch_chat_concurrent(self, batch_messages: List[List[Dict[str, str]]], max_tokens: int = 2048,
                              temperature: float = 0, top_p: float = 0.9, top_k: int = 50) -> List[Optional[Dict[str, Any]]]:
        """Send batch chat requests - fallback to concurrency"""
        import concurrent.futures
        
        def single_request(idx_messages):
            idx, messages = idx_messages
            return idx, self.chat(messages, max_tokens, temperature, top_p, top_k)
        
        # Use thread pool for concurrent requests, preserve order
        results = [None] * len(batch_messages)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(256, len(batch_messages))) as executor:
            future_to_idx = {executor.submit(single_request, (i, messages)): i for i, messages in enumerate(batch_messages)}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Concurrent request error: {e}")
                    idx = future_to_idx[future]
                    results[idx] = None
        
        return results


    def chat_with_session(self, max_tokens: int = 1024, temperature: float = 0.7,
                          top_p: float = 0.9, top_k: int = 50) -> Optional[Dict[str, Any]]:
        """Send chat request using current session"""
        result = self.chat(self.session.messages, max_tokens, temperature, top_p, top_k)
        if result and result.get("choices"):
            # Add assistant reply to session
            assistant_message = result["choices"][0]["message"]["content"]
            self.session.add_message("assistant", assistant_message)
        return result

    def add_user_message(self, content: str) -> None:
        """Add user message to session"""
        self.session.add_message("user", content)

    def reset_session(self, system_message: Optional[str] = None) -> None:
        """Reset current session"""
        self.session = ChatSession(system_message)


def interactive_mode(client: ChatClient, system_message: Optional[str] = None):
    """Interactive mode"""
    client.reset_session(system_message)
    print("\nInteractive mode, type 'exit' to exit, type 'reset' to reset this chat\n")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "reset":
            new_system = input("Input new system prompt (return to use default): ")
            client.reset_session(new_system or system_message)
            print("Chat is reseted")
            continue

        client.add_user_message(user_input)
        print("\n助手: ", end="", flush=True)
        result = client.chat_with_session()

        if result:
            assistant_message = result["choices"][0]["message"]["content"]
            print(assistant_message)
            print(f"\nToken usage: {result['usage']}")
        else:
            print("Request Failed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="vLLM ChatCompletion API Client")
    parser.add_argument("--server", type=str, default=None, help="Server address, server ip")
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct", help="Model name, model name")
    parser.add_argument("--system", type=str, default=None, help="system prompt")
    parser.add_argument("--user", type=str, help="user prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Create client
    client = ChatClient(server_url=args.server, model=args.model)

    # Check server health status
    print(f"Cheking ready status of {client.server_url}...")
    if not client.wait_for_server():
        print("Server is not ready")
        return

    print("Server is ready")

    if args.interactive:
        # Interactive mode
        interactive_mode(client, args.system)
    else:
        # Single request mode
        if not args.user:
            print("Non-interactive mode must inlcude param: --user")
            return

        print("Dialogue begin...")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.user})

        result = client.chat(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )

        if result:
            assistant_message = result["choices"][0]["message"]["content"]
            print("\nChat result:")
            print("-" * 50)
            print(assistant_message)
            print("-" * 50)
            print(f"Token usage: {result['usage']}")


if __name__ == "__main__":
    main()
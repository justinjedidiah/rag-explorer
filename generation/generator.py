import json
import re


def parse_json_list(raw: str) -> list:
    """Pull the first JSON array out of an LLM reply; [] when there is none."""
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    try:
        value = json.loads(match.group()) if match else []
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


class LLM:
    """One chat-completion interface over Claude and OpenAI that records token usage.

    Every call appends {"label", "input_tokens", "output_tokens"} to self.usage,
    so the UI can show what each step of the pipeline cost.
    """

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.usage: list[dict] = []
        if provider == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, prompt: str, max_tokens: int = 2000, label: str = "LLM call") -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            tokens_in, tokens_out = response.usage.input_tokens, response.usage.output_tokens
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_tokens,
                messages=messages,
            )
            text = response.choices[0].message.content or ""
            tokens_in, tokens_out = response.usage.prompt_tokens, response.usage.completion_tokens

        self.usage.append({"label": label, "input_tokens": tokens_in, "output_tokens": tokens_out})
        return text


def chunk_text_for_llm(chunk: dict) -> str:
    """The text of a chunk that is sent to the LLM: its compressed version if
    contextual compression ran, else the parent paragraph when hierarchical
    chunking is used, else the chunk itself."""
    return chunk.get("compressed_text") or chunk.get("parent_text") or chunk["text"]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i} | Page {chunk['page']}]\n{chunk_text_for_llm(chunk)}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""Answer the question using only the sources below.
Cite sources as [Source N]. If the answer isn't in the sources, say so.

{context}

Question: {question}
Answer:"""


def generate(question: str, chunks: list[dict], llm: LLM, max_tokens: int) -> str:
    return llm.complete(build_prompt(question, chunks), max_tokens=max_tokens, label="Generation")

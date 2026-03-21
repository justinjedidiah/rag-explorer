def generate(question: str, chunks: list[dict], provider: str, model: str, max_tokens: int, api_key: str) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk["parent_text"] if chunk["parent_text"] else chunk['text']
        context_parts.append(f"[Source {i} | Page {chunk['page']}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Answer the question using only the sources below.
Cite sources as [Source N]. If the answer isn't in the sources, say so.

{context}

Question: {question}
Answer:"""

    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # elif provider == "ollama":
    #     import ollama
    #     response = ollama.chat(
    #         model=model,
    #         messages=[{"role": "user", "content": prompt}],
    #         options={
    #             'num_predict': max_tokens
    #         }
    #     )
    #     return response["message"]["content"]

    else:
        raise ValueError(f"Unknown provider: {provider}")

def quick_llm(provider: str, model: str, api_key: str):
    """Returns a callable(prompt) -> str for internal use."""
    def call(prompt: str) -> str:
        if provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        # elif provider == "ollama":
        #     import ollama
        #     response = ollama.chat(
        #         model=model,
        #         messages=[{"role": "user", "content": prompt}]
        #     )
        #     return response["message"]["content"]
    return call
"""Notebook prompts and prompt/completion preparation."""
LORA_PROMPT = """Summarize the conversation in 1-2 concise sentences.
Include the main topic, important decisions and outcomes.
Do not invent information.
Output only the summary."""
ZERO_PROMPT = "Summarize the dialogue in 1-2 concise sentences. Preserve important facts and decisions. Output only the summary."


def messages(dialogue, prompt):
    return [{"role": "system", "content": prompt}, {"role": "user", "content": f"Dialogue:\n{dialogue}"}]


def prepare_dataset(df, tokenizer, prompt):
    from datasets import Dataset
    examples = []
    for row in df.itertuples():
        text = tokenizer.apply_chat_template(messages(row.dialogue, prompt), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        examples.append({"prompt": text, "completion": row.summary.strip() + tokenizer.eos_token})
    return Dataset.from_list(examples)

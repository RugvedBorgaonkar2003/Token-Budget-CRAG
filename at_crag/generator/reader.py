"""
reader.py — Answer generator using google/flan-t5-base.

This module wraps the HuggingFace ``flan-t5-base`` model (~250 MB) for
conditional text generation.  It is the final step in both the baseline
CRAG and AT-CRAG pipelines: given a query and a set of supporting
passages, it produces a short extractive/abstractive answer.
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    """Flan-T5-Base answer generator (CPU-only, greedy decoding).

    Attributes:
        model_name: HuggingFace model identifier.
        tokenizer:  Loaded tokenizer.
        model:      Loaded Seq2Seq model (CPU).
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Load the generator model and tokenizer.

        Args:
            model_name: HuggingFace model ID.  Defaults to
                        ``google/flan-t5-base``.
        """
        self.model_name = model_name
        print(f"[generator] Loading '{model_name}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        print("[generator] Model loaded (CPU).")

    @torch.no_grad()
    def generate_answer(
        self,
        query: str,
        context_passages: List[str],
        max_new_tokens: int = 100,
    ) -> str:
        """Generate an answer from the query and supporting context.

        Prompt template::

            Answer the question based on the context below.
            Context: {joined passages}
            Question: {query}
            Answer:

        Args:
            query:            The user question.
            context_passages: List of passage texts (already filtered by
                              the CRAG evaluator).
            max_new_tokens:   Maximum tokens in the generated answer.

        Returns:
            Decoded answer string.
        """
        context = "\n".join(context_passages)

        prompt = (
            "Answer the question based on the context below.\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy decoding for reproducibility
            num_beams=1,
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = Generator()
    ans = gen.generate_answer(
        query="Who invented the telephone?",
        context_passages=[
            "Alexander Graham Bell is credited with inventing the telephone in 1876.",
            "Bell was born in Edinburgh, Scotland.",
        ],
    )
    print(f"Answer: {ans}")

# conftest.py
# ─────────────────────────────────────────────────────────────
# Shared configuration for all test modules.
# The GroqModel is initialised once here and imported
# by every test file keeping things DRY and consistent.
# ─────────────────────────────────────────────────────────────

import os
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq


class GroqModel(DeepEvalBaseLLM):
    """
    Free LLM wrapper that plugs Groq into DeepEval.
    DeepEval uses this as the judge for all evaluations.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile"
    ):
        self.client = Groq(api_key=api_key)
        self.model = model

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


# Initialise once — imported by all test modules
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_model = GroqModel(api_key=GROQ_API_KEY)

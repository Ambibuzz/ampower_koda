"""The Gemini chat model. Imported only for a Gemini run: Google's SDK is slow to load."""

from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiChat(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI that takes the per-call ``max_tokens`` the other providers take.

    Google's field is ``max_output_tokens``; the adapter fails a call that sends any other name.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

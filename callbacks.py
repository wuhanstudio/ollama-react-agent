
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.base import BaseCallbackManager
from langchain.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
        ) -> Any:
            print(f"Prompts: {prompts[0]}")
            print("*******")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response.generations[0][0].text}")
        print("*******")

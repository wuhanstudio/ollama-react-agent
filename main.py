from dotenv import load_dotenv
from typing import Union, List

from langchain import hub
from langchain_ollama import OllamaLLM
from langchain.agents import tool
from langchain.schema import AgentAction, AgentFinish

from langchain_core.tools import Tool
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser

from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the input text.
    """
    return len(text.strip("'\n").strip('"'))

def find_tool_by_name(name: str, tools: List[Tool]) -> Union[None, tool]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None

if __name__ == '__main__':
    tools = [get_text_length]

    prompt = hub.pull("hwchase17/react")
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join(tool.name for tool in tools),
    )
    callbackAgent = AgentCallbackHandler() # Create callback instance
    llm = OllamaLLM(temperature=0, 
                    model="llama3.2",
                    stop=["\nObservation"],
                    callbacks=[callbackAgent])

    agent = (
        { "input": lambda x: x["input"], 
          "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        }
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    intermediate_steps = []

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of 'DOG' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tool_name, tools)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            # print(f"Observation: {observation}")

            intermediate_steps.append((agent_step, str(observation)))

    print("Final observation:", agent_step.return_values)

    # prompt = input('Enter some text: ')
    # print("The length of the text is:", get_text_length.invoke(input={"text": text}))

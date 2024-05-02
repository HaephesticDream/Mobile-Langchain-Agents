import langchain
from langchain.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent, create_json_agent, create_json_chat_agent, Tool
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain.utilities.python import PythonREPL
import getpass, socket, sys, subprocess, platform, os, multiprocessing, psutil, io
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field, Extra
from langchain.tools import BaseTool
from langchain_core.callbacks.base import BaseCallbackHandler
import langchain

langchain.debug = False

class BaseTool:
    def run(self, tool_input):
        raise NotImplementedError()

    async def arun(self, tool_input):
        raise NotImplementedError()

class SubprocessTool(BaseTool):
    name = "subprocess"
    description = "Execute a command in a subprocess and return the output"

    def run(self, tool_input: Union[str, Dict]) -> str:
        if isinstance(tool_input, dict):
            command = tool_input["command"]
        else:
            command = tool_input

        try:
            output = subprocess.check_output(command, shell=True, universal_newlines=True)
            return output.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"

    async def arun(self, tool_input: Union[str, Dict]) -> str:
        raise NotImplementedError("SubprocessTool does not support async")
    
class PythonExecInput(BaseModel):
    code: str = Field(description="The Python code to execute")

class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "Execute arbitrary Python code"
    args_schema: Type[BaseModel] = PythonExecInput

    def run(self, tool_input: PythonExecInput) -> str:
        # Redirect stdout to a buffer
        buffer = io.StringIO()
        sys.stdout = buffer

        try:
            # Execute the Python code
            exec(tool_input.code)
            output = buffer.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

        return output

    async def arun(self, tool_input: PythonExecInput) -> str:
        raise NotImplementedError("PythonExecTool does not support async")

tools = [
    Tool(
        name="Subprocess Tool",
        description="Executes a command in a subprocess and returns the output",
        func=SubprocessTool,
        input_schema=Union[str, Dict],
        output_schema=str,
    ),
    Tool(
        name="Python Exec Tool",
        description="Executes arbitrary Python code",
        func=PythonExecTool,
        input_schema=PythonExecInput,
        output_schema=str,
    )
]

#llm = Ollama(model="phi3")
llm = OpenAI(base_url="", api_key="None", max_tokens=8000, streaming=True)

username = getpass.getuser()
current_working_directory = os.getcwd()
operating_system = platform.system()

# Additional system and network information
host_name = socket.gethostname()
try:
    ip_address = socket.gethostbyname(host_name)
except socket.gaierror:
    ip_address = "Unavailable"
architecture = platform.machine()
processor = platform.processor()
cpu_count = multiprocessing.cpu_count()
total_memory = psutil.virtual_memory().total / (1024**3)  # Convert from bytes to GB
python_version = sys.version

tool_names = [tool.name for tool in tools]

output_format = """
{
  "workflow": "Generate Prompt and Replicate Image",
  "steps": [
    {
      "name": "Tool name 1",
      "function": "function_runner_1",
      "input_schema": {
        "query": str
      },
      "ouput_key": "result_1"
    },
    {
      "name": "Function 2",
      "function": "function_runner_2",
      "input_schema": {
        "input": "<result_1>"
      },
      "output_key": "result_2"
    }
  ]
}
"""

prompt_template = """


chat_history:
{chat_history}


user_system_details:
Name: {username}
CWD: {current_working_directory}
OS: {operating_system}
Host Name: {host_name}
IP Address: {ip_address}
Architecture: {architecture}
Processor: {processor}
CPU Count: {cpu_count}
Total Memory: {total_memory:.2f} GB
Python Version: {python_version}


Objective:

Your objective is to create a sequential workflow based on the users query.



Create a plan represented in JSON by only using the tools listed below. The workflow should be a JSON array containing only the sequence index, function name and input. A step in the workflow can receive the output from a previous step as input.



Output example 1:

{output_format}



Tools: {tools}

The available tools to call are: {tool_names}

Only answer with the specified JSON format, no other text

HUMAN

{input}



agent scratchpad:

{agent_scratchpad}

"""


prompt = PromptTemplate(
    input_variables=[
        "chat_history", "username", "current_working_directory", "operating_system",
        "host_name", "ip_address", "architecture", "processor", "cpu_count",
        "total_memory:.2f", "python_version", "output_format",  # Include 'output_format' here
        "tools", "input", "agent_scratchpad"
    ],
    template=prompt_template
)

agent = create_json_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

while True:
    user_input = input("Enter your input (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        break

    result = agent_executor.invoke(
    {
        "input": user_input,
        "chat_history": [],
        "username": username,
        "current_working_directory": current_working_directory,
        "operating_system": operating_system,
        "host_name": host_name,
        "ip_address": ip_address,
        "architecture": architecture,
        "processor": processor,
        "cpu_count": cpu_count,
        "total_memory": total_memory,
        "python_version": python_version,
        "output_format": output_format,
        "agent_scratchpad": ""
    }
)
    
    print(result)

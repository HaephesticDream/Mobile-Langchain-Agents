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
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
from langchain.utilities.python import PythonREPL
import getpass, socket, sys, subprocess, platform, os, multiprocessing, psutil, io
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field, Extra
from langchain.tools import BaseTool
from playwright.sync_api import sync_playwright, Page, Response, Browser
from langchain_core.callbacks.base import BaseCallbackHandler
import langchain

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_API_KEY"] = "ls__18fd3bc1dc4146629d6831e0c0fd9ad8"

class SubprocessTool(BaseTool):
    name = "subprocess"
    description = "Execute a command in a subprocess and return the output"

    def _run(self, tool_input: Union[str, Dict]) -> str:
        if isinstance(tool_input, dict):
            command = tool_input["command"]
        else:
            command = tool_input

        try:
            output = subprocess.check_output(command, shell=True, universal_newlines=True)
            return output.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"

    async def _arun(self, tool_input: Union[str, Dict]) -> str:
        raise NotImplementedError("SubprocessTool does not support async")
    
class PythonExecInput(BaseModel):
    code: str = Field(description="The Python code to execute")

class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "Execute arbitrary Python code"
    args_schema: Type[BaseModel] = PythonExecInput

    def _run(self, tool_input: PythonExecInput) -> str:
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

    async def _arun(self, tool_input: PythonExecInput) -> str:
        raise NotImplementedError("PythonExecTool does not support async")
    
subprocess_tool = SubprocessTool()
python_exec_tool = PythonExecTool()

tools = [
    subprocess_tool,
    python_exec_tool,
]

#llm = Ollama(model="phi3")
llm = Ollama(model="phi3")

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

prompt_template = """

You are a helpful cybersecurity assistant

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

Cybersecurity Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
"""



prompt = PromptTemplate(
    input_variables=["chat_history", "username", "current_working_directory", "operating_system", "host_name", "ip_address", "architecture", "processor", "cpu_count", "total_memory:.2f", "python_version", "input", "agent_scratchpad"], template=prompt_template
)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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
        "agent_scratchpad": ""
    }
)
    
    print(result)

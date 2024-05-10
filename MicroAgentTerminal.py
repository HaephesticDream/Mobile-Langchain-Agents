import microchain
from microchain import OpenAITextGenerator, OpenAIChatGenerator, HFChatTemplate, LLM, Agent, Function, Engine
from microchain.functions import Reasoning, Stop
import subprocess, getpass, sys, platform, socket, multiprocessing, psutil, os, shlex


current_working_directory = os.getcwd()
operating_system = platform.system()
# Additional system and network information
architecture = platform.machine()
python_version = sys.version

system_info = f"""

system_details:
CWD: {current_working_directory}
OS: {operating_system}
Architecture: {architecture}
Python Version: {python_version}

"""

generator = OpenAIChatGenerator(
    model="gpt-3.5-turbo",
    api_key="None",
    api_base="https://sheep-beloved-mentally.ngrok.io/v1",
    temperature=0.8
)

llm = LLM(generator=generator)

engine = Engine()

import shlex

class ConvertToCommand(Function):
    @property
    def description(self):
        return "Use this function to convert a user's task into a command for the terminal"

    @property
    def example_args(self):
        return ["ls -l"]

    def __call__(self, task: str):
        try:
            # Replace any occurrences of backticks with single quotes to prevent shell execution
            task = task.replace("`", "'")

            # Split the task into tokens using shlex for proper handling of quoted arguments
            tokens = shlex.split(task)

            # Join the tokens to form the command
            command = ' '.join(tokens)

            return command
        except Exception as e:
            return f"Error converting task to command: {e}"

class SubprocessCommandExecution(Function):
    @property
    def description(self):
        return "Use this function to execute a command using subprocess. Input schema: 'command: str'"

    @property
    def example_args(self):
        return ["ls -l"]

    def __call__(self, command: str):
        try:
            # Parse the command using shlex to handle arguments within quotations
            command_parts = shlex.split(command)
            
            # Join the command parts to ensure proper execution
            command_str = ' '.join(command_parts)
            
            result = subprocess.run(command_str, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Command execution failed with return code {result.returncode}:\n{result.stderr.strip()}"
        except Exception as e:
            return f"Error executing command: {e}"

class PythonExec(Function):
    @property
    def description(self):
        return "Use this function to execute Python code"

    @property
    def example_args(self):
        return ["print('Hello, World!')"]

    def __call__(self, code: str):
        try:
            exec(code)
            return "Code executed successfully"
        except Exception as e:
            return f"Error executing Python code: {e}"

tool_classes = {
    'Reasoning': Reasoning,
    'Stop': Stop,
    'ConvertToCommand': ConvertToCommand,
    'SubprocessCommandExecution': SubprocessCommandExecution,  # Add the SubprocessCommandExecution class here
    'PythonExec': PythonExec  # Add the PythonExec class here
}

for name, cls in tool_classes.items():
    engine.register(cls())

user_input = input("Make a task/query to the Agent:")

agent = Agent(llm=llm, engine=engine)
agent.prompt = f"""You are an Artificially Intelligent Terminal dedicated to converting user natural language input, task, or query into a appropriate terminal command. You can use the following functions:

{engine.help}

Only output valid Python function calls.

Begin!

Terminal Context:
{system_info}

New Input:
{user_input}
"""

agent.bootstrap = [
    'Reasoning("I need to reason step-by-step considering system info and user input")',
]

while True:
    agent.run(15)  # Run the agent for 1 iteration
    user_input = input("Make another task/query to the Agent (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    agent.prompt = f"""You are an Artificially Intelligent Terminal dedicated to converting user natural language input, task, or query into a appropriate terminal command. You can use the following functions:

{engine.help}

Only output valid Python function calls.

Begin!

Terminal Context:
{system_info}

New Input:
{user_input}
"""
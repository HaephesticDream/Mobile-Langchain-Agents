import microchain
from microchain import OpenAITextGenerator, OpenAIChatGenerator, HFChatTemplate, LLM, Agent, Function, Engine, FunctionResult
from microchain.functions import Reasoning, Stop
import subprocess, getpass, os, platform, socket, multiprocessing, psutil, sys, shlex

generator = OpenAIChatGenerator(
    model="gpt-3.5-turbo",
    api_key="None",
    api_base="https://sheep-beloved-mentally.ngrok.io/v1",
    temperature=0.8
)

llm = LLM(generator=generator)

engine = Engine()

class OODAReasoning(Function):
    @property
    def description(self):
        return """Use this function for OODA loop decision-making. Input schema: 'observe: str, orient: str, decide: str, act: str'
                you should fill in the keys with your own str values to steer thought processes."""

    @property
    def example_args(self):
        return [
            "observe='User\'s input was \"hello\"'",
            "orient='Understanding the context'",
            "decide='Possible actions include gathering more details about user needs'",
            "act='Gathering system information'"
        ]

    def __call__(self, observe: str, orient: str, decide: str, act: str):
        try:
            return f"""
            OODA Loop Steps:
            1. Observe: {observe}
            2. Orient: {orient}
            3. Decide: {decide}
            4. Act: {act}
            """
        except Exception as e:
            return f"Error in OODA reasoning: {e}"

class SystemInformation(Function):
    @property
    def description(self):
        return "Use this function to gather system information, call when you need system details for command execution. Try to use this sparingly as it uses up context length."

    @property
    def example_args(self):
        return []

    def __call__(self):
        try:
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

            user_system_info = f"""

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
            """

            return user_system_info.strip()
        except Exception as e:
            return f"Error gathering system information: {e}"

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
        return "Use this function to execute Python code. Input schema: 'code: str'"

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
    'OODAReasoning': OODAReasoning,  # Add the OODA class here
    'SystemInformation': SystemInformation,
    'SubprocessCommandExecution': SubprocessCommandExecution,  # Add the SubprocessCommandExecution class here
    'PythonExec': PythonExec  # Add the PythonExec class here
}

for name, cls in tool_classes.items():
    engine.register(cls())

user_input = input("Make a task/query to the Agent:")

# Initialize chat history as an empty list
chat_history = []

agent = Agent(llm=llm, engine=engine)
agent.prompt = f"""Artificial Intelligence OODA Loop. You can use the following functions:

    {engine.help}

    Only output valid Python function calls.

    If you need to directly respond to user, use python with print statements.

    Your most important tool is the Reasoning tool to steer your thought processes.
    Second most important tool is the OODAReasoning tool to extrapolate quick plans from inputs.

    Begin!

    Chat History:
    {chat_history}

    New Input:
    {user_input}
    """

agent.bootstrap = [
    'SystemInformation()',
    'Reasoning("I need to reason step-by-step considering system information and user input")' ,
    #'OODAReasoning(observe="Observe", orient="Orient", decide="Decide", act="Act")',
]

# Initialize chat history as an empty list
chat_history = []

while True:
    # Run the agent for 50 iterations
    agent.run(15)
    
    # Get the user input
    user_input = input("Make another task/query to the Agent (or 'exit' to quit): ")

    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        break
    
    # Execute the user input and store the response
    result, output = engine.execute(user_input)
    if result == FunctionResult.ERROR:
        print(output)
    else:
        print(output)

    # Store the last agent response manually
    agent_response = chat_history[-1]['content'] if chat_history else ""
    
    # Append the user input, agent response, and tool output to the chat history
    chat_history.append({'role': 'user', 'content': user_input})
    chat_history.append({'role': 'agent', 'content': agent_response})
    chat_history.append({'role': 'tool_output', 'content': output})

    # Update the agent's prompt
    agent.prompt = f"""Artificial Intelligence OODA Loop. You can use the following functions:

    {engine.help}

    Only output valid Python function calls.

    If you need to directly respond to user, use python with print statements.

    Your most important tool is the Reasoning tool to steer your thought processes.
    Second most important tool is the OODAReasoning tool to extrapolate quick plans from inputs.

    Begin!

    Chat History:
    {chat_history}

    New Input:
    {user_input}
    """
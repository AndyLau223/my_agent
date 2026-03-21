from my_agent.tools.web_search import web_search
from my_agent.tools.code_executor import execute_python
from my_agent.tools.file_system import read_file, write_file
from my_agent.tools.http_client import http_get, http_post

ALL_TOOLS = [web_search, execute_python, read_file, write_file, http_get, http_post]

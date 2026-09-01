
from get_weather import *
from get_attraction import *

# 将所有工具函数放入一个字典，方便后续调用

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
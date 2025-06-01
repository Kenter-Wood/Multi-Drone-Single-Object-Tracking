import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(__file__))
def env_settings():
    env_module_name = 'paths'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except Exception as e:
        print(e)
        print('Failed to import all the paths, check paths.py in the same directory')
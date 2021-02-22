import subprocess
import os
from pathlib import Path

base_path = (Path(__file__).parent).resolve()

jar_path = f"{base_path}/simulators/synthea-with-dependencies.jar"
if not os.path.exists(jar_path):
    print("Installing dependencies")
    cmd = f"wget -O {jar_path} https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"
    os.system(cmd)

# Once all dependencies are installed start app
print("Starting Hospital Simulations")
DAEMON_PATH = "./venv/bin/wave" if os.path.isdir("./venv/bin/") else "/resources/venv/bin/wave"
cmd = f"{DAEMON_PATH} run src.app"
os.system(cmd)

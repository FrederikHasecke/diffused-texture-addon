FROM blender:4.2

RUN apt update && apt install -y python3-pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

WORKDIR /addon
COPY . .

# CMD ["blender", "--background", "--python-exit-code", "1", "--python", "tests/test_runner.py"]

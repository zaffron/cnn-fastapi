# Fast API + CNN Model inferencing

### Basic Requirements

- All the requirements for the model is listed on the requirements.txt
- If you want to run locally then please add `tensorflow==2.17.1` on the `requirements.txt`

### Note

- The reason why tensorflow is not included in requirements.txt is because we are already using base docker image for tensorflow for the required version. It takes a lot of time, if fetched form pip

### Usage Requirements

- Make sure docker is installed
- Make sure python `3.10.12` is installed if you are planning to run locally
- Having GPU will greatly improve the inference time

### Usage with docker

- Run `docker compose up` or `docker compose up -d` on the base folder to run
- The API will be available at `http://localhost:8000`
- To view the API docs go to `http://localhost:8000/docs`

### Usage without docker

- Run `pip install -r requirements.txt`
- Run `python main.py` on the base folder
- Other API instructions are same as on the docker usage

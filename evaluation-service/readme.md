# Evaluation service

## Setup

- Update submodules with `git submodule update --init --recursive`

- Run `./setup.sh` to setup

## Backend

Change to the `backend` directory

1. Create a python env with e.g. `conda create -n cloth-competition python=3` and activate it (`conda activate cloth-competition`)
2. Install deps with `pip install -r requirements.txt`
3. Run with `python server.py <path-to-images-folder>` e.g. `python server.py ../images`, which will open the server at http://127.0.0.1:5000

## Frontend

Open a new terminal window and change to the `frontend` directory

1. Install node from https://nodejs.org/en/download
2. Install yarn with
```
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update && sudo apt install yarn
```
3. Run `yarn` to install dependencies
4. Run `yarn start` to start the UI on http://localhost:3000/, the browser will open automatically
6. Select an image from the dropdown and click on a part of it to display a mask
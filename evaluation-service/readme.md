# Evaluation service

## Setup

- Update submodules with `git submodule update --init --recursive`

- Create a python env with e.g. `conda create -n cloth-competition python=3` and activate it (`conda activate cloth-competition`)

- Run `./setup.sh` to setup

## Backend

Change to the backend directory with `cd backend`

Install deps with `pip install -r requirements.txt`

Run with `python server.py <path-to-images-folder>`

## Frontend
1. Install node from https://nodejs.org/en/download
2. Install yarn with 
```
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update && sudo apt install yarn
```
3. Run `yarn` to install deps
4. Run `yarn start` to start the UI
5. Input an image name from `<path-to-images-folder>`
6. Click on the part to segment and wait for the result
![segment stretched shirt](https://i.imgur.com/lyoj63a.png)

# Evaluation service 🏆
The evaluation service will be used at live ICRA 2024 to calculate the participants scores using the result observations.
The organizers will segment the stretched cloth from the image and calculate its coverage, which is the fraction of the maximum possible area that the cloth item could have covered.

The service is divided into two parts:
- **Backend:** a Flask server that uses [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- **Frontend:** a React app that allows the user to select an image and click on a part of it to display a mask

## Usage 📖

Complete the [Installation 🔧](#installation-🔧) first.
### Starting the backend
To start the backend, run the following command in the `backend` directory:
```bash
python server.py ../dataset
```
where you can replace `../dataset` with the path to the directory containing the competition dataset folders.

This will start the server at [http://localhost:5000](http://localhost:5000)

### Starting the frontend
To start the frontend, run the following command in the `frontend` directory:
```bash
yarn start
```
This will start the UI at [http://localhost:3000/](http://localhost:3000/)

There you can select an image from the dropdown and click on a part of it to display a mask

## Installation 🔧


### Backend Installation
First make sure you have a Python env, such as the `cloth-competition-dev` conda environemnt.
Then follow these steps in the `evaluation_service` directory:

1. Install the [Segment-Anything](https://github.com/facebookresearch/segment-anything) repository as a submodule with:
```bash
git submodule update --init --recursive
```
1. Download the SAM-model weights with:
```bash
./download_sam_weights.sh
```
1. Install the requirements with:
```bash
pip install -r backend/requirements.txt
```

### Frontend Installation
Follow these steps in the `frontend` directory:
1. Install `node` (for example with [nvm](https://github.com/nvm-sh/nvm))
2. Install `yarn` (for example with [npm](https://classic.yarnpkg.com/lang/en/docs/install/#debian-stable))
3. Run `yarn` in the `frontend` directory to install the dependencies

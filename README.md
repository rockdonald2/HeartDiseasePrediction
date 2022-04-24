# README

Machine learning based algorithm to diagnose heart disease.

## How to get it to work

1. `git clone` this repo,
2. `docker compose up` to install services,
3. ready to go.

You have a container deployed Jupyter Lab (port `8889`), FastAPI with uvicorn (port `1111`) and NGINX (port `80`).

### Other remarks

- Before trying to use the api in the website, update the API address in the `consts.js` file;
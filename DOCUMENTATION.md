# Documentation

The purpose of this document is to document the commands to be used and reused for the different stages of the project.

## Streamlit

Run app locally

```console
cd ./frontend

streamlit run app.py
```

## Heroku

Login to Heroku

```console
heroku login
```

Run the app locally

```console
heroku local
```

Push the frontend to Heroku

```console
git subtree push --prefix frontend heroku-lingorank-frontend master
```

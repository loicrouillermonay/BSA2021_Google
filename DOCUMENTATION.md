# Documentation

The purpose of this document is to document the commands to be used and reused for the different stages of the project.

## Streamlit (frontend)

Run app locally

```console
cd ./frontend

streamlit run app.py
```

## Heroku (frontend)

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

## Docker (backend)

Build image from Dockerfile

```console
docker build -t lingorank_backend .
```

Run Docker image

```console
docker run -p 4000:80 lingorank_backend
```

Create a tag (version of the backend)

```console
docker tag lingorank_backend YOUR-USERNAME/bsa-backend:v1
```

Push the image to registry

```console
docker push YOUR-USERNAME/bsa-backend:v1
```

## Microsoft Azure - Container Instances (backend)

To host the LingoRank backend. You have to connect to the Azure Portal and create a "Container Instance" with the Docker image that has been uploaded on your personal Dockerhub via the manipulations described in the chapter "Docker" above.

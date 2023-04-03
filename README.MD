# Auto Lane Generation Research
This repository is used as a playground to run auto lane gen experiments. Each of these are containerized through Docker.

**Pre-requisite**: Before you run any of the commands below, make sure you are familiar with [Docker](https://www.docker.com/). If not, we highly recommend going through this [2-hour video](https://www.youtube.com/watch?v=fqMOX6JJhGo) which teaches you the basics of Docker.

## Getting Started
To get started, run the `docker compose up` command, followed by the name of the service you are interested in (if you don't specify, you will end up launching every single service...). You can find all the services we have under [docker-compose.yml](./docker-compose.yml).
```bash
docker compose -p <userId> up <imageName> # ex: docker compose -p jackm up atrous_attn_vit
```

Then, to enter the terminal of the Docker container, open a new terminal and run
```bash
docker exec -it <ContainerID> /bin/bash
```

If you want to develop from the inside container itself, we recommend using VSCode built-in Docker container.

## File Structure
- `docker/` contains a set of custom `Dockerfile` to build our own containers
- `src/` contains the source code for our investigation projects, usually accompanied their own separate `README.md` to explain how the project is being conducted.


#!/usr/bin/env bash

cd "$(dirname "$0")"

docker build -t landmine:v2 .

docker-compose up -d

sleep 1

pipenv run python3 test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
fi

docker-compose down

exit ${ERROR_CODE}
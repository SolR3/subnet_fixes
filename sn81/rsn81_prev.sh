#!/bin/bash

cd ~/.bittensor/subnets/patrol

docker compose -f docker-compose.1_5_2.yml restart db_1_5_2
docker compose -f docker-compose.1_5_2.yml stop validator_1_5_2
docker compose -f docker-compose.1_5_2.yml rm -f validator_1_5_2
docker compose -f docker-compose.1_5_2.yml up --wait validator_1_5_2

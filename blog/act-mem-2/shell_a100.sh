#!/usr/bin/env bash
det shell start -w Garrett --template decoders_notebook\
    --config resources.resource_pool=A100 --include . \
    "$@" -- -L8080:localhost:8080 \


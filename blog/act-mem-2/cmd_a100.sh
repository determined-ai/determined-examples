#!/usr/bin/env bash
det cmd run -w Garrett --template decoders_cache\
    --config resources.resource_pool=A100 -c ../act_mem/ "$@"


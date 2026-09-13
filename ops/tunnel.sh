#!/bin/bash
# SSH tunnel to the AMD droplet's vLLM endpoint, so the model is reachable at
# http://localhost:8000/v1 WITHOUT exposing port 8000 to the internet.
#
# Why: the droplet currently publishes 8000 on its public IP with no auth, so
# anyone who knows the IP can consume ...[truncated]
#!/bin/bash

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

cloud-sql-proxy ${PROJECT_ID}:${REGION}:${INSTANCE_NAME} \
    --credentials-file ${GOOGLE_APPLICATION_CREDENTIALS} \
    --port ${PROXY_PORT}
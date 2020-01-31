#!/usr/bin/env bash
set -x
rsync -av --exclude-from=./exclude ./ ${REMOTE_SERVER}:/data1/${USER}/jarvisrelease/
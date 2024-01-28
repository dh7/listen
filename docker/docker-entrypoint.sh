#!/usr/bin/env bash
# source: https://github.com/tianon/docker-bash/blob/master/docker-entrypoint.sh

set -Eeuo pipefail

# first arg is `-f` or `--some-option`
# or there are no args
if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
	# docker run bash -c 'echo hi'
	exec bash "$@"
fi

exec "$@"
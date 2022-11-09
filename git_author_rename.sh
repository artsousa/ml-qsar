#!/bin/sh

git filter-branch -f --env-filter 'if [ "$GIT_AUTHOR_EMAIL" = "arthursvr" ]; then \
GIT_AUTHOR_EMAIL=arthursvrr@gmail.com; \
GIT_AUTHOR_NAME="Arthur Sousa" \
GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL; \
GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"; fi' -- --all
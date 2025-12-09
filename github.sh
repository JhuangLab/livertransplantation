#!/bin/bash

COMMIT_MSG=${1-"livertransplantation"}
BRANCH=${2-"main"}
# Usage: ./gitpush.sh "Your commit message" "branch-name"
git add .
git commit -m "$COMMIT_MSG"
git push origin $BRANCH

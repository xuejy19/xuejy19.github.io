#!/usr/bin/env sh
set -eu
npm run build
npm run check
npm run deploy

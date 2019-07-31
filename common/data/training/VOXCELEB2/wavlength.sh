#!/bin/bash
find -name '*.wav' -print | while read f ; do duration=$(ffprobe "$f" 2>&1 | awk '/Duration/ { print $2 }'); echo -e $duration"\t"$f ; done | sort -n
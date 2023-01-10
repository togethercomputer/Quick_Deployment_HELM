curl https://computer.together.xyz -d '{ "method": "together_getDepth", "id": 1}' \
  -H 'Content-Type: application/json' | jq '.result'
set -v

black -l 100 .
isort . --atomic -m 3 --profile black
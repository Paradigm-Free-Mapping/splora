.PHONY: all lint

all_tests: lint

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  all_tests		to run all tests"

lint:
	@flake8 splora

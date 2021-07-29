.PHONY: all lint

all_tests: lint unittest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on tedana"
	@echo "  all_tests		to run all tests"

lint:
	@flake8 splora

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=splora splora/

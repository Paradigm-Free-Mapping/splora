.PHONY: all lint

all_tests: lint unittest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on splora"
	@echo "  single-echo	to run the single-echo test set on splora"
	@echo "  multi-echo		to run the multi-echo test set on splora"
	@echo "  all_tests		to run all tests"

lint:
	@flake8 splora

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=splora splora/

single-echo:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=splora -k test_integration_single_echo splora/tests/test_integration.py

multi-echo:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=splora -k test_integration_multi_echo splora/tests/test_integration.py
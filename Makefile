init:
	pip install -r requirements.txt .

editable:
	pip install -r requirements.txt -e .

test: init
	pytest test

coverage: init
	pytest --cov=plenpy --cov-report term-missing

uninstall:
	pip uninstall plenpy

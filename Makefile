.PHONY: prep figures report all

prep:
	python scripts/prep_pipeline.py --lag 5 --thr 10

figures:
	python scripts/build_figures.py

report:
	quarto render report/index.qmd --to html

all: prep figures report

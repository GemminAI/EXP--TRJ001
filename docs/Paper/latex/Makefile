# EXP--TRJ001 conference paper
# Usage:
#   make            # figures + PDF
#   make figures    # generate Figure 1--4
#   make paper      # latexmk (runs pdflatex + bibtex)
#   make clean

PYTHON   ?= python3
LATEXMK  ?= latexmk
TEX      := paper.tex

export MPLCONFIGDIR := $(CURDIR)/.mplconfig

.PHONY: all figures paper clean

all: figures paper

figures:
	@mkdir -p figures .mplconfig
	$(PYTHON) generate_figures.py

paper: figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	$(LATEXMK) -C
	rm -f figures/*.pdf figures/*.png

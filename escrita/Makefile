#!/usr/bin/make

BASEFILE = tesis

.PHONY: $(BASEFILE).pdf

all: pdf

pdf: $(BASEFILE).pdf

$(BASEFILE).pdf: $(BASEFILE).tex
	latexmk -silent -pdf -f $<

clean:
	latexmk -C
	rm -f *.bbl *.spl make.log

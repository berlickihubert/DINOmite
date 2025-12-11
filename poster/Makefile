# Makefile for compiling LaTeX poster

POSTER = poster
LATEX = pdflatex

all: $(POSTER).pdf

$(POSTER).pdf: $(POSTER).tex
	$(LATEX) $(POSTER).tex
	$(LATEX) $(POSTER).tex

clean:
	rm -f $(POSTER).aux $(POSTER).log $(POSTER).out $(POSTER).toc $(POSTER).fdb_latexmk $(POSTER).fls

.PHONY: all clean

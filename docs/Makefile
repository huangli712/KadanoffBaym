MAIN  := main
SRC   := $(wildcard chapters/*.tex appendices/*.tex)
DEPS  := $(MAIN).tex kb.sty $(SRC)

.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(DEPS)
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	rm -f chapters/*.aux appendices/*.aux

distclean: clean
	rm -f $(MAIN).pdf

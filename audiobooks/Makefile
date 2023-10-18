# execute in each subdir
# https://stackoverflow.com/a/17845120

SUBDIRS := $(wildcard */.)

check: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && b2sum -c check.b2sum

.PHONY: check $(SUBDIRS)

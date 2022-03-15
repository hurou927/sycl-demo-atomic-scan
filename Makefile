################################
# C++ Make
#################################

CXX=dpcpp
CXXFLAGS=
CXXLIBS=
CXXSRC=$(wildcard *.cpp)
CXXTARGET=$(basename $(CXXSRC))

.PHONY: all
all:$(CXXTARGET)
$(CXXDEP):%:%.cpp
	$(CXX) -o $* $^ $(CXXFLAGS) $(CXXLIBS)

#################################
# all : force all source code compiler
#################################
.PHONY: force
force: clean $(CXXTARGET)

#################################
# clean : make clean
#################################
.PHONY: clean
clean:
	rm -f $(CXXTARGET)

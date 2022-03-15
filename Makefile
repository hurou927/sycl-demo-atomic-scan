#################################
# C++ Make
#################################

CXX=dpcpp
CXXFLAGS=
CXXLIBS=
CXXSRC=$(wildcard *.cpp)
CXXTARGET=$(basename $(CXXSRC))
CXXDEP=$(addsuffix .dpp,$(CXXTARGET))



.PHONY: all
all:$(CXXDEP) $(CXXTARGET) $(CDEP) $(CTARGET) $(HSTARGET) $(NDEP) $(NTARGET)
#@rm -f $(DEP)
$(CXXDEP):%.dpp:%.cpp
	@$(CXX) -MM $^ $(CXXFLAGS) | sed -e 's/\.o//' > $@
	@printf "\t$(CXX) -o $* $^ $(CXXFLAGS) $(CXXLIBS)\n" >> $@
	$(CXX) -o $* $^ $(CXXFLAGS) $(CXXLIBS)
-include *.dpp

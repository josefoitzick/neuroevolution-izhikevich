CXX = g++
CXXFLAGS = -Wall -std=c++11 -g

# Auto-detecta Python (3.10, 3.12, etc.) usando python3-config.
PYTHON_CFG   ?= python3-config
PYTHON       ?= python3
PY_INCLUDES  := $(shell $(PYTHON_CFG) --includes)
PY_LDFLAGS   := $(shell $(PYTHON_CFG) --ldflags --embed)
NUMPY_INC    := -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")

LIBS = $(PY_LDFLAGS)

SRCDIR = src
BINDIR = bin
HEADERDIR = headers

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

EXECUTABLE = NEAT

all: $(EXECUTABLE)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) $(NUMPY_INC) -I$(HEADERDIR) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LIBS)

$(BINDIR):
	mkdir -p $(BINDIR)

.PHONY: clean Aclean
clean:
	rm -f NEAT
	rm -fr $(BINDIR)
	rm -rf __pycache__
	rm -rf annarchy
	rm -rf results

Aclean:
	rm -rf annarchy

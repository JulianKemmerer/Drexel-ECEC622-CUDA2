# Build script for project

# Add source files here
EXECUTABLE	:= vector_dot_product 
# Cuda source files (compiled with cudacc)
CUFILES_sm_13		:= vector_dot_product.cu
CCFILES		:= \
		   vector_dot_product_gold.cpp


# Rules and targets
# NVCCFLAGS := -arch sm_13
include ../../common/common.mk

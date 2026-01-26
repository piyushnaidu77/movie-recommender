FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for compilation
RUN dnf install -y gcc gcc-c++ python3-devel

# Upgrade pip
RUN pip install --upgrade pip

# Install Cython first (required for scikit-surprise)
RUN pip install --no-cache-dir Cython

# Install numpy first (needed by many packages)
RUN pip install --no-cache-dir numpy

# Install pandas and scikit-learn
RUN pip install --no-cache-dir pandas scikit-learn

# Install scikit-surprise (now that Cython and numpy are available)
RUN pip install --no-cache-dir scikit-surprise

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    mangum==0.17.0 \
    boto3 \
    pyarrow \
    uvicorn \
    pydantic

# Copy application code
COPY app.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["app.handler"]
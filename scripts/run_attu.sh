#!/bin/bash

# 启动attu
docker run -p 8000:3000 -e MILVUS_URL=192.168.3.23:19530 zilliz/attu:v2.5
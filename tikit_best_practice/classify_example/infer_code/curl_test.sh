#!/bin/bash
# 健康检查：检测服务是否Alive
echo "===== test server alive ====="
curl localhost:8501
echo
echo

# 查询服务加载了什么模型算法
echo "===== test model meta ====="
curl localhost:8501/v1/models
echo
echo

# 健康检查：查询模型算法是否ready
echo "===== test model ready ====="
curl localhost:8501/v1/models/m
echo
echo

# 预测请求
echo "===== test predict API ====="
curl -d @test/test_case.json localhost:8501/v1/models/m:predict
echo

echo "===== test predict API ====="
curl localhost:8501/v1/models/m:predict -d'{"image": "https://tione-dev-1256580188.cos.ap-guangzhou.myqcloud.com/image_06738.jpg"}'
echo
echo

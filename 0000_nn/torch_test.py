from transformers import pipeline

# 自动下载模型和权重
classifier = pipeline("sentiment-analysis")

# 推理
result = classifier("I love studying robotics with Hugging Face!")
print(result)
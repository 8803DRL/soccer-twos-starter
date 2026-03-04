import ray

# 初始化 Ray
ray.init()

# 定义一个需要 1 张 GPU 的远端任务
@ray.remote(num_gpus=1)
def check_allocated_gpu():
    # 此时在 Task 内部调用，就会返回被分配的 GPU ID
    return ray.get_gpu_ids()

# 提交任务并获取结果
assigned_gpus = ray.get(check_allocated_gpu.remote())

print(f"分配给 Task 的 GPU IDs: {assigned_gpus}")
print(f"集群总可用资源: {ray.available_resources()}")
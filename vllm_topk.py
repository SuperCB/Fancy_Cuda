import torch
import vllm_topk

base_list=[i for i in range(100)]
data=[base_list for _ in range(10)]
data=torch.Tensor(data).to(dtype=torch.float).cuda()
logits=torch.Tensor(data.shape[0]).to(dtype=torch.float).cuda()
top_ks=[2,2,2,2,2,3,3,3,3,6 ]
top_ps=[0.8]*5+[0.9]*5
vllm_topk.top_k(data,logits,top_ks,top_ps)
print(logits)

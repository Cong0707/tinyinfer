import torch
from torch.profiler import profile, ProfilerActivity
a = torch.randn(256,256,device="cuda"); b = torch.randn(256,256,device="cuda")
def fn():
    torch.mm(a,b); torch.relu(a)
fn(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    fn()
    torch.cuda.synchronize()
print("cpu events", len(list(prof.events())))
for e in list(prof.events())[:10]: print(e.name, e.device_type, e.cpu_time_total)
print("key avgs", [(e.key, e.count) for e in prof.key_averages()][:8])

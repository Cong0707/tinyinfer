# 贡献

## 环境

见 [docs/ENV.md](docs/ENV.md)，Windows 用 [scripts/bootstrap.ps1](scripts/bootstrap.ps1)，
Linux 用 [scripts/bootstrap.sh](scripts/bootstrap.sh)。

自写 kernel 需要 CUDA toolkit 的 `nvcc`：

```powershell
.\scripts\build_native.ps1        # 编译到 src/tinyinfer/native/*.cubin
```

## 改代码前

```powershell
.\scripts\check.ps1              # 语法 + 导入 + 单测
```

## 约定

**正确性先于速度。** 改 kernel 或融合算子时，先在相同输入上与参考实现逐位比对，再测速。
仓库里每个"更快"的数字都伴随一个"输出与参考一致"的验证；只报速度不报一致性不算完成。

**数值差异要写清楚。** 用重排归约换速度时，在提交信息里写明差异量，例如
"state diff 1.5e-8，贪心输出前 20 token 一致"。

**一个模型一条路径。** 新模型接入走 `adapters.py` + `Backend` 协议，不加旁路脚本；
客户端只连端口，模型选择权在启动引擎的人手里。

## 提交

提交信息说明**为什么**这样改，而不是罗列改了什么。实测数字要写明测试条件。

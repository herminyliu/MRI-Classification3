import torch


M = torch.ones(490, requires_grad=True)


if __name__ == "__main__":
    print(M.grad)
    print(M.data)
    # 假设有一些操作，例如：
    output = M * 2

    # 通过output计算梯度
    output.backward(torch.ones_like(output))
    print(M.grad)
    print(M.data)


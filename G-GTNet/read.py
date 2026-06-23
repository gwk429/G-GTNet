import torch
import torch.nn.functional as F


def ce_loss(p, alpha, num_classes, global_step, lambda_epochs):
    # 将标签转换为 one-hot 编码
    label = torch.nn.functional.one_hot(p, num_classes=num_classes)

    # 接下来进行你的损失计算，假设是交叉熵损失
    loss = F.cross_entropy(alpha, label.float())

    return loss


# 示例用法
num_classes = 2
p = torch.tensor([0., 1., 0.]).to(torch.int64)  # 假设这是你的标签张量
alpha = torch.randn(5, num_classes)  # 假设这是你的预测张量

loss = ce_loss(p, alpha, num_classes, global_step=0, lambda_epochs=1.0)
print(loss)
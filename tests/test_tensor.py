import torch

from pos import model


def test_soft_stack():
    a = torch.randn(1, 2, 3)
    b = torch.randn(1, 2)
    c = b.mul(a[:, :, 0])
    for l in range(1, a.shape[-1]):
        c += b.mul(a[:, :, l])


def test_loss():
    criterion = torch.nn.CrossEntropyLoss()

    test_score = torch.Tensor([[1, 2]]).float()
    test_idx = torch.Tensor([0.0]).long()
    loss = criterion(test_score, test_idx)
    expected_loss = 1.31326162815094
    assert loss.item() == expected_loss

    test_score2 = torch.Tensor([[2, 1]]).float()
    test_idx2 = torch.Tensor([1.0]).long()
    loss2 = criterion(test_score2, test_idx2)
    expected_loss2 = 1.31326162815094
    assert loss2.item() == expected_loss2

    # self-made sum
    assert expected_loss + expected_loss2 - 0.01 <= (loss + loss2).item() <= expected_loss + expected_loss2
    # pytorch sum
    test_score_combined = torch.cat((test_score, test_score2))
    test_idx_combined = torch.cat((test_idx, test_idx2))
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = criterion(test_score_combined, test_idx_combined)
    assert loss_sum.eq(loss + loss2)

    # self-made mean
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss_mean = criterion(test_score_combined, test_idx_combined)
    assert loss_mean.eq((loss + loss2) / 2)

    # self-made sum with no reduction
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    loss_none = criterion(test_score_combined, test_idx_combined)
    assert loss_none.sum().eq(loss_sum)
    # self-made mean with no reduction
    assert loss_none.mean().eq(loss_mean)

    # with ignore padding
    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=1)
    loss_ignore = criterion(test_score_combined, test_idx_combined)
    assert loss_ignore.shape[0] == 2
    assert loss_ignore.sum().eq(loss2)

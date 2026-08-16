Static scan of 178 PyTorch repos on 2026-10-24 (318-17456 stars); 148 had training code.

| Problem | Share of repos with training code |
|---|---|
| No mixed precision anywhere | 52% (77/148) |
| Training DataLoader on the main thread | 13% (19/148) |
| At least one of those two | 56% (83/148) |
| Custom training loop that syncs CPU and GPU every step (share of custom loops) | 63% (73/116) |
| Never uses torch.compile | 91% (135/148) |

At least one of the big two: 47% (14/30) of the 30 most-starred, 42% (32/76) of repos pushed since 2025.

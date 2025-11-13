import torch


class StreamingChunkEMAEnsembler:
    def __init__(self, chunk_size: int, stride: int, action_dim: int,
                 decay: float = 0.9, device: torch.device = torch.device("cpu")):
        """
        chunk_size : 1チャンクの長さ
        stride     : チャンク開始ステップのずれ
        action_dim : アクション次元
        decay      : 古いチャンクの寄与係数 (0 < decay < 1)
                     - 古いチャンクほど decay^k で指数減衰
        """
        assert 0.0 < decay < 1.0
        self.chunk_size = chunk_size
        self.stride = stride
        self.action_dim = action_dim
        self.decay = decay
        self.device = device

        # ここには常に「指数重みつき平均済みのアクション」が入る
        self.ensemble = torch.zeros(0, action_dim, device=device)  # (H, action_dim)

        # どの時間indexが一度でも観測済みか（未観測はそのままだと0なので区別するため）
        self.initialized = torch.zeros(0, dtype=torch.bool, device=device)

        self.num_chunks = 0  # 何個目のチャンクまで来たか

    @property
    def horizon(self) -> int:
        return self.ensemble.size(0)

    def _ensure_length(self, needed_H: int):
        """必要な長さ H になるように末尾にゼロパディングする"""
        if needed_H <= self.horizon:
            return
        pad_len = needed_H - self.horizon
        pad_ens = torch.zeros(pad_len, self.action_dim, device=self.device)
        pad_init = torch.zeros(pad_len, dtype=torch.bool, device=self.device)
        self.ensemble = torch.cat([self.ensemble, pad_ens], dim=0)
        self.initialized = torch.cat([self.initialized, pad_init], dim=0)

    def update(self, action_chunk: torch.Tensor):
        """
        新しい action_chunk を受け取って、逐次的に ensemble を更新する。

        action_chunk: (chunk_size, action_dim)
        """
        assert action_chunk.shape == (self.chunk_size, self.action_dim)

        # 何本目のチャンクか
        chunk_idx = self.num_chunks

        # このチャンクがカバーする時間区間 [start, end)
        start = chunk_idx * self.stride
        end = start + self.chunk_size
        needed_H = max(self.horizon, end)

        # 長さを必要分だけ拡張
        self._ensure_length(needed_H)

        # 既存のensembleを取り出し
        old = self.ensemble[start:end]          # (chunk_size, action_dim)
        init = self.initialized[start:end]      # (chunk_size,)

        new = torch.from_numpy(action_chunk).to(self.device)
        
        # --- まだ一度も値が入っていない時間indexは、そのまま new を採用 ---
        # mask: すでに一度以上更新された index
        mask = init

        # 初期化されていない地点
        not_mask = ~mask
        if not_mask.any():
            self.ensemble[start:end][not_mask] = new[not_mask]
            self.initialized[start:end][not_mask] = True

        # --- すでに値がある地点は EMA 更新 ---
        if mask.any():
            old_val = old[mask]
            new_val = new[mask]
            updated = self.decay * old_val + (1.0 - self.decay) * new_val
            self.ensemble[start:end][mask] = updated

        self.num_chunks += 1

    def get_ensemble_actions(self) -> torch.Tensor:
        """
        現時点での指数移動平均されたアクション列を返す。

        戻り値:
            ensemble: (H, action_dim)
        """
        return self.ensemble

    def get_current_action(self, t: int = 0) -> torch.Tensor:
        """
        特定時刻 t のアクションを取り出す（例: t=0 で「今すぐの行動」など）。

        戻り値:
            action: (action_dim,)
        """
        if t >= self.ensemble.size(0):
            raise IndexError(
                f"requested time t={t} exceeds horizon={self.ensemble.size(0)}"
            )
        return self.ensemble[t]



# == 簡単な使用例 == #
def test_streaming_chunk_ema_ensembler():
    class DummyPolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, action_dim),
            )

        def forward(self, obs_chunk: torch.Tensor) -> torch.Tensor:
            # obs_chunk: (L, obs_dim)
            return self.net(obs_chunk)

    obs_dim = 16
    action_dim = 7
    chunk_size = 16
    stride = 4
    decay = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DummyPolicy(obs_dim, action_dim).to(device)
    ensembler = StreamingChunkEMAEnsembler(
        chunk_size=chunk_size,
        stride=stride,
        action_dim=action_dim,
        decay=decay,
        device=device,
    )

    T_total = 200
    obs_stream = torch.randn(T_total, obs_dim, device=device)

    t = 0
    while t + chunk_size <= T_total:
        obs_chunk = obs_stream[t:t+chunk_size]
        with torch.no_grad():
            action_chunk = policy(obs_chunk)  # (chunk_size, action_dim)

        ensembler.update(action_chunk)

        # 「今」のアクション（t=0起点としての1ステップ目）を取り出す例
        current_action = ensembler.get_current_action(t=0)
        # robot.apply_action(current_action) など

        t += stride

    full_ensemble = ensembler.get_ensemble_actions()
    print(full_ensemble.shape)  # -> (H, action_dim)

    return ensembler
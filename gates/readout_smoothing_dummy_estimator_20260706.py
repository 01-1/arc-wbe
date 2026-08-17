class Estimator:
    """Unused compatibility stub for the Fly bank downloader interface."""

    def predict(self, mlp, budget):
        raise RuntimeError("readout-smoothing gate entrypoint must not call Estimator.predict")

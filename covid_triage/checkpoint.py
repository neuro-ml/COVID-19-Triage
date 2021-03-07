import shutil
from dpipe.train import Checkpoints


class CheckpointWithBestMetrics(Checkpoints):
    """
    Saves the checkpoint with the best validation metrics to ``path``.
    Parameters
    ----------
    path: str
        path to save/restore checkpoint object in/from.
    objects: Dict[PathLike, Any]
        objects to save. Each key-value pair represents
        the path relative to ``base_path`` and the corresponding object.
    are_better: callable
        are_better(new_metrics, current_best_metrics) -> bool.
        True if new_metrics are better then current best metrics
    """
    def __init__(self, base_path, objects, are_better):
        super().__init__(base_path, objects)
        self.best_metrics = None
        self.are_better = are_better

    def save(self, iteration, train_losses, metrics):
        super().save(iteration, train_losses, metrics)

        if self.best_metrics is None or self.are_better(metrics, self.best_metrics):
            self.best_metrics = metrics

            path = self.base_path / 'best'
            if path.exists():
                shutil.rmtree(str(path))
            path.mkdir(parents=True)
            self._save_to(path)

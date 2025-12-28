"""Helper functions for MLflow tasks."""

import mlflow


class MLFlowHelper:
    """Helper functions for MLflow tasks."""

    @classmethod
    def get_or_create_experiment(cls, experiment_name: str) -> str:
        """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

        This function checks if an experiment with the given name exists within MLflow.
        If it does, the function returns its ID. If not, it creates a new experiment
        with the provided name and returns its ID.

        :param experiment_name: Name of the MLflow experiment.
        :return: ID of the existing or newly created MLflow experiment.
        """
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        return mlflow.create_experiment(experiment_name)

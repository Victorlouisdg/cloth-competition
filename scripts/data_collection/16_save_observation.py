from pathlib import Path

from cloth_tools.dataset.bookkeeping import datetime_for_filename, ensure_dataset_dir
from cloth_tools.dataset.collection import collect_observation
from cloth_tools.dataset.format import save_competition_observation
from cloth_tools.stations.competition_station import CompetitionStation

if __name__ == "__main__":
    station = CompetitionStation()
    dataset_dir = Path(ensure_dataset_dir("notebooks/data/extra_observations_0000"))

    sample_id = datetime_for_filename()

    sample_dir = dataset_dir / f"sample_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=False)

    observation_result_dir = str(sample_dir / "observation_result")

    observation_result = collect_observation(station)
    save_competition_observation(observation_result, observation_result_dir)
